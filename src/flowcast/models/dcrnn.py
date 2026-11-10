"""DCRNN: Diffusion Convolutional Recurrent Neural Network (Li et al., ICLR 2018).

The idea in two parts:

* **Diffusion convolution.** Traffic spreads along roads, and roads are directed: a jam moves
  upstream, against the flow. For each sensor the layer mixes its own signal with the signals
  of sensors one and two hops away along the road graph, separately for the downstream and the
  upstream direction (a random walk on the adjacency, and on its transpose). Weights are learned
  per hop and per direction, and shared by all sensors.
* **Recurrent encoder-decoder.** A GRU whose matrix multiplications are replaced by diffusion
  convolutions. The encoder reads the last hour (12 steps); the decoder then produces the next
  12 steps one at a time, each feeding the next.

Departures from the paper, all deliberate:

* Missing readings are NaN here, not 0. Inputs fill them with the training mean and add a
  missing-flag channel, so the model can tell "no reading" from "average traffic". The loss
  ignores missing targets (masked MAE), as the paper does.
* Time features of the target step (time of day, weekend) are given to the decoder. They are
  known in advance, so this is not leakage.
* Diffusion steps are true powers ``P^k x`` (the paper's definition), not the reference code's
  Chebyshev-style recursion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from flowcast.features.build import STEPS_PER_DAY

N_IN = 12  # input steps: the last hour
N_OUT = 12  # output steps: the next hour, every 5 minutes
COVARIATES = ("tod_sin", "tod_cos", "is_weekend")


@dataclass(frozen=True)
class Config:
    hidden: int = 64
    layers: int = 2
    diffusion_steps: int = 2  # K: hops along the road graph per direction
    n_covariates: int = len(COVARIATES)

    @property
    def n_inputs(self) -> int:
        return 2 + self.n_covariates  # speed (standardised), missing flag, covariates

    def to_dict(self) -> dict:
        return asdict(self)


# --- data -------------------------------------------------------------------------------------


def random_walk_supports(adjacency: np.ndarray) -> np.ndarray:
    """Forward and backward transition matrices, oriented so ``support @ x`` aggregates neighbours.

    Row-normalising ``A`` gives the probability of stepping from i to j. Rows of an isolated
    sensor are all zero and stay zero (it only sees itself, through the identity term).
    """

    def walk(a: np.ndarray) -> np.ndarray:
        deg = a.sum(axis=1, keepdims=True)
        return np.divide(a, deg, out=np.zeros_like(a), where=deg > 0)

    a = adjacency.astype(np.float64)
    return np.stack([walk(a).T, walk(a.T).T]).astype(np.float32)


def time_covariates(index: pd.DatetimeIndex) -> np.ndarray:
    """(T, 3): time of day on the unit circle, and a weekend flag."""
    slot = (index.hour * 60 + index.minute).to_numpy() / (24 * 60)
    angle = 2 * np.pi * slot
    weekend = (index.day_of_week >= 5).astype(np.float64)
    out = np.stack([np.sin(angle), np.cos(angle), weekend], axis=1)
    assert out.shape[1] == len(COVARIATES) and STEPS_PER_DAY == 288
    return out.astype(np.float32)


@dataclass
class Arrays:
    """Whole-timeline arrays; windows are sliced from them on the fly (no 12x copies in memory)."""

    inputs: np.ndarray  # (T, N, 2): standardised speed with NaN -> 0, missing flag
    covariates: np.ndarray  # (T, 3)
    target: np.ndarray  # (T, N): speed in mph, NaN where missing
    mean: float
    std: float


def build_arrays(speed: pd.DataFrame, mean: float, std: float) -> Arrays:
    x = speed.to_numpy(np.float32)
    missing = np.isnan(x)
    z = np.where(missing, 0.0, (x - mean) / std).astype(np.float32)
    return Arrays(
        inputs=np.stack([z, missing.astype(np.float32)], axis=-1),
        covariates=time_covariates(pd.DatetimeIndex(speed.index)),
        target=x,
        mean=mean,
        std=std,
    )


def standardisation(train_speed: pd.DataFrame) -> tuple[float, float]:
    """Mean and std of the valid training readings (missing ones excluded)."""
    v = train_speed.to_numpy(np.float64)
    v = v[~np.isnan(v)]
    return float(v.mean()), float(v.std())


def windows(arr: Arrays, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For as-of indices ``t``: encoder inputs (B, 12, N, F) from t-11..t, decoder covariates
    (B, 12, N, 3) and targets (B, 12, N) for t+1..t+12. Nothing after t enters the inputs."""
    past = t[:, None] + np.arange(-N_IN + 1, 1)
    future = t[:, None] + np.arange(1, N_OUT + 1)
    n = arr.inputs.shape[1]
    cov_past = np.broadcast_to(arr.covariates[past][:, :, None, :], (*past.shape, n, arr.covariates.shape[1]))
    x = np.concatenate([arr.inputs[past], cov_past], axis=-1)
    cov_future = np.broadcast_to(
        arr.covariates[future][:, :, None, :], (*future.shape, n, arr.covariates.shape[1])
    )
    return x, np.ascontiguousarray(cov_future), arr.target[future]


# --- model ------------------------------------------------------------------------------------


class DiffusionConv(nn.Module):
    """``out = sum_k  [x, P_f x, P_f^2 x, P_b x, P_b^2 x] W``: one weight block per hop and direction."""

    def __init__(self, supports: torch.Tensor, steps: int, in_dim: int, out_dim: int, bias: float = 0.0):
        super().__init__()
        self.register_buffer("supports", supports, persistent=False)
        self.steps = steps
        n_terms = 1 + supports.shape[0] * steps
        self.linear = nn.Linear(in_dim * n_terms, out_dim)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N, in_dim)
        terms = [x]
        for s in self.supports.to(x.dtype):
            h = x
            for _ in range(self.steps):  # P^k x, one hop at a time
                h = torch.einsum("nm,bmf->bnf", s, h)
                terms.append(h)
        return self.linear(torch.cat(terms, dim=-1))


class DCGRUCell(nn.Module):
    """A GRU cell whose weight multiplications are diffusion convolutions over the road graph."""

    def __init__(self, supports: torch.Tensor, steps: int, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        # bias 1.0 on the gates starts the cell close to "keep the old state", which stabilises training
        self.gates = DiffusionConv(supports, steps, in_dim + hidden, 2 * hidden, bias=1.0)
        self.candidate = DiffusionConv(supports, steps, in_dim + hidden, hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], dim=-1))).chunk(2, dim=-1)
        c = torch.tanh(self.candidate(torch.cat([x, r * h], dim=-1)))
        return u * h + (1.0 - u) * c


class DCRNN(nn.Module):
    def __init__(self, supports: np.ndarray, cfg: Config):
        super().__init__()
        self.cfg = cfg
        s = torch.from_numpy(supports)

        def stack(first_in: int) -> nn.ModuleList:
            dims = [first_in] + [cfg.hidden] * (cfg.layers - 1)
            return nn.ModuleList(DCGRUCell(s, cfg.diffusion_steps, d, cfg.hidden) for d in dims)

        self.encoder = stack(cfg.n_inputs)
        self.decoder = stack(1 + cfg.n_covariates)  # previous speed + covariates of the target step
        self.project = nn.Linear(cfg.hidden, 1)

    def forward(
        self,
        x: torch.Tensor,  # (B, 12, N, F) encoder inputs
        cov: torch.Tensor,  # (B, 12, N, C) decoder covariates
        teacher: torch.Tensor | None = None,  # (B, 12, N) standardised targets, NaN where missing
        teacher_prob: float = 0.0,
    ) -> torch.Tensor:
        """Standardised predictions (B, 12, N)."""
        b, _, n, _ = x.shape
        h = [x.new_zeros(b, n, self.cfg.hidden) for _ in self.encoder]
        for t in range(x.shape[1]):
            inp = x[:, t]
            for i, cell in enumerate(self.encoder):
                h[i] = cell(inp, h[i])
                inp = h[i]

        prev = x.new_zeros(b, n, 1)  # "go" symbol
        out = []
        for t in range(cov.shape[1]):
            inp = torch.cat([prev, cov[:, t]], dim=-1)
            for i, cell in enumerate(self.decoder):
                h[i] = cell(inp, h[i])
                inp = h[i]
            y = self.project(inp)  # (B, N, 1)
            out.append(y.squeeze(-1))
            prev = y
            if teacher is not None and teacher_prob > 0 and torch.rand(()) < teacher_prob:
                truth = teacher[:, t, :, None]
                # scheduled sampling: feed the true value where it exists, the prediction elsewhere
                prev = torch.where(torch.isnan(truth), y, truth.to(y.dtype))
        return torch.stack(out, dim=1)


def masked_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over the entries whose target exists (missing targets excluded)."""
    ok = ~torch.isnan(target)
    return (pred[ok] - target[ok]).abs().mean()


def teacher_probability(step: int, decay: float = 2000.0) -> float:
    """Inverse sigmoid decay (Bengio et al., 2015): mostly true inputs early, mostly own predictions late."""
    return decay / (decay + np.exp(min(step / decay, 50.0)))

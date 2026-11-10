"""Tests for DCRNN building blocks (synthetic data, CPU; no dataset or GPU needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from flowcast.models.dcrnn import (  # noqa: E402
    DCRNN,
    N_IN,
    N_OUT,
    Config,
    DiffusionConv,
    build_arrays,
    masked_mae,
    random_walk_supports,
    teacher_probability,
    windows,
)


def chain(n: int) -> np.ndarray:
    """Directed road i -> i+1, plus one isolated last sensor."""
    a = np.zeros((n, n), np.float32)
    for i in range(n - 2):
        a[i, i + 1] = 0.5 + i
    return a


def test_supports_are_random_walks_and_isolated_sensors_stay_zero():
    s = random_walk_supports(chain(6))
    assert s.shape == (2, 6, 6)
    walk = s[0].T  # row-normalised transitions
    assert np.allclose(walk[:4].sum(axis=1), 1.0)
    assert np.all(s[:, 5, :] == 0) and np.all(s[:, :, 5] == 0)


def test_diffusion_reaches_exactly_k_hops():
    n = 8
    conv = DiffusionConv(torch.from_numpy(random_walk_supports(chain(n))), steps=2, in_dim=1, out_dim=1)
    x = torch.zeros(1, n, 1)
    x[0, 0, 0] = 1.0  # impulse at sensor 0
    base = conv(torch.zeros(1, n, 1))
    moved = (conv(x) - base).abs().squeeze()
    assert moved[:3].min() > 0  # itself and 2 hops downstream
    assert torch.all(moved[3:] == 0)  # nothing beyond K = 2 hops


def frame(t: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    values = rng.uniform(20, 70, (t, n))
    values[rng.random((t, n)) < 0.1] = np.nan
    return pd.DataFrame(values, index=pd.date_range("2012-03-01", periods=t, freq="5min"))


def test_windows_never_contain_the_future():
    speed = frame(60, 3)
    arr = build_arrays(speed, 50.0, 10.0)
    t = np.array([20, 30])
    x, cov, y = windows(arr, t)
    assert x.shape == (2, N_IN, 3, 5) and cov.shape == (2, N_OUT, 3, 3) and y.shape == (2, N_OUT, 3)

    changed = speed.copy()
    changed.iloc[21:] = 1.0  # rewrite everything after the first as-of time
    x2, _, y2 = windows(build_arrays(changed, 50.0, 10.0), t)
    np.testing.assert_array_equal(x2[0], x[0])  # inputs at t=20 are unaffected
    assert not np.allclose(np.nan_to_num(y2[0]), np.nan_to_num(y[0]))
    np.testing.assert_array_equal(y[0], speed.to_numpy(np.float32)[21:33])


def test_missing_inputs_are_filled_and_flagged():
    speed = frame(30, 2)
    arr = build_arrays(speed, 50.0, 10.0)
    missing = speed.isna().to_numpy()
    assert np.all(arr.inputs[..., 0][missing] == 0) and np.all(arr.inputs[..., 1][missing] == 1)
    assert not np.isnan(arr.inputs).any()


def test_masked_mae_ignores_missing_targets():
    pred = torch.tensor([1.0, 2.0, 100.0])
    target = torch.tensor([2.0, 2.0, float("nan")])
    assert masked_mae(pred, target).item() == pytest.approx(0.5)


def test_teacher_probability_decays_from_one_to_zero():
    assert teacher_probability(0) > 0.99 and teacher_probability(40_000) < 0.01
    assert teacher_probability(1000) > teacher_probability(20_000)


def test_model_shapes_and_one_step_of_learning():
    torch.manual_seed(0)
    speed = frame(80, 4)
    arr = build_arrays(speed, 50.0, 10.0)
    model = DCRNN(random_walk_supports(chain(4)), Config(hidden=8))
    x, cov, y = (torch.from_numpy(np.ascontiguousarray(a)) for a in windows(arr, np.arange(11, 60)))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(2):
        loss = masked_mae(model(x, cov) * arr.std + arr.mean, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert model(x, cov).shape == (49, N_OUT, 4)
    assert losses[1] < losses[0]

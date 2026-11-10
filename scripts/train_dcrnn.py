"""Train DCRNN on the GPU and score validation and test once.

Usage:
  python scripts/train_dcrnn.py            full run (after scripts/build_features.py)
  python scripts/train_dcrnn.py --timing   time a few batches and exit; writes nothing
  python scripts/train_dcrnn.py --resume   continue an interrupted run from its last finished epoch

Settings are the published DCRNN ones for METR-LA (hidden 64, 2 layers, 2 diffusion steps,
Adam lr 0.01 with step decay, scheduled sampling), not tuned here. Early stopping on validation
MAE over all 12 steps. The test period is scored once, after training, with nothing left to change.

Training windows are visited in random order each epoch. That is not the "shuffled time series"
leak: each window holds only its own past (inputs) and future (targets), all inside the training
period. What must never be shuffled is the split itself, and it is not.

The full training state (weights, optimiser, schedules, random generators, history, best weights)
is saved after every epoch, so a crash or kill loses at most the epoch in progress.

Writes:
  models/checkpoints/dcrnn/last.pt                   resumable training state (git-ignored)
  models/checkpoints/dcrnn/model.pt                  weights + normalisation (git-ignored)
  data/processed/predictions/dcrnn_{split}.parquet   per-row forecasts (git-ignored)
  models/metrics/dcrnn.json                          metrics, slices, learning curve
  docs/model_card_dcrnn.md                           generated model card
  docs/figures/dcrnn/learning_curve.png
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from flowcast import paths  # noqa: E402
from flowcast.data.processed import load_adjacency, load_speed, owned_period  # noqa: E402
from flowcast.evaluation.rows import EvalRows, load_eval_rows  # noqa: E402
from flowcast.features.build import HORIZONS, row_times  # noqa: E402
from flowcast.models.dcrnn import (  # noqa: E402
    DCRNN,
    N_OUT,
    Config,
    build_arrays,
    masked_mae,
    random_walk_supports,
    standardisation,
    teacher_probability,
    windows,
)
from flowcast.reporting import (  # noqa: E402
    COMPARISON,
    FORECAST,
    MODEL_FIGURE_STYLE,
    git_commit,
    peak_memory_mb,
    print_report,
)

ROOT = paths.PROJECT_ROOT
BASELINES = ROOT / "models" / "metrics" / "baselines.json"
LIGHTGBM = ROOT / "models" / "metrics" / "lightgbm.json"
METRICS = ROOT / "models" / "metrics" / "dcrnn.json"
CHECKPOINT = ROOT / "models" / "checkpoints" / "dcrnn" / "model.pt"
RESUME = CHECKPOINT.with_name("last.pt")
PREDICTIONS = paths.PROCESSED_DIR / "predictions"
CARD = ROOT / "docs" / "model_card_dcrnn.md"
FIGURES = ROOT / "docs" / "figures" / "dcrnn"

BATCH, EVAL_BATCH = 64, 128
LR, ADAM_EPS, MILESTONES, GAMMA = 0.01, 1e-3, (20, 30, 40, 50), 0.1
MAX_EPOCHS, PATIENCE, CLIP_NORM = 100, 10, 5.0
CL_DECAY = 2000.0  # scheduled-sampling decay, in batches
FLOOR = "ar_residual"


def to_device(*arrays: np.ndarray, device: torch.device) -> list[torch.Tensor]:
    return [torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True) for a in arrays]


@torch.no_grad()
def predict(model: DCRNN, arr, t: np.ndarray, device: torch.device) -> np.ndarray:
    """(len(t), 12, N) forecasts in mph, clipped at 0."""
    model.eval()
    out = []
    for i in range(0, len(t), EVAL_BATCH):
        x, cov, _ = windows(arr, t[i : i + EVAL_BATCH])
        x, cov = to_device(x, cov, device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x, cov)
        out.append((z.float() * arr.std + arr.mean).clamp_min(0).cpu().numpy())
    return np.concatenate(out)


def masked_mae_np(pred: np.ndarray, target: np.ndarray) -> float:
    ok = ~np.isnan(target)
    return float(np.abs(pred[ok] - target[ok]).mean())


def train_epoch(model, opt, scaler, arr, t_train, rng, step, device, limit=None) -> tuple[float, int]:
    model.train()
    order = rng.permutation(t_train)
    n_batches = len(order) // BATCH if limit is None else limit
    total = 0.0
    for b in range(n_batches):
        x, cov, y = windows(arr, order[b * BATCH : (b + 1) * BATCH])
        x, cov, y = to_device(x, cov, y, device=device)
        teacher = (y - arr.mean) / arr.std
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x, cov, teacher, teacher_probability(step, CL_DECAY))
        loss = masked_mae(z.float() * arr.std + arr.mean, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        step += 1
    return total / n_batches, step


def save_state(state: dict) -> None:
    """Write to a temporary file, then rename: a kill mid-write never corrupts the last good state."""
    RESUME.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESUME.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, RESUME)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true", help="time 50 batches and one validation pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help=f"continue from {RESUME.name}")
    args = ap.parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available; DCRNN training needs it (see README)")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    speed = load_speed()
    mean, std = standardisation(load_speed("train"))
    arr = build_arrays(speed, mean, std)
    t_of = {s: row_times(speed.index, owned_period(s)) for s in ("train", "validation", "test")}
    # windows whose 12 targets are all missing carry no training signal
    has_target = ~np.isnan(arr.target).all(axis=1)
    t_train = t_of["train"][np.array([has_target[t + 1 : t + 1 + N_OUT].any() for t in t_of["train"]])]

    cfg = Config()
    model = DCRNN(random_walk_supports(load_adjacency()), cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, eps=ADAM_EPS)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, list(MILESTONES), GAMMA)
    scaler = torch.amp.GradScaler("cuda")
    val_target = arr.target[t_of["validation"][:, None] + np.arange(1, N_OUT + 1)]

    if args.timing:
        t0 = time.perf_counter()
        train_epoch(model, opt, scaler, arr, t_train, rng, 0, device, limit=50)
        torch.cuda.synchronize()
        per_batch = (time.perf_counter() - t0) / 50
        t0 = time.perf_counter()
        mae = masked_mae_np(predict(model, arr, t_of["validation"], device), val_target)
        val_s = time.perf_counter() - t0
        epoch = per_batch * (len(t_train) // BATCH) + val_s
        print(
            f"{n_params:,} parameters; {per_batch * 1000:.0f} ms/batch; validation pass {val_s:.0f} s; "
            f"~{epoch / 60:.1f} min/epoch; GPU peak {torch.cuda.max_memory_allocated() >> 20} MB; "
            f"val MAE after 50 batches {mae:.2f}"
        )
        return

    history, best, best_state, step, waited, start, prior_s, resumed = [], np.inf, None, 0, 0, 1, 0.0, []
    if args.resume:
        if not RESUME.exists():
            raise SystemExit(f"--resume: no saved state at {RESUME}")
        st = torch.load(RESUME, map_location=device, weights_only=False)
        if st["seed"] != args.seed or st["config"] != cfg.to_dict():
            raise SystemExit("--resume: saved state was made with a different seed or config")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        rng.bit_generator.state = st["numpy_rng"]
        torch.set_rng_state(st["torch_rng"].cpu())
        torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        history, best, best_state = st["history"], st["best"], st["best_state"]
        step, waited, prior_s, resumed = st["step"], st["waited"], st["elapsed_s"], st["resumed_at"]
        start = history[-1]["epoch"] + 1
        resumed = [*resumed, start]
        print(f"resuming at epoch {start}; best val so far {best:.3f}", flush=True)
        if waited >= PATIENCE or start > MAX_EPOCHS:
            print("saved run had already stopped; scoring it", flush=True)
            start = MAX_EPOCHS + 1
    for epoch in range(start, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        lr, tp = opt.param_groups[0]["lr"], teacher_probability(step, CL_DECAY)
        train_mae, step = train_epoch(model, opt, scaler, arr, t_train, rng, step, device)
        sched.step()
        val_mae = masked_mae_np(predict(model, arr, t_of["validation"], device), val_target)
        history.append(
            {
                "epoch": epoch,
                "train_mae": round(train_mae, 4),
                "val_mae": round(val_mae, 4),
                "lr": lr,
                "teacher_prob_start": round(tp, 4),
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(
            f"epoch {epoch}: train {train_mae:.3f}  val {val_mae:.3f}  ({history[-1]['seconds']} s)",
            flush=True,
        )
        if val_mae < best:
            best, waited = val_mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        save_state(
            {
                "seed": args.seed,
                "config": cfg.to_dict(),
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "history": history,
                "best": best,
                "best_state": best_state,
                "step": step,
                "waited": waited,
                "elapsed_s": prior_s + time.perf_counter() - started,
                "resumed_at": resumed,
            }
        )
        if waited >= PATIENCE:
            break
    model.load_state_dict(best_state)
    best_epoch = min(history, key=lambda r: r["val_mae"])["epoch"]
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg.to_dict(), "mean": mean, "std": std}, CHECKPOINT)

    # score validation and test (once) on exactly the rows every other model is scored on
    results, residuals = {}, {}
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        rows = load_eval_rows(split, speed)
        pred = predict(model, arr, t_of[split], device)
        grids = {}
        for h in HORIZONS:
            grid = np.full(speed.shape, np.nan, np.float32)
            grid[t_of[split]] = pred[:, h - 1]
            grids[h] = grid
        results[split] = {f"h{h}": rows.score(grids[h], h) for h in HORIZONS}
        save_predictions(rows, grids, split)
        if split == "validation":
            residuals = residual_quantiles(rows, grids)

    payload = {
        "model": "DCRNN (diffusion convolutional recurrent network), one model for all 12 steps",
        "git_commit": git_commit(),
        "seed": args.seed,
        "config": cfg.to_dict(),
        "training": {
            "batch": BATCH,
            "optimizer": f"Adam lr {LR} eps {ADAM_EPS}, x{GAMMA} at epochs {list(MILESTONES)}",
            "early_stopping": f"validation masked MAE over 12 steps, patience {PATIENCE}, "
            f"max {MAX_EPOCHS} epochs",
            "scheduled_sampling_decay_batches": CL_DECAY,
            "gradient_clip_norm": CLIP_NORM,
            "mixed_precision": "float16 autocast + GradScaler",
            "train_windows": int(len(t_train)),
            "resumed_at_epochs": resumed,
        },
        "normalisation": {"mean_mph": mean, "std_mph": std},
        "parameters": n_params,
        "data": {s: [str(v) for v in owned_period(s)] for s in ("train", "validation", "test")},
        "best_epoch": best_epoch,
        "history": history,
        "results": results,
        "residuals_validation": residuals,
        "device": torch.cuda.get_device_name(0),
        "gpu_peak_mb": torch.cuda.max_memory_allocated() >> 20,
        # after a resume, both peaks cover only the final session; runtime covers all sessions
        "peak_rss_mb": peak_memory_mb(),
        "runtime_s": round(prior_s + time.perf_counter() - started, 1),
    }
    METRICS.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    learning_curve(history, best_epoch)
    baselines = json.loads(BASELINES.read_text(encoding="utf-8"))["results"]
    lightgbm = json.loads(LIGHTGBM.read_text(encoding="utf-8"))["results"]
    CARD.write_text(card(payload, baselines, lightgbm), encoding="utf-8")
    print_report(CARD)


def save_predictions(rows: EvalRows, grids: dict[int, np.ndarray], split: str) -> None:
    frame = rows.frame[["timestamp", "node_id"]].reset_index(drop=True)
    for h, grid in grids.items():
        frame[f"pred_h{h}"] = grid[rows.t_idx, rows.node].astype(np.float32)
    frame.to_parquet(PREDICTIONS / f"dcrnn_{split}.parquet", index=False)


def residual_quantiles(rows: EvalRows, grids: dict[int, np.ndarray]) -> dict:
    out = {}
    for h, grid in grids.items():
        err = grid[rows.t_idx, rows.node] - rows.target(h)
        err = err[~np.isnan(err)]
        out[f"h{h}"] = {f"p{q}": round(float(np.percentile(err, q)), 3) for q in (1, 5, 25, 50, 75, 95, 99)}
        out[f"h{h}"]["mean"] = round(float(err.mean()), 4)
    return out


def learning_curve(history: list[dict], best_epoch: int) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(MODEL_FIGURE_STYLE)
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(epochs, [r["train_mae"] for r in history], color=COMPARISON, linewidth=2, label="training")
    ax.plot(epochs, [r["val_mae"] for r in history], color=FORECAST, linewidth=2, label="validation")
    ax.axvline(best_epoch, color="#8a8984", linewidth=1, linestyle="--")
    ax.annotate(" kept", (best_epoch, ax.get_ylim()[1]), va="top", color="#52514e", fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MAE over 12 steps (mph)")
    ax.set_title("DCRNN learning curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curve.png", dpi=150)
    plt.close(fig)


def card(p: dict, baselines: dict, lightgbm: dict) -> str:
    rows = []
    for split in ("validation", "test"):
        for h in HORIZONS:
            k = f"h{h}"
            ours, lgb, ar = (
                p["results"][split][k]["all"],
                lightgbm[split][k]["all"],
                baselines[split][FLOOR][k]["all"],
            )
            rows.append(
                f"| {split} | {h * 5} min | {ours['mae']:.2f} | {ours['rmse']:.2f} | {ours['mape']:.1f}% "
                f"| {lgb['mae']:.2f} / {lgb['rmse']:.2f} | {ar['mae']:.2f} / {ar['rmse']:.2f} |"
            )
    slices = []
    for h in HORIZONS:
        k = f"h{h}"
        cells = []
        for s in ("peak", "off_peak", "congested", "free_flowing"):
            ours = p["results"]["validation"][k][s]["mae"]
            cells.append(f"{ours:.2f} ({lightgbm['validation'][k][s]['mae']:.2f})")
        slices.append(f"| {h * 5} min | " + " | ".join(cells) + " |")
    d, t = p["data"], p["training"]
    return "\n".join(
        [
            "# Model card: DCRNN speed forecaster",
            "",
            "Generated by `scripts/train_dcrnn.py` from `models/metrics/dcrnn.json`. Do not edit by hand.",
            "",
            f"- **Model:** {p['model']}; {p['parameters']:,} parameters; 207 METR-LA sensors on the "
            "published directed road graph (random walk in both directions).",
            f"- **Code version:** git `{p['git_commit']}`; seed {p['seed']}.",
            f"- **Training data:** {d['train'][0]} → {d['train'][1]} ({t['train_windows']:,} windows). "
            f"Validation {d['validation'][0]} → {d['validation'][1]} (early stopping). "
            f"Test {d['test'][0]} → {d['test'][1]} (scored once).",
            "- **Inputs per sensor and step (last 12 steps):** standardised speed (missing → training mean), "
            "missing flag, time of day (sin, cos), weekend flag. The decoder also gets the time features "
            "of each target step.",
            f"- **Settings:** `{json.dumps(p['config'])}`; {t['optimizer']}; batch {t['batch']}; "
            f"{t['mixed_precision']}; early stopping on {t['early_stopping']}. "
            "Published DCRNN settings, not tuned.",
            f"- **Training:** best epoch {p['best_epoch']} of {len(p['history'])}; "
            f"{p['runtime_s'] / 60:.0f} min end to end on {p['device']}; GPU peak {p['gpu_peak_mb']:,} MB; "
            f"process peak memory {p['peak_rss_mb']:,} MB.",
            "",
            "## Accuracy (masked; MAPE excludes targets under 5 mph)",
            "",
            "| Split | Horizon | MAE | RMSE | MAPE | LightGBM MAE / RMSE | AR baseline MAE / RMSE |",
            "|---|---|---:|---:|---:|---|---|",
            *rows,
            "",
            "## Validation MAE by slice (LightGBM in brackets)",
            "",
            "| Horizon | peak | off-peak | congested | free-flowing |",
            "|---|---:|---:|---:|---:|",
            *slices,
            "",
            "![learning curve](figures/dcrnn/learning_curve.png)",
            "",
            "## Limits",
            "",
            "- Trained on 12 weeks of spring 2012 traffic; no holidays, events or weather inputs.",
            "- One seed; published hyperparameters, not tuned for this split.",
            "- Run-to-run variation is not measured.",
            "- Point forecasts only: no prediction intervals.",
            "- Needs a CUDA GPU to train; inference also runs on CPU.",
            "",
        ]
    )


if __name__ == "__main__":
    main()

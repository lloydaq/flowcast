"""Train the final LightGBM models and score validation and test once.

Usage:  python scripts/train_lightgbm.py   (after scripts/tune_lightgbm.py; ~20-30 min on CPU)

Settings come from models/metrics/lightgbm_search.json (chosen on validation only). The final
models use the full training period and a lower learning rate (0.05), early-stopped on validation.
The test period is scored here, once, with nothing left to change.

Writes:
  models/checkpoints/lightgbm/h{h}.txt         boosters (git-ignored; rebuild with this script)
  data/processed/predictions/lightgbm_{split}.parquet   per-row forecasts, for the API (git-ignored)
  models/metrics/lightgbm.json                 metrics, slices, importance, residual quantiles
  docs/model_card_lightgbm.md                  generated model card
  docs/figures/lightgbm/*.png
"""

from __future__ import annotations

import json
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

from flowcast import paths  # noqa: E402
from flowcast.data.metr_la import FREQ  # noqa: E402
from flowcast.data.processed import owned_period  # noqa: E402
from flowcast.evaluation.metrics import evaluate  # noqa: E402
from flowcast.features.build import HORIZONS  # noqa: E402
from flowcast.models.lgbm import BASE_PARAMS, group_permutation_importance, load_split, train  # noqa: E402

ROOT = paths.PROJECT_ROOT
SEARCH = ROOT / "models" / "metrics" / "lightgbm_search.json"
BASELINES = ROOT / "models" / "metrics" / "baselines.json"
METRICS = ROOT / "models" / "metrics" / "lightgbm.json"
CHECKPOINTS = ROOT / "models" / "checkpoints" / "lightgbm"
PREDICTIONS = paths.PROCESSED_DIR / "predictions"
CARD = ROOT / "docs" / "model_card_lightgbm.md"
FIGURES = ROOT / "docs" / "figures" / "lightgbm"
FINAL = {"learning_rate": 0.05}
MAX_ROUNDS, PATIENCE = 6000, 100
PERMUTATION_ROWS = 100_000
FLOOR = "ar_residual"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FORECAST, COMPARISON = "#2a78d6", "#a86f0c"  # architect.md §10 data hues


def main() -> None:
    started = time.perf_counter()
    search = json.loads(SEARCH.read_text(encoding="utf-8"))
    params = {**BASE_PARAMS, **search["chosen"], **FINAL}
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    results: dict = {"validation": {}, "test": {}}
    models: dict = {}
    importance: dict = {}
    residuals: dict = {}
    preds: dict = {"validation": {}, "test": {}}
    peak_rss = 0

    for h in HORIZONS:
        x, y, _ = load_split("train", h)
        xv, yv, mv = load_split("validation", h)
        t = time.perf_counter()
        booster = train(params, x, y, xv, yv, MAX_ROUNDS, PATIENCE)
        mem = psutil.Process().memory_info()  # peak_wset: lifetime peak on Windows; rss elsewhere
        peak_rss = max(peak_rss, getattr(mem, "peak_wset", mem.rss))
        fit_s = time.perf_counter() - t
        n_train = int(y.notna().sum())
        del x, y
        best = booster.best_iteration
        path = CHECKPOINTS / f"h{h}.txt"
        booster.save_model(path, num_iteration=best)
        models[f"h{h}"] = {
            "best_iteration": best,
            "fit_seconds": round(fit_s, 1),
            "train_rows": n_train,
            "file_mb": round(path.stat().st_size / 2**20, 1),
        }

        def predict(frame: pd.DataFrame, b=booster, n=best) -> np.ndarray:
            return np.clip(b.predict(frame, num_iteration=n), 0.0, None)

        xt, yt, mt = load_split("test", h)
        for split, (xs, ys, ms) in {"validation": (xv, yv, mv), "test": (xt, yt, mt)}.items():
            p = predict(xs)
            preds[split][f"h{h}"] = p
            preds[split].setdefault("meta", ms[["timestamp", "node_id"]])
            results[split][f"h{h}"] = evaluate(
                ys.to_numpy(np.float64),
                p,
                pd.DatetimeIndex(ms["timestamp"]) + h * FREQ,
                ms["node_id"].to_numpy(),
                ms["free_flow"].to_numpy(np.float64),
            )
        err = preds["validation"][f"h{h}"] - yv.to_numpy()
        err = err[~np.isnan(err)]
        residuals[f"h{h}"] = {
            f"p{q}": round(float(np.percentile(err, q)), 3) for q in (1, 5, 25, 50, 75, 95, 99)
        }
        residuals[f"h{h}"]["mean"] = round(float(err.mean()), 4)

        gain = booster.feature_importance("gain")
        gain_pct = 100 * gain / gain.sum()
        sample = np.random.default_rng(h).choice(len(xv), size=min(PERMUTATION_ROWS, len(xv)), replace=False)
        importance[f"h{h}"] = {
            "gain_pct": dict(
                sorted(zip(booster.feature_name(), map(float, gain_pct), strict=True), key=lambda kv: -kv[1])
            ),
            "group_permutation_mae_increase": group_permutation_importance(
                predict, xv.iloc[sample], yv.to_numpy(np.float64)[sample], seed=h
            ),
        }
        print(
            f"h{h}: {best} rounds, {fit_s:.0f} s, val MAE {results['validation'][f'h{h}']['all']['mae']:.3f}",
            flush=True,
        )
        del xv, yv, xt, yt

    for split, by_h in preds.items():
        frame = by_h.pop("meta").reset_index(drop=True)
        for k, v in by_h.items():
            frame[f"pred_{k}"] = v.astype(np.float32)
        frame.to_parquet(PREDICTIONS / f"lightgbm_{split}.parquet", index=False)

    baselines = json.loads(BASELINES.read_text(encoding="utf-8"))["results"]
    payload = {
        "model": "LightGBM, one direct global model per horizon",
        "git_commit": git_commit(),
        "params": params,
        "data": {s: [str(v) for v in owned_period(s)] for s in ("train", "validation", "test")},
        "early_stopping": f"validation MAE, patience {PATIENCE}, max {MAX_ROUNDS} rounds",
        "models": models,
        "results": results,
        "residuals_validation": residuals,
        "importance": importance,
        "peak_rss_mb": peak_rss // 2**20,
        "runtime_s": round(time.perf_counter() - started, 1),
    }
    METRICS.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    figures(results, baselines, preds)
    CARD.write_text(card(payload, baselines), encoding="utf-8")
    print(CARD.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:

        def git(*args: str) -> str:
            return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=True).stdout

        dirty = git("status", "--porcelain", "--untracked-files=no").strip()
        return git("rev-parse", "--short", "HEAD").strip() + (" + uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def figures(results: dict, baselines: dict, preds: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "grid.color": GRID,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        }
    )
    hours = np.arange(24)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, h in zip(axes, (3, 12), strict=True):
        ours = results["validation"][f"h{h}"]["mae_by_hour"]
        floor = baselines["validation"][FLOOR][f"h{h}"]["mae_by_hour"]
        ax.plot(
            hours,
            [ours[str(k)] if str(k) in ours else ours.get(k) for k in hours],
            color=FORECAST,
            linewidth=2,
            label="LightGBM",
        )
        ax.plot(hours, [floor[str(k)] for k in hours], color=COMPARISON, linewidth=2, label="AR baseline")
        ax.set_title(f"{h * 5}-minute forecast")
        ax.set_xticks(range(0, 24, 3), [f"{k:02d}:00" for k in range(0, 24, 3)])
        ax.set_xlabel("hour of the target time")
    axes[0].set_ylabel("validation MAE (mph)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Error by hour of day: both models struggle in the rush hours; LightGBM cuts the peaks",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "mae_by_hour.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = load_split("validation", 12)[1].to_numpy()
    err = preds["validation"]["h12"] - y
    err = err[~np.isnan(err)]
    ax.hist(
        np.clip(err, -30, 30), bins=np.arange(-30, 30.5, 1), color=FORECAST, edgecolor=SURFACE, linewidth=0.5
    )
    ax.set_yscale("log")
    ax.set_xlabel("forecast − actual (mph), clipped to ±30")
    ax.set_ylabel("rows (log scale)")
    ax.set_title(
        "60-minute residuals on validation\nCentred on zero; the right tail is jams the model missed"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "residuals_h12.png", dpi=150)
    plt.close(fig)


def card(p: dict, baselines: dict) -> str:
    rows = []
    for split in ("validation", "test"):
        for h in HORIZONS:
            ours = p["results"][split][f"h{h}"]["all"]
            floor = baselines[split][FLOOR][f"h{h}"]["all"]
            gain = 100 * (floor["mae"] - ours["mae"]) / floor["mae"]
            rows.append(
                f"| {split} | {h * 5} min | {ours['mae']:.2f} | {ours['rmse']:.2f} | {ours['mape']:.1f}% "
                f"| {floor['mae']:.2f} | {floor['rmse']:.2f} | {gain:+.1f}% |"
            )
    slices = []
    for h in HORIZONS:
        r = p["results"]["validation"][f"h{h}"]
        f = baselines["validation"][FLOOR][f"h{h}"]
        slices.append(
            f"| {h * 5} min | "
            + " | ".join(
                f"{r[s]['mae']:.2f} ({f[s]['mae']:.2f})"
                for s in ("peak", "off_peak", "congested", "free_flowing")
            )
            + " |"
        )
    perm = p["importance"]["h12"]["group_permutation_mae_increase"]
    perm_rows = [
        f"| {g} | {v:+.3f} | {p['importance']['h3']['group_permutation_mae_increase'][g]:+.3f} |"
        for g, v in sorted(perm.items(), key=lambda kv: -kv[1])
    ]
    top = list(p["importance"]["h12"]["gain_pct"].items())[:8]
    params = {k: v for k, v in p["params"].items() if k not in ("verbose", "metric")}
    d = p["data"]
    return "\n".join(
        [
            "# Model card: LightGBM speed forecaster",
            "",
            "Generated by `scripts/train_lightgbm.py` from `models/metrics/lightgbm.json`. "
            "Do not edit by hand.",
            "",
            f"- **Model:** {p['model']} (horizons 15 / 30 / 45 / 60 min), 207 METR-LA sensors.",
            f"- **Code version:** git `{p['git_commit']}` (the commit the script ran on).",
            f"- **Training data:** {d['train'][0]} → {d['train'][1]}. "
            f"Validation {d['validation'][0]} → {d['validation'][1]} (early stopping and model selection). "
            f"Test {d['test'][0]} → {d['test'][1]} (scored once).",
            "- **Inputs:** the 39 point-in-time features for the horizon ([features.md](features.md)); "
            "missing values stay NaN.",
            f"- **Settings:** `{json.dumps(params)}`; early stopping on {p['early_stopping']}.",
            "- **Rounds:** " + ", ".join(f"{k} {v['best_iteration']}" for k, v in p["models"].items()),
            f"- **Cost:** {p['runtime_s'] / 60:.0f} min end to end on 8 CPU threads; "
            + (
                f"peak memory {p['peak_rss_mb']:,} MB."
                if p["peak_rss_mb"]
                else f"peak memory not recorded ({p['memory_note']})."
            ),
            "",
            "## Accuracy (masked; MAPE excludes targets under 5 mph)",
            "",
            "| Split | Horizon | MAE | RMSE | MAPE | AR baseline MAE | AR baseline RMSE "
            "| MAE reduction vs AR |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Validation MAE by slice (AR baseline in brackets)",
            "",
            "| Horizon | peak | off-peak | congested | free-flowing |",
            "|---|---:|---:|---:|---:|",
            *slices,
            "",
            "## What the model relies on",
            "",
            f"Validation MAE increase when a feature group is shuffled ({PERMUTATION_ROWS:,} random rows):",
            "",
            "| Group | 60 min | 15 min |",
            "|---|---:|---:|",
            *perm_rows,
            "",
            "Top features by split gain at 60 min: " + ", ".join(f"`{k}` {v:.1f}%" for k, v in top) + ".",
            "",
            "## Limits",
            "",
            "- Trained on 12 weeks of spring 2012 traffic; no holidays, events or weather inputs.",
            "- Point forecasts only: no prediction intervals yet.",
            "- Accuracy drops sharply on congested targets; see the slice table.",
            "- Not evaluated beyond the METR-LA network or after June 2012.",
            "",
        ]
    )


if __name__ == "__main__":
    main()

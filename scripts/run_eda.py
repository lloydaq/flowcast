"""Phase 2 exploratory data analysis.

Usage:  python scripts/run_eda.py   (after scripts/prepare_metr_la.py)

All modelling-relevant analysis uses the TRAINING period only. The validation
period is used once, for a drift check. The test period is not touched.

Writes figures to docs/figures/eda/ and numbers to docs/eda_summary.json.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from flowcast import paths  # noqa: E402
from flowcast.analysis import eda  # noqa: E402
from flowcast.data.metr_la import load_sensor_graph  # noqa: E402
from flowcast.data.processed import load_sensors, load_speed  # noqa: E402

FIG_DIR = paths.PROJECT_ROOT / "docs" / "figures" / "eda"
SUMMARY_PATH = paths.PROJECT_ROOT / "docs" / "eda_summary.json"
HORIZONS = {"15 min": 3, "30 min": 6, "45 min": 9, "60 min": 12}

# Reference palette (dataviz skill, light mode)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list(
    "flowcast_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "lines.linewidth": 2,
            "font.size": 10,
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.close(fig)


def hour_ticks(ax: plt.Axes) -> None:
    ax.set_xticks(range(0, 289, 36), [f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_xlim(0, 287)


def main() -> None:
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train = load_speed("train")
    sensors = load_sensors()
    graph = load_sensor_graph()
    s: dict = {"period": {"split": "train", "start": str(train.index[0]), "end": str(train.index[-1])}}

    # 1. Speed distribution -------------------------------------------------
    observed = train.to_numpy()[~np.isnan(train.to_numpy())]
    s["distribution"] = {
        "percentiles_mph": {
            str(p): round(float(np.percentile(observed, p)), 2) for p in (1, 5, 10, 25, 50, 75, 90, 99)
        },
        "share_below_mph": {str(t): round(float((observed < t).mean() * 100), 2) for t in (20, 30, 40, 50)},
    }
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(observed, bins=np.arange(0, 71, 1), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set(xlabel="Speed (mph)", ylabel="Readings", title="Observed speed, training period")
    ax.text(2, ax.get_ylim()[1] * 0.85, f"median {np.median(observed):.1f} mph", color=INK_2)
    save(fig, "01_speed_distribution")

    # 2. Congestion definition ----------------------------------------------
    ff = eda.free_flow_speed(train)
    cong = eda.congestion_mask(train, ff)
    cong_rate = cong.mean()  # per sensor
    s["congestion"] = {
        "definition": f"speed < {eda.CONGESTION_RATIO} x per-sensor p{eda.FREE_FLOW_PERCENTILE} speed",
        "free_flow_mph": ff.describe().round(2).to_dict(),
        "overall_rate_pct": round(float(np.nanmean(cong.to_numpy()) * 100), 2),
        "sensor_rate_pct": (cong_rate * 100).describe().round(2).to_dict(),
        "sensors_never_congested": int((cong_rate == 0).sum()),
    }

    # 3. Time-of-day profile --------------------------------------------------
    profile = eda.time_of_day_profile(train)
    wd = profile["weekday"]
    am, pm = wd.loc[: 12 * 12 - 1], wd.loc[12 * 12 :]

    def clock(slot: int) -> str:
        return f"{slot // 12:02d}:{slot % 12 * 5:02d}"

    s["daily_profile"] = {
        "weekday_am_trough": {"time": clock(am.idxmin()), "mph": round(am.min(), 2)},
        "weekday_pm_trough": {"time": clock(pm.idxmin()), "mph": round(pm.min(), 2)},
        "weekday_max_mph": round(wd.max(), 2),
        "weekend_min_mph": round(profile["weekend"].min(), 2),
        "weekend_min_time": clock(profile["weekend"].idxmin()),
    }
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for col, color in (("weekday", BLUE), ("weekend", ORANGE)):
        ax.plot(profile.index, profile[col], color=color, label=col.capitalize())
        ax.annotate(
            col.capitalize(),
            (287, profile[col].iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=INK_2,
            va="center",
        )
    hour_ticks(ax)
    ax.set(ylabel="Network mean speed (mph)", title="Average speed by time of day")
    ax.legend(loc="lower left")
    save(fig, "02_time_of_day_profile")

    # 4. Weekly congestion heatmap -------------------------------------------
    grid = eda.weekly_grid(cong.mean(axis=1) * 100)
    s["weekly_congestion_peak"] = {
        "day": str(grid.stack().idxmax()[0]),
        "hour": int(grid.stack().idxmax()[1]),
        "rate_pct": round(float(grid.to_numpy().max()), 2),
    }
    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=SEQ, vmin=0)
    ax.set_yticks(range(7), grid.index)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.grid(False)
    ax.set(xlabel="Hour of day", title="Share of sensors congested, by weekday and hour (%)")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.outline.set_visible(False)
    save(fig, "03_weekly_congestion_heatmap")

    # 5. Daily network speed over time ---------------------------------------
    full_days = train.loc[: train.index[-1].normalize() - pd.Timedelta(minutes=5)]  # drop partial last day
    daily = full_days.mean(axis=1).resample("D").mean()
    daily_missing = full_days.isna().mean(axis=1).resample("D").mean() * 100
    s["daily_network_speed"] = {
        "min_day": str(daily.idxmin().date()),
        "min_mph": round(daily.min(), 2),
        "max_day": str(daily.idxmax().date()),
        "max_mph": round(daily.max(), 2),
        "weekday_mean": round(daily[daily.index.day_of_week < 5].mean(), 2),
        "weekend_mean": round(daily[daily.index.day_of_week >= 5].mean(), 2),
    }
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(daily.index, daily, color=BLUE, linewidth=1.5)
    wkend = daily[daily.index.day_of_week >= 5]
    ax.scatter(
        wkend.index,
        wkend,
        color=ORANGE,
        s=46,
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1.5,
        label="Weekend day",
    )
    ax.legend(loc="lower left")
    ax.set(ylabel="Mean speed (mph)", title="Daily network mean speed")
    save(fig, "04_daily_network_speed")

    # 6. Missingness timeline ------------------------------------------------
    miss = train.isna().resample("6h").mean().T * 100
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(miss.to_numpy(), aspect="auto", cmap=SEQ, vmin=0, vmax=100, interpolation="nearest")
    ticks = np.arange(0, miss.shape[1], 28)
    ax.set_xticks(ticks, [miss.columns[i].strftime("%b %d") for i in ticks])
    ax.set(ylabel="Sensor (node_id)", title="Missing readings per sensor, 6-hour bins (%)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.outline.set_visible(False)
    save(fig, "05_missingness_timeline")
    s["missing_by_hour_pct"] = (
        (train.isna().mean(axis=1).groupby(train.index.hour).mean() * 100).round(2).to_dict()
    )
    s["worst_missing_days"] = {str(k.date()): round(v, 1) for k, v in daily_missing.nlargest(5).items()}

    # 7. Sensor map ------------------------------------------------------------
    rate = (cong_rate * 100).reindex(sensors["sensor_id"]).to_numpy()
    top = sensors.assign(rate=rate).nlargest(5, "rate")
    s["most_congested_sensors"] = top[["node_id", "sensor_id", "rate"]].round(2).to_dict("records")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sc = ax.scatter(
        sensors["longitude"],
        sensors["latitude"],
        c=rate,
        cmap=SEQ,
        s=34,
        edgecolor=SURFACE,
        linewidth=1.2,
        vmin=0,
    )
    ax.scatter(
        top["longitude"],
        top["latitude"],
        s=150,
        facecolor="none",
        edgecolor=INK,
        linewidth=1.2,
        label="5 most congested (listed in docs/eda.md)",
    )
    ax.legend(loc="upper right")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Congestion rate per sensor (% of readings)")
    ax.set_aspect(1 / np.cos(np.deg2rad(34.1)))
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.75)
    cbar.outline.set_visible(False)
    save(fig, "06_sensor_congestion_map")

    # 8. Autocorrelation -------------------------------------------------------
    network = train.mean(axis=1)
    lags = list(range(0, 7 * 288 + 1, 12))
    acf = eda.autocorrelation(network, lags)
    s["network_autocorrelation"] = {
        lbl: round(float(network.autocorr(lag)), 3)
        for lbl, lag in {"5 min": 1, "1 h": 12, "6 h": 72, "12 h": 144, "1 day": 288, "1 week": 2016}.items()
    }
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(np.array(lags) / 288, acf.to_numpy(), color=BLUE)
    for d, lbl in ((1, "1 day"), (7, "1 week")):
        ax.axvline(d, color=GRID, linewidth=1, zorder=0)
        ax.annotate(lbl, (d, 0.95), xytext=(-4, 0), textcoords="offset points", ha="right", color=INK_2)
    ax.set(
        xlabel="Lag (days)",
        ylabel="Autocorrelation",
        title="Autocorrelation of network mean speed",
        xlim=(0, 7.05),
    )
    save(fig, "07_autocorrelation")

    # 9. Per-sensor predictability across horizons ---------------------------
    hac = eda.horizon_autocorrelation(train, list(HORIZONS.values()))
    hac.columns = list(HORIZONS)
    s["horizon_autocorrelation"] = {h: hac[h].describe().round(3).to_dict() for h in HORIZONS}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.boxplot(
        [hac[h].dropna() for h in HORIZONS],
        tick_labels=list(HORIZONS),
        widths=0.45,
        patch_artist=True,
        boxprops={"facecolor": "#b7d3f6", "edgecolor": BLUE},
        medianprops={"color": INK, "linewidth": 2},
        whiskerprops={"color": BLUE},
        capprops={"color": BLUE},
        flierprops={"markeredgecolor": BLUE, "markersize": 4},
    )
    ax.set(
        ylabel="corr(speed now, speed later)",
        xlabel="Forecast horizon",
        title="Per-sensor correlation with the future value",
    )
    save(fig, "08_horizon_autocorrelation")

    # 10. Spatial correlation vs road distance -------------------------------
    bins = [0, 1, 2, 3, 5, 8, 12, 20, 40, 80]
    by_dist = eda.correlation_by_distance(train, graph.distances, sensors["sensor_id"].tolist(), bins)
    s["correlation_by_road_distance_km"] = {
        str(k): {"mean_corr": round(float(v["mean"]), 3), "pairs": int(v["count"])}
        for k, v in by_dist.iterrows()
    }
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [f"{iv.left:g}-{iv.right:g}" for iv in by_dist.index]
    ax.bar(labels, by_dist["mean"], color=BLUE, width=0.7)
    ax.set(
        xlabel="Directed road distance between sensors (km)",
        ylabel="Mean speed correlation",
        title="Speed correlation falls with road distance",
    )
    save(fig, "09_correlation_vs_distance")

    # 11. Lead-lag along graph edges -----------------------------------------
    lagc = eda.edge_lag_correlation(train, graph.adjacency, max_lag=6)
    best = lagc.idxmax(axis=1)
    s["edge_lead_lag"] = {
        "edges": len(lagc),
        "best_lag_share_pct": (best.value_counts(normalize=True).sort_index() * 100).round(2).to_dict(),
        "mean_corr_of_differences_by_lag": lagc.mean().round(4).to_dict(),
    }
    fig, ax = plt.subplots(figsize=(7, 3.4))
    share = best.value_counts(normalize=True).reindex(range(7), fill_value=0) * 100
    ax.bar([f"{k * 5}" for k in share.index], share, color=BLUE, width=0.7)
    ax.set(
        xlabel="Lag giving the highest correlation of speed changes (minutes)",
        ylabel="Share of edges (%)",
        title="How quickly speed changes propagate along graph edges",
    )
    save(fig, "10_edge_lead_lag")

    # 12. Congestion episodes -------------------------------------------------
    episodes = eda.congestion_episodes(cong)
    minutes = episodes["steps"] * 5
    long = episodes[episodes["steps"] >= 3]
    s["congestion_episodes"] = {
        "count": len(episodes),
        "duration_min": minutes.describe().round(1).to_dict(),
        "share_single_step_pct": round(float((episodes["steps"] == 1).mean() * 100), 2),
        "onset_hour_share_pct_ge_15min": (
            long["start"].dt.hour.value_counts(normalize=True).sort_index() * 100
        )
        .round(2)
        .to_dict(),
    }
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(minutes, bins=np.arange(0, 6 * 60 + 5, 10), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set_yscale("log")
    ax.set(
        xlabel="Episode duration (minutes, truncated at 6 h)",
        ylabel="Episodes (log scale)",
        title="Congestion episode durations",
    )
    save(fig, "11_congestion_episode_durations")

    # 13. Data-quality: low readings and DST ---------------------------------
    s["low_speed_readings"] = eda.isolated_low_readings(train, low=5, neighbour_floor=30)
    pre, post = train.loc[:"2012-03-10"], train.loc["2012-03-12":"2012-04-10"]
    s["dst_check"] = {
        "weekday_am_trough_before_2012_03_11": eda.morning_trough_slot(pre) * 5,
        "weekday_am_trough_after_2012_03_11": eda.morning_trough_slot(post) * 5,
        "unit": "minutes after midnight",
    }

    # 14. Drift check: train vs validation (validation used only here) -------
    val = load_speed("validation")
    val_cong = eda.congestion_mask(val, ff)  # train free-flow, as a model would see it
    vp = eda.time_of_day_profile(val)
    s["train_vs_validation"] = {
        "mean_speed": [
            round(float(np.nanmean(train.to_numpy())), 2),
            round(float(np.nanmean(val.to_numpy())), 2),
        ],
        "congestion_rate_pct": [
            s["congestion"]["overall_rate_pct"],
            round(float(np.nanmean(val_cong.to_numpy()) * 100), 2),
        ],
        "missing_pct": [
            round(float(train.isna().mean().mean() * 100), 2),
            round(float(val.isna().mean().mean() * 100), 2),
        ],
        "weekday_profile_corr": round(float(profile["weekday"].corr(vp["weekday"])), 3),
        "note": "validation contains Memorial Day (2012-05-28, a Monday holiday)",
    }

    SUMMARY_PATH.write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()

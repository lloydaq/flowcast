// Models: "Which model is most accurate, at which horizon, and where does it fail?"
import { useMemo, useState } from "react";
import { Link } from "react-router";
import { Block, Ph } from "../components/Block";
import { ChartTable, LineChart } from "../components/charts/LineChart";
import { Segmented } from "../components/Segmented";
import type { Gap, HorizonScores, Metrics, ModelScores } from "../lib/api";
import { dec1, dec2, int, signed } from "../lib/format";
import { useMetrics, useSensorErrors } from "../lib/queries";
import { useReplay } from "../lib/replay";

const MINUTES = [15, 30, 45, 60] as const;
type Minutes = (typeof MINUTES)[number];
type Split = "test" | "validation";
type Metric = "mae" | "rmse" | "mape";
type Slice = "all" | "peak" | "off_peak" | "congested" | "free_flowing";

const SPLITS = [
  { value: "test", label: "Test" },
  { value: "validation", label: "Validation" },
] as const;
const METRICS = [
  { value: "mae", label: "MAE" },
  { value: "rmse", label: "RMSE" },
  { value: "mape", label: "MAPE" },
] as const;
const SLICES = [
  { value: "all", label: "All" },
  { value: "peak", label: "Rush hours" },
  { value: "off_peak", label: "Off-peak" },
  { value: "congested", label: "Congested" },
] as const;
const HORIZON_OPTS = MINUTES.map((m) => ({ value: m, label: `${m} min` }));
const LEARNED = ["DCRNN", "LightGBM"] as const;

const key = (m: Minutes) => String(m) as "15";

/** One model's scores at one split and horizon (the API always returns all four horizons). */
function at(x: ModelScores, split: Split, h: Minutes): HorizonScores {
  const v = x[split][key(h)];
  if (!v) throw new Error(`${x.model}: no ${split} scores at ${h} min`);
  return v;
}

export function ModelsScreen() {
  const metrics = useMetrics();
  const [split, setSplit] = useState<Split>("test");
  const [metric, setMetric] = useState<Metric>("mae");
  const [slice, setSlice] = useState<Slice>("all");

  return (
    <>
      <div className="title-row">
        <div>
          <h1>Models</h1>
          <Block query={metrics} what="the model scores" placeholder={<p className="finding"><Ph w={460} h={16} /></p>}>
            {(m) => <ModelsFinding m={m} />}
          </Block>
        </div>
      </div>

      <section className="card" aria-labelledby="table-title">
        <div className="card-head">
          <h2 id="table-title">Error by horizon</h2>
          <div className="controls">
            <Segmented<Split> label="Period" options={SPLITS} value={split} onChange={setSplit} />
            <Segmented<Metric> label="Metric" options={METRICS} value={metric} onChange={setMetric} />
            <Segmented<Slice> label="Traffic" options={SLICES} value={slice} onChange={setSlice} />
          </div>
        </div>
        <Block query={metrics} what="the model scores" placeholder={<TablePlaceholder />}>
          {(m) => <ScoreTable m={m} split={split} metric={metric} slice={slice} />}
        </Block>
      </section>

      <div className="grid-1-1" style={{ alignItems: "stretch" }}>
        <WorstSensors split={split} />
        <section className="card" aria-labelledby="hour-title">
          <div className="card-head">
            <h2 id="hour-title">Error by hour of day, 60 min ahead</h2>
            <div className="chart-legend" aria-hidden="true">
              <span>
                <i style={{ color: "var(--forecast)" }} /> DCRNN
              </span>
              <span>
                <i style={{ color: "var(--comparison)" }} /> LightGBM
              </span>
            </div>
          </div>
          <div className="card-body">
            <Block query={metrics} what="error by hour" placeholder={<Ph w="100%" h={300} />}>
              {(m) => <HourChart m={m} split={split} />}
            </Block>
          </div>
        </section>
      </div>

      <Block query={metrics} what="the significance check" placeholder={<TablePlaceholder rows={4} />}>
        {(m) => <GapTable gaps={m.dcrnn_minus_lightgbm[split]} split={split} />}
      </Block>
    </>
  );
}

function ModelsFinding({ m }: { m: Metrics }) {
  const gaps = m.dcrnn_minus_lightgbm.test ?? {};
  const better = MINUTES.filter((h) => verdict(gaps[key(h)]) === "DCRNN better");
  const tied = MINUTES.filter((h) => verdict(gaps[key(h)]) === "No clear difference");
  const dcrnn = m.models.find((x) => x.model === "DCRNN");
  const ar = m.models.find((x) => x.model === "AR on residuals");
  const cut =
    dcrnn && ar ? Math.round((1 - at(dcrnn, "test", 15).all.mae / at(ar, "test", 15).all.mae) * 100) : null;
  return (
    <p className="finding">
      {better.length > 0 && (
        <>
          <strong>DCRNN is the most accurate up to {better.at(-1)} min</strong> on the test period
          {tied.length > 0 ? `; at ${tied.join(" and ")} min it ties LightGBM` : ""}.
        </>
      )}{" "}
      {cut != null && <>At 15 min its typical error is {cut}% below the best simple baseline.</>}
    </p>
  );
}

function ScoreTable({ m, split, metric, slice }: { m: Metrics; split: Split; metric: Metric; slice: Slice }) {
  const value = (model: ModelScores, h: Minutes) => at(model, split, h)[slice][metric];
  const best = MINUTES.map((h) => Math.min(...m.models.map((x) => value(x, h))));
  const unit = metric === "mape" ? "%" : " mph";
  return (
    <div className="table-scroll" style={{ padding: "12px 0 8px" }}>
      <table className="data">
        <caption className="visually-hidden">
          {metric.toUpperCase()} by model and horizon, {split} period, {slice.replace("_", "-")} traffic. Lowest per column in bold.
        </caption>
        <thead>
          <tr>
            <th scope="col">Model</th>
            {MINUTES.map((h) => (
              <th key={h} scope="col" className="num">
                {h} min
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {m.models.map((x) => (
            <tr key={x.model}>
              <th scope="row">{sentence(x.model)}</th>
              {MINUTES.map((h, i) => {
                const v = value(x, h);
                return (
                  <td key={h} className={`num${v === best[i] ? " best" : ""}`}>
                    {metric === "mape" ? dec1(v) : dec2(v)}
                    <span className="unit">{unit.trim()}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ padding: "8px 20px 0", fontSize: "var(--fs-small)" }}>
        Lowest per column in bold. Every model is scored on the same rows with masked metrics; MAPE leaves out speeds under 5 mph.
      </p>
    </div>
  );
}

/** "persistence" → "Persistence"; names that start with a capital (DCRNN, AR) are left alone. */
const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function verdict(g: Gap | undefined): string {
  if (!g) return "–";
  const [lo = 0, hi = 0] = g.ci95;
  if (hi < 0) return "DCRNN better";
  if (lo > 0) return "LightGBM better";
  return "No clear difference";
}

function GapTable({ gaps, split }: { gaps: Record<string, Gap> | undefined; split: Split }) {
  return (
    <section className="card" aria-labelledby="gap-title">
      <div className="card-head">
        <h2 id="gap-title">Is DCRNN really better than LightGBM?</h2>
        <span className="sub">MAE gap with a 95% interval from resampling whole days (2,000 times)</span>
      </div>
      <div className="table-scroll" style={{ padding: "12px 0 8px" }}>
        <table className="data">
          <thead>
            <tr>
              <th scope="col">Horizon</th>
              <th scope="col" className="num">
                Gap (mph)
              </th>
              <th scope="col" className="num">
                95% interval
              </th>
              <th scope="col" className="num">
                Days DCRNN better
              </th>
              <th scope="col">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {MINUTES.map((h) => {
              const g = gaps?.[key(h)];
              const v = verdict(g);
              return (
                <tr key={h}>
                  <th scope="row">{h} min</th>
                  <td className="num">{g ? signed(g.mae_gap) : "–"}</td>
                  <td className="num">{g ? `${signed(g.ci95[0] ?? 0)} to ${signed(g.ci95[1] ?? 0)}` : "–"}</td>
                  <td className="num">{g ? `${int(g.days_dcrnn_better)} of ${int(g.days)}` : "–"}</td>
                  <td>
                    <span className={`pill ${v === "DCRNN better" ? "pill-positive" : "pill-neutral"}`}>{v}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="muted" style={{ padding: "8px 20px 0", fontSize: "var(--fs-small)" }}>
          Negative means DCRNN’s error is lower. A difference counts only when the interval excludes 0. It covers day-to-day
          variation in the {split} period, not training randomness: each model was trained once.
        </p>
      </div>
    </section>
  );
}

function WorstSensors({ split }: { split: Split }) {
  const r = useReplay();
  const [model, setModel] = useState<(typeof LEARNED)[number]>("DCRNN");
  const [h, setH] = useState<Minutes>(60);
  const q = useSensorErrors(model, h, split);
  return (
    <section className="card" aria-labelledby="worst-title">
      <div className="card-head">
        <h2 id="worst-title">Where it fails most</h2>
        <div className="controls">
          <Segmented label="Model" options={LEARNED.map((x) => ({ value: x, label: x }))} value={model} onChange={setModel} />
          <Segmented<Minutes> label="Horizon" options={HORIZON_OPTS} value={h} onChange={setH} />
        </div>
      </div>
      <Block query={q} what="the per-sensor errors" placeholder={<BarsPlaceholder />}>
        {(rows) => {
          const top = rows.slice(0, 6);
          const max = top[0]?.mae ?? 1;
          return (
            <ol className="bar-list" aria-label={`Sensors with the largest ${model} error at ${h} min`}>
              {top.map((s) => (
                <li key={s.sensor_id}>
                  <Link to={r.href(`/sensor/${s.sensor_id}`, { tab: "accuracy" })}>
                    <span className="bar-label">
                      <span className="mono">{s.sensor_id}</span>
                    </span>
                    <span className="bar-track">
                      <span className="bar" style={{ width: `${(s.mae / max) * 82}%` }} />
                      <span className="num">
                        {dec2(s.mae)}
                        <span className="unit">mph</span>
                      </span>
                    </span>
                  </Link>
                </li>
              ))}
            </ol>
          );
        }}
      </Block>
    </section>
  );
}

function HourChart({ m, split }: { m: Metrics; split: Split }) {
  const series = useMemo(
    () =>
      LEARNED.map((name) => {
        const found = m.models.find((x) => x.model === name);
        const by = found ? at(found, split, 60).mae_by_hour : {};
        return {
          key: name,
          label: name,
          color: name === "DCRNN" ? "var(--forecast)" : "var(--comparison)",
          points: Array.from({ length: 24 }, (_, hr) => [hr, by[String(hr) as "0"] ?? null] as const),
        };
      }),
    [m, split],
  );
  const worst = series[0]?.points.reduce((a, b) => ((b[1] ?? 0) > (a[1] ?? 0) ? b : a));
  return (
    <>
      <LineChart
        series={series}
        height={300}
        xDomain={[0, 23]}
        xTicks={[0, 4, 8, 12, 16, 20]}
        formatX={(x) => `${String(x).padStart(2, "0")}:00`}
        formatY={(y) => (Number.isInteger(y) ? String(y) : y.toFixed(1))}
        summary={`Mean absolute error 60 minutes ahead by hour of day, ${split} period. DCRNN is worst at ${String(worst?.[0] ?? 0).padStart(2, "0")}:00 with ${dec2(worst?.[1] ?? 0)} mph.`}
      />
      <ChartTable
        caption="MAE by hour of day, 60 minutes ahead"
        columns={["Hour", "DCRNN (mph)", "LightGBM (mph)"]}
        rows={Array.from({ length: 24 }, (_, hr) => [
          `${String(hr).padStart(2, "0")}:00`,
          dec2(series[0]?.points[hr]?.[1] ?? 0),
          dec2(series[1]?.points[hr]?.[1] ?? 0),
        ])}
      />
    </>
  );
}

function TablePlaceholder({ rows = 6 }: { rows?: number }) {
  return (
    <div style={{ padding: "12px 20px" }}>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} style={{ height: 44, display: "flex", alignItems: "center", gap: 24 }}>
          <Ph w={140} />
          <Ph w="60%" />
        </div>
      ))}
    </div>
  );
}

function BarsPlaceholder() {
  return (
    <div style={{ padding: "8px 20px" }}>
      {Array.from({ length: 6 }, (_, i) => (
        <div key={i} style={{ padding: "8px 0" }}>
          <Ph w={64} />
          <Ph w={`${80 - i * 6}%`} h={10} style={{ marginTop: 8 }} />
        </div>
      ))}
    </div>
  );
}

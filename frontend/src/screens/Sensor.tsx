// Sensor: "What will speed be here over the next hour, and how right were we?"
import { useQueries } from "@tanstack/react-query";
import { useMemo } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { Block, Ph } from "../components/Block";
import { ChartTable, LineChart, type Series } from "../components/charts/LineChart";
import { SensorMap } from "../components/map/SensorMap";
import { getJSON, type SensorDetail, type SensorError, type SensorForecast } from "../lib/api";
import { dec2, int, pct1, speed, speedChange } from "../lib/format";
import { useSensor, useSensorForecast, useSensors } from "../lib/queries";
import { HORIZONS, useReplay, type Horizon } from "../lib/replay";
import { band } from "../lib/severity";
import { formatClock, formatDay, toMs } from "../lib/time";

const TABS = [
  { key: "forecast", label: "Forecast" },
  { key: "history", label: "Past 24 hours" },
  { key: "accuracy", label: "Accuracy" },
] as const;
type Tab = (typeof TABS)[number]["key"];

export function SensorScreen() {
  const { id = "" } = useParams();
  const r = useReplay();
  const [params] = useSearchParams();
  const tabParam = params.get("tab");
  const tab: Tab = TABS.some((t) => t.key === tabParam) ? (tabParam as Tab) : "forecast";
  const detail = useSensor(id);
  const fc = useSensorForecast(id, r.t, r.model, 3);

  return (
    <>
      <div className="title-row">
        <div>
          <nav className="breadcrumb" aria-label="Breadcrumb">
            <Link to={r.href("/")}>Network</Link>
            <span aria-hidden="true">/</span>
            <h1 className="mono" style={{ fontSize: "var(--fs-title)", color: "var(--text)" }}>
              {id}
            </h1>
          </nav>
          <SensorFinding fc={fc.data} detail={detail.data} failed={detail.isError || fc.isError} error={detail.error?.message} />
        </div>
      </div>

      <nav className="tabs" aria-label="Sensor views">
        {TABS.map((t) => (
          <Link key={t.key} to={r.href(`/sensor/${id}`, { tab: t.key })} aria-current={t.key === tab ? "page" : undefined} replace>
            {t.label}
          </Link>
        ))}
      </nav>

      {tab === "forecast" && (
        <>
          <Block query={fc} what="this sensor’s forecast" placeholder={<KpiPlaceholder />}>
            {(data) => <HorizonStrip fc={data} />}
          </Block>
          <div className="grid-2-1" style={{ alignItems: "stretch" }}>
            <section className="card" aria-labelledby="fc-title">
              <div className="card-head">
                <h2 id="fc-title">Speed, the next hour and what happened</h2>
                <ForecastLegend />
              </div>
              <div className="card-body">
                <Block query={fc} what="this sensor’s forecast" placeholder={<Ph w="100%" h={260} />}>
                  {(data) => <ForecastChart fc={data} freeFlow={detail.data?.free_flow ?? null} />}
                </Block>
              </div>
            </section>
            <Locator id={id} detail={detail.data} />
          </div>
        </>
      )}
      {tab === "history" && <HistoryTab id={id} detail={detail.data} />}
      {tab === "accuracy" && <AccuracyTab id={id} />}
    </>
  );
}

function SensorFinding({
  fc,
  detail,
  failed,
  error,
}: {
  fc?: SensorForecast;
  detail?: SensorDetail;
  failed: boolean;
  error?: string;
}) {
  const r = useReplay();
  if (failed) return <p className="finding">{error ?? "This sensor could not be loaded."}</p>;
  if (!fc || !detail) return <p className="finding"><Ph w={420} h={16} /></p>;
  const now = fc.history.at(-1)?.speed ?? null;
  const pick = fc.forecasts.find((f) => f.horizon_min === (r.h === 0 ? 60 : r.h)) ?? fc.forecasts.at(-1);
  if (!pick) return null;
  const b = band(pick.forecast, detail.free_flow);
  if (pick.forecast == null)
    return <p className="finding">No forecast was saved for this moment: the sensors had no readings to check it against.</p>;
  return (
    <p className="finding">
      {now != null ? <>{speed(now)} mph now; </> : <>No reading now; </>}
      forecast <strong>{speed(pick.forecast)} mph by {formatClock(pick.target_time)}</strong> ({b.label.toLowerCase()}).{" "}
      {pick.actual != null ? (
        <>It was actually {speed(pick.actual)} mph, {Math.abs(Math.round(pick.forecast - pick.actual))} mph {pick.forecast > pick.actual ? "above" : pick.forecast < pick.actual ? "below" : "off"}.</>
      ) : (
        <>There was no reading then to check it against.</>
      )}
    </p>
  );
}

function HorizonStrip({ fc }: { fc: SensorForecast }) {
  const r = useReplay();
  const now = fc.history.at(-1)?.speed ?? null;
  return (
    <div className="kpis odd" style={{ "--n": 5 } as React.CSSProperties} role="group" aria-label="Speed now and forecast by horizon">
      {HORIZONS.map((h) => {
        const f = fc.forecasts.find((x) => x.horizon_min === h);
        const value = h === 0 ? now : (f?.forecast ?? null);
        return (
          <button key={h} type="button" className="kpi" aria-pressed={r.h === h} onClick={() => r.setHorizon(h as Horizon)}>
            <div className="kpi-label">{h === 0 ? `Now, ${formatClock(fc.as_of)}` : `+${h} min, ${formatClock(f?.target_time ?? fc.as_of)}`}</div>
            <div className="kpi-value fade-value" key={`${fc.as_of}-${fc.model}`}>
              {value == null ? <span className="dash">–</span> : speed(value)}
              <span className="unit">mph</span>
            </div>
            <div className="kpi-note">
              {h === 0 ? "observed" : `${speedChange(now, value)} · actual ${speed(f?.actual)}`}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function ForecastLegend() {
  return (
    <div className="chart-legend" aria-hidden="true">
      <span>
                <i style={{ color: "var(--actual)" }} /> Before now
      </span>
      <span>
                <i className="dashed" style={{ color: "var(--forecast)" }} /> Forecast
      </span>
      <span>
                <i style={{ color: "var(--actual-after)" }} /> What happened
      </span>
    </div>
  );
}

function ForecastChart({ fc, freeFlow }: { fc: SensorForecast; freeFlow: number | null }) {
  const r = useReplay();
  const nowMs = toMs(fc.as_of);
  const nowSpeed = fc.history.at(-1)?.speed ?? null;
  const series = useMemo<Series[]>(() => {
    const history = fc.history.map((p) => [toMs(p.time), p.speed] as const);
    const after = [[nowMs, nowSpeed] as const, ...fc.actual_after.map((p) => [toMs(p.time), p.speed] as const)];
    const forecast = [
      ...(nowSpeed != null ? [[nowMs, nowSpeed] as const] : []),
      ...fc.forecasts.map((f) => [toMs(f.target_time), f.forecast] as const),
    ];
    return [
      { key: "history", label: "Observed", color: "var(--actual)", points: history, gapBands: true },
      { key: "after", label: "What happened", color: "var(--actual-after)", points: after, gapBands: true },
      {
        key: "forecast",
        label: "Forecast",
        color: "var(--forecast)",
        points: forecast,
        dashed: true,
        unrollKey: `${fc.model}-${fc.as_of}`,
        endLabel: (y) => `${speed(y)} mph`,
      },
    ];
  }, [fc, nowMs, nowSpeed]);
  const start = toMs(fc.history[0]?.time ?? fc.as_of);
  const end = toMs(fc.forecasts.at(-1)?.target_time ?? fc.as_of);
  const ticks: number[] = [];
  for (let x = Math.ceil(start / 3_600_000) * 3_600_000; x <= end; x += 3_600_000) ticks.push(x);
  const last = fc.forecasts.at(-1);
  return (
    <>
      <LineChart
        series={series}
        height={260}
        xDomain={[start, end]}
        xTicks={ticks}
        formatX={(x) => formatClock(x)}
        now={{ x: nowMs, label: `Now ${formatClock(fc.as_of)}` }}
        reference={freeFlow != null ? { y: freeFlow, label: "Free flow" } : undefined}
        marker={r.h > 0 ? { x: nowMs + r.h * 60_000, seriesKey: "forecast" } : undefined}
        summary={`Speed at sensor ${fc.sensor_id}: ${speed(nowSpeed)} mph at ${formatClock(fc.as_of)}; forecast ${speed(last?.forecast)} mph at ${formatClock(last?.target_time ?? fc.as_of)}; actually ${speed(last?.actual)} mph.`}
      />
      <ChartTable
        caption={`Speeds at sensor ${fc.sensor_id}`}
        columns={["Time", "Observed (mph)", "Forecast (mph)"]}
        rows={[
          ...fc.history.map((p) => [formatClock(p.time), speed(p.speed), "–"] as const),
          ...fc.actual_after.map((p) => {
            const f = fc.forecasts.find((x) => x.target_time === p.time);
            return [formatClock(p.time), speed(p.speed), f ? speed(f.forecast) : "–"] as const;
          }),
        ]}
      />
    </>
  );
}

function Locator({ id, detail }: { id: string; detail?: SensorDetail }) {
  const sensors = useSensors();
  const points = useMemo(
    () =>
      (sensors.data ?? []).map((s) => ({
        id: s.sensor_id,
        lat: s.latitude,
        lon: s.longitude,
        color: s.sensor_id === id ? "var(--text)" : "var(--free-flow)",
        tip: s.sensor_id,
      })),
    [sensors.data, id],
  );
  return (
    <section className="card" aria-labelledby="where-title" style={{ display: "flex", flexDirection: "column" }}>
      <div className="card-head">
        <h2 id="where-title">Where it is</h2>
      </div>
      <div className="map" style={{ height: 200, marginTop: 12 }}>
        {points.length > 0 && <SensorMap points={points} selected={id} radius={4} focus={id} />}
      </div>
      <dl className="facts">
        <div>
          <dt>Free-flow speed</dt>
          <dd className="num">{detail ? <>{speed(detail.free_flow)} mph</> : <Ph w={48} />}</dd>
        </div>
        <div>
          <dt>Congested in training</dt>
          <dd className="num">{detail ? pct1(detail.train_congested_pct) : <Ph w={48} />}</dd>
        </div>
        <div>
          <dt>Readings missing</dt>
          <dd className="num">{detail ? pct1(detail.missing_pct) : <Ph w={48} />}</dd>
        </div>
      </dl>
    </section>
  );
}

function HistoryTab({ id, detail }: { id: string; detail?: SensorDetail }) {
  const r = useReplay();
  const fc = useSensorForecast(id, r.t, r.model, 24);
  return (
    <section className="card" aria-labelledby="hist-title">
      <div className="card-head">
        <h2 id="hist-title">The 24 hours up to now</h2>
        <div className="chart-legend" aria-hidden="true">
          <span>
                <i style={{ color: "var(--actual)" }} /> Observed
          </span>
          <span>
                <i className="dashed" style={{ color: "var(--actual-after)" }} /> Typical weekday hour (training)
          </span>
        </div>
      </div>
      <div className="card-body">
        <Block query={fc} what="this sensor’s history" placeholder={<Ph w="100%" h={260} />}>
          {(data) => {
            const pts = data.history.map((p) => [toMs(p.time), p.speed] as const);
            const start = pts[0]?.[0] ?? toMs(data.as_of);
            const end = toMs(data.as_of);
            const typical = detail
              ? pts
                  .filter(([x]) => new Date(x).getUTCMinutes() === 0)
                  .map(([x]) => [x, detail.weekday_hourly_speed[new Date(x).getUTCHours()] ?? null] as const)
              : [];
            const ticks: number[] = [];
            for (let x = Math.ceil(start / 14_400_000) * 14_400_000; x <= end; x += 14_400_000) ticks.push(x);
            const missing = pts.filter((p) => p[1] == null).length;
            return (
              <>
                <LineChart
                  series={[
                    { key: "typical", label: "Typical weekday", color: "var(--actual-after)", points: typical, dashed: true, width: 1.5 },
                    { key: "observed", label: "Observed", color: "var(--actual)", points: pts, gapBands: true },
                  ]}
                  height={260}
                  xDomain={[start, end]}
                  xTicks={ticks}
                  formatX={(x) => (new Date(x).getUTCHours() === 0 ? formatDay(x) : formatClock(x))}
                  reference={detail?.free_flow != null ? { y: detail.free_flow, label: "Free flow" } : undefined}
                  summary={`Speed at sensor ${id} over the 24 hours to ${formatClock(data.as_of)}; ${missing} of ${pts.length} readings missing.`}
                />
                <ChartTable
                  caption={`Speeds at sensor ${id}, past 24 hours`}
                  columns={["Time", "Observed (mph)"]}
                  rows={data.history.filter((_, i) => i % 3 === 0).map((p) => [`${formatDay(p.time)} ${formatClock(p.time)}`, speed(p.speed)] as const)}
                />
              </>
            );
          }}
        </Block>
      </div>
    </section>
  );
}

const ACCURACY_MODELS = ["DCRNN", "LightGBM", "AR on residuals"] as const;
const MINUTES = [15, 30, 45, 60] as const;

function AccuracyTab({ id }: { id: string }) {
  const queries = useQueries({
    queries: ACCURACY_MODELS.flatMap((model) =>
      MINUTES.map((horizon) => ({
        queryKey: ["sensor-errors", model, horizon, "test"],
        queryFn: () => getJSON<SensorError[]>("/api/metrics/sensors", { model, horizon, split: "test" }),
      })),
    ),
  });
  const failed = queries.find((q) => q.isError);
  const ready = queries.every((q) => q.data);
  const cell = (mi: number, hi: number) => {
    const rows = queries[mi * MINUTES.length + hi]?.data;
    const idx = rows?.findIndex((x) => x.sensor_id === id) ?? -1;
    return rows && idx >= 0 ? { mae: rows[idx]?.mae ?? null, rank: idx + 1, of: rows.length } : null;
  };
  const best = MINUTES.map((_, hi) =>
    Math.min(...ACCURACY_MODELS.map((_, mi) => cell(mi, hi)?.mae ?? Infinity)),
  );
  const d60 = ready ? cell(0, 3) : null;
  return (
    <section className="card" aria-labelledby="acc-title">
      <div className="card-head">
        <h2 id="acc-title">Typical error at this sensor</h2>
        <span className="sub">Masked MAE in mph, test period 4–27 Jun 2012</span>
      </div>
      {failed ? (
        <div className="block-error" role="alert">
          Couldn’t load the per-sensor errors: {failed.error?.message}.
          <button type="button" className="text-button" onClick={() => queries.forEach((q) => void q.refetch())}>
            Retry
          </button>
        </div>
      ) : (
        <>
          <p className="card-body" style={{ paddingBottom: 4 }}>
            {d60 ? (
              <>
                At 60 minutes DCRNN is off by <strong className="num">{dec2(d60.mae ?? 0)} mph</strong> on a typical reading here,
                the {ordinal(d60.rank)} largest error of {int(d60.of)} sensors.
              </>
            ) : (
              <Ph w={380} />
            )}
          </p>
          <div className="table-scroll" style={{ paddingBottom: 12 }}>
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  {MINUTES.map((m) => (
                    <th key={m} scope="col" className="num">
                      {m} min
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ACCURACY_MODELS.map((m, mi) => (
                  <tr key={m}>
                    <th scope="row">{m}</th>
                    {MINUTES.map((_, hi) => {
                      const c = cell(mi, hi);
                      return (
                        <td key={hi} className={`num${c?.mae != null && c.mae === best[hi] ? " best" : ""}`}>
                          {!ready ? <Ph w={40} /> : c?.mae != null ? dec2(c.mae) : <span className="dash">–</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

function ordinal(n: number): string {
  const s = n % 100 >= 11 && n % 100 <= 13 ? "th" : (["th", "st", "nd", "rd"][n % 10] ?? "th");
  return `${int(n)}${s}`;
}

function KpiPlaceholder() {
  return (
    <div className="kpis odd" style={{ "--n": 5 } as React.CSSProperties}>
      {Array.from({ length: 5 }, (_, i) => (
        <div className="kpi" key={i}>
          <Ph w={90} h={12} />
          <Ph w={56} h={28} style={{ marginTop: 8 }} />
          <Ph w={120} h={12} style={{ marginTop: 8 }} />
        </div>
      ))}
    </div>
  );
}

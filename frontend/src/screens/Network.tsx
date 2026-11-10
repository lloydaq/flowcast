// Network: "Where will traffic slow down in the next hour?"
import { Activity, Gauge, Radio, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { Block, Empty, Ph } from "../components/Block";
import { CountUp } from "../components/CountUp";
import { SensorMap, type MapPoint } from "../components/map/SensorMap";
import { Segmented } from "../components/Segmented";
import type { ModelInfo, Sensor } from "../lib/api";
import { int, plural, speed } from "../lib/format";
import { comparedWithNow, summarize, type NetworkSummary, type SensorOutlook } from "../lib/network";
import { useModels, useNetwork, useSensors } from "../lib/queries";
import { HORIZONS, useReplay, type Horizon } from "../lib/replay";
import { BANDS, severityRank } from "../lib/severity";
import { addMinutes, formatClock } from "../lib/time";

const LIST_CAP = 8;
const HORIZON_OPTIONS = HORIZONS.map((h) => ({ value: h, label: h === 0 ? "Now" : `${h} min` }));

export function NetworkScreen() {
  const r = useReplay();
  const sensors = useSensors();
  const models = useModels();
  // "Now" needs only current speeds, which every horizon's response carries.
  const network = useNetwork(r.t, r.h === 0 ? 15 : r.h, r.model);
  const target = r.t ? formatClock(addMinutes(r.t, r.h)) : "";
  const summary = useMemo(() => (network.data ? summarize(network.data, r.h) : undefined), [network.data, r.h]);

  return (
    <>
      <div className="title-row">
        <div>
          <h1>Network</h1>
          <Finding summary={summary} horizon={r.h} target={target} failed={network.isError} />
        </div>
        <div className="controls">
          <Segmented<Horizon> label="Forecast horizon" options={HORIZON_OPTIONS} value={r.h} onChange={r.setHorizon} />
          <ModelSelect models={models.data} value={r.model} onChange={r.setModel} />
        </div>
      </div>

      <div className="grid-2-1" style={{ alignItems: "stretch" }}>
        <section className="card" aria-labelledby="map-title">
          <div className="card-head">
            <h2 id="map-title">{r.h === 0 ? `Speeds at ${target}` : `Forecast for ${target}`}</h2>
            <ModelNote models={models.data} horizon={r.h} />
          </div>
          <div className="map" style={{ marginTop: 12 }}>
            <Block query={sensors} what="the sensor map" placeholder={<div style={{ height: 480 }} />}>
              {(list) => <NetworkMap sensors={list} summary={summary} horizon={r.h} />}
            </Block>
            <Legend />
          </div>
        </section>
        <section className="card" aria-labelledby="watch-title" style={{ display: "flex", flexDirection: "column" }}>
          <div className="card-head">
            <h2 id="watch-title">{r.h === 0 ? "Slowest now" : "Slowing next"}</h2>
            {summary && <span className="sub num">{plural(summary.watch.length, "sensor")}</span>}
          </div>
          <Block query={network} what="the network forecast" placeholder={<RowsPlaceholder />}>
            {() => (summary ? <WatchList summary={summary} horizon={r.h} target={target} /> : null)}
          </Block>
        </section>
      </div>

      <Block query={network} what="the network figures" placeholder={<KpiPlaceholder />}>
        {() => (summary ? <NetworkKpis summary={summary} horizon={r.h} target={target} /> : null)}
      </Block>
    </>
  );
}

function Finding({
  summary,
  horizon,
  target,
  failed,
}: {
  summary?: NetworkSummary;
  horizon: number;
  target: string;
  failed: boolean;
}) {
  if (!summary) return <p className="finding">{failed ? "The forecast could not be loaded." : <Ph w={420} h={16} />}</p>;
  if (summary.forecasts === 0)
    return (
      <p className="finding">
        No forecasts for {target}: almost no sensor was reporting at this moment. Step the clock on.
      </p>
    );
  if (horizon === 0)
    return (
      <p className="finding">
        <strong>{plural(summary.congestedNow, "sensor")}</strong> congested at {target}, out of{" "}
        {int(summary.reporting)} reporting.
      </p>
    );
  return (
    <p className="finding">
      <strong>{plural(summary.congestedThen, "sensor")}</strong> forecast congested by <strong>{target}</strong>,{" "}
      {comparedWithNow(summary.congestedThen, summary.congestedNow)}.
    </p>
  );
}

function ModelSelect({
  models,
  value,
  onChange,
}: {
  models?: ModelInfo[];
  value?: string;
  onChange: (m: string) => void;
}) {
  return (
    <select className="select" aria-label="Forecasting model" value={value ?? ""} onChange={(e) => onChange(e.target.value)} disabled={!models}>
      {(models ?? []).map((m) => (
        <option key={m.name} value={m.name}>
          {m.label}
        </option>
      ))}
    </select>
  );
}

function NetworkMap({
  sensors,
  summary,
  horizon,
}: {
  sensors: Sensor[];
  summary?: NetworkSummary;
  horizon: number;
}) {
  const navigate = useNavigate();
  const r = useReplay();
  const target = r.t ? formatClock(addMinutes(r.t, horizon)) : "";
  const points = useMemo<MapPoint[]>(() => {
    const byId = new Map(summary?.sensors.map((s) => [s.point.sensor_id, s]));
    return sensors.map((s) => {
      const o = byId.get(s.sensor_id);
      const b = o?.then;
      const value = horizon === 0 ? o?.point.speed_now : o?.point.forecast;
      const tip = !o
        ? s.sensor_id
        : horizon === 0
          ? `${s.sensor_id} · ${b?.label ?? ""} · ${speed(o.point.speed_now)} mph`
          : `${s.sensor_id} · ${b?.label ?? ""} by ${target} · ${speed(o.point.speed_now)} → ${speed(value)} mph`;
      return {
        id: s.sensor_id,
        lat: s.latitude,
        lon: s.longitude,
        color: b?.color ?? "var(--free-flow)",
        hollow: !b || b.key === "missing",
        tip,
        details: o
          ? [
              [horizon === 0 ? "State now" : `State by ${target}`, b?.label ?? "–"],
              ["Speed now", o.point.speed_now == null ? "No reading" : `${speed(o.point.speed_now)} mph`],
              ...(horizon > 0 ? ([[`Forecast, ${target}`, value == null ? "No forecast" : `${speed(value)} mph`]] as const) : []),
              ["Free-flow speed", `${speed(s.free_flow)} mph`],
            ]
          : undefined,
      };
    });
  }, [sensors, summary, horizon, target]);
  return <SensorMap points={points} onSelect={(id) => navigate(r.href(`/sensor/${id}`))} />;
}

/** The model behind the map and its measured test-period error at this horizon. */
function ModelNote({ models, horizon }: { models?: ModelInfo[]; horizon: number }) {
  const r = useReplay();
  const model = models?.find((m) => m.name === r.model);
  if (!model) return <span className="sub" />;
  const mae = horizon > 0 ? model.test_mae[String(horizon) as keyof typeof model.test_mae] : undefined;
  return (
    <span className="sub">
      {model.label}
      {mae != null && <> · typical error {mae.toFixed(2)} mph at {horizon} min</>} · <Link to={r.href("/models")}>Compare models</Link>
    </span>
  );
}

function Legend() {
  return (
    <div className="legend" aria-label="Map key">
      {BANDS.map((b) => (
        <span key={b.key}>
          <i className={`swatch${b.key === "missing" ? " ring" : ""}`} style={{ background: b.color }} />
          {b.label}
        </span>
      ))}
    </div>
  );
}

function WatchList({ summary, horizon, target }: { summary: NetworkSummary; horizon: number; target: string }) {
  const r = useReplay();
  const [all, setAll] = useState(false);
  if (summary.watch.length === 0)
    return (
      <Empty>
        {horizon === 0 ? `No sensor is slower than free flow at ${target}.` : `No sensor is forecast to slow down by ${target}.`}
      </Empty>
    );
  const rows = all ? summary.watch : summary.watch.slice(0, LIST_CAP);
  return (
    <>
      <ul className="rows" style={{ flex: 1, overflowY: all ? "auto" : undefined, maxHeight: all ? 432 : undefined }}>
        {rows.map((o) => (
          <WatchRow key={o.point.sensor_id} o={o} horizon={horizon} target={target} href={r.href(`/sensor/${o.point.sensor_id}`)} />
        ))}
      </ul>
      {summary.watch.length > LIST_CAP && (
        <div style={{ padding: "8px 20px 16px", borderTop: "1px solid var(--rule)" }}>
          <button type="button" className="text-button" onClick={() => setAll((v) => !v)}>
            {all ? "Show the worst 8" : `View all ${int(summary.watch.length)}`}
          </button>
        </div>
      )}
    </>
  );
}

function WatchRow({ o, horizon, target, href }: { o: SensorOutlook; horizon: number; target: string; href: string }) {
  const value = horizon === 0 ? o.point.speed_now : o.point.forecast;
  return (
    <li>
      <Link className="row-link" to={href}>
        <i className="swatch" style={{ background: o.then.color }} aria-hidden="true" />
        <span>
          <span className="mono">{o.point.sensor_id}</span>
          <span className="row-sub">
            {o.then.label}
            {horizon === 0 ? " now" : ` by ${target}`}
          </span>
        </span>
        <span className="num" style={{ textAlign: "right" }}>
          {horizon === 0 ? (
            <>
              {speed(value)}
              <span className="unit">mph</span>
            </>
          ) : (
            <>
              {speed(o.point.speed_now)} → {speed(value)}
              <span className="unit">mph</span>
            </>
          )}
        </span>
      </Link>
    </li>
  );
}

function NetworkKpis({ summary, horizon, target }: { summary: NetworkSummary; horizon: number; target: string }) {
  const slower = summary.sensors.filter((o) => severityRank(o.now) > 0).length;
  return (
    <section className="kpis" style={{ "--n": 4 } as React.CSSProperties} aria-label="Network overview">
      <div className="kpi">
        <div className="kpi-label">
          <Radio aria-hidden="true" />
          Sensors
        </div>
        <div className="kpi-value">
          <CountUp value={summary.total} />
        </div>
        <div className="kpi-note">on the METR-LA highway network</div>
      </div>
      <div className="kpi">
        <div className="kpi-label">
          <TriangleAlert aria-hidden="true" />
          Congested now
        </div>
        <div className="kpi-value">
          <CountUp value={summary.congestedNow} />
        </div>
        <div className="kpi-note">{horizon === 0 ? "below 60% of free-flow speed" : `${int(summary.congestedThen)} forecast by ${target}`}</div>
      </div>
      <div className="kpi">
        <div className="kpi-label">
          <Gauge aria-hidden="true" />
          Slower than free flow
        </div>
        <div className="kpi-value">
          <CountUp value={slower} />
        </div>
        <div className="kpi-note">slowing or worse, now</div>
      </div>
      <div className="kpi">
        <div className="kpi-label">
          <Activity aria-hidden="true" />
          Reporting
        </div>
        <div className="kpi-value">
          <CountUp value={summary.reporting} />
        </div>
        <div className="kpi-note">
          {int(Math.round((summary.reporting / summary.total) * 100))}% of sensors have a reading now
        </div>
      </div>
    </section>
  );
}

function RowsPlaceholder() {
  return (
    <ul className="rows">
      {Array.from({ length: LIST_CAP }, (_, i) => (
        <li key={i} className="row-link" style={{ gridTemplateColumns: "10px 1fr 80px" }}>
          <Ph w={10} h={10} />
          <span>
            <Ph w={64} />
            <Ph w={110} h={12} style={{ marginTop: 4 }} />
          </span>
          <Ph w={72} />
        </li>
      ))}
    </ul>
  );
}

function KpiPlaceholder() {
  return (
    <div className="kpis" style={{ "--n": 4 } as React.CSSProperties}>
      {Array.from({ length: 4 }, (_, i) => (
        <div className="kpi" key={i}>
          <Ph w={110} h={12} />
          <Ph w={64} h={28} style={{ marginTop: 8 }} />
          <Ph w={140} h={12} style={{ marginTop: 8 }} />
        </div>
      ))}
    </div>
  );
}

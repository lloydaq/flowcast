// Patterns: "When and where does congestion happen?" All from the training period (1 Mar – 23 May 2012).
import { Link } from "react-router";
import { Block, Ph } from "../components/Block";
import { ChartTable, LineChart } from "../components/charts/LineChart";
import type { DailyProfile, Sensor, WeeklyCongestion } from "../lib/api";
import { dec1, speed } from "../lib/format";
import { useDailyProfile, useSensors, useWeeklyCongestion } from "../lib/queries";
import { useReplay } from "../lib/replay";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const hh = (h: number) => `${String(h).padStart(2, "0")}:00`;

/** Congestion share bins for the grid: one red ramp, darker (light theme) or brighter (dark) = more. */
const BINS = [
  { min: 25, color: "var(--cong-4)", label: "25% or more" },
  { min: 15, color: "var(--cong-3)", label: "15–25%" },
  { min: 8, color: "var(--cong-2)", label: "8–15%" },
  { min: 3, color: "var(--cong-1)", label: "3–8%" },
  { min: 0, color: "var(--raised)", label: "under 3%" },
] as const;
const binOf = (v: number | null) => BINS.find((b) => (v ?? 0) >= b.min) ?? BINS[4];

export function PatternsScreen() {
  const weekly = useWeeklyCongestion();
  const profile = useDailyProfile();
  const sensors = useSensors();
  return (
    <>
      <div className="title-row">
        <div>
          <h1>Patterns</h1>
          <Block query={weekly} what="the weekly pattern" placeholder={<p className="finding"><Ph w={460} h={16} /></p>}>
            {(w) => <PatternsFinding w={w} />}
          </Block>
        </div>
      </div>

      <section className="card" aria-labelledby="week-title">
        <div className="card-head">
          <h2 id="week-title">Share of readings congested, by day and hour</h2>
          <div className="chart-legend">
            {[...BINS].reverse().map((b) => (
              <span key={b.label}>
                <i style={{ width: 10, height: 10, border: 0, borderRadius: 2, background: b.color }} />
                {b.label}
              </span>
            ))}
          </div>
        </div>
        <div className="card-body">
          <Block query={weekly} what="the weekly pattern" placeholder={<Ph w="100%" h={200} />}>
            {(w) => <WeekGrid w={w} />}
          </Block>
        </div>
      </section>

      <div className="grid-2-1" style={{ alignItems: "stretch" }}>
        <section className="card" aria-labelledby="day-title">
          <div className="card-head">
            <h2 id="day-title">Average network speed through the day</h2>
            <div className="chart-legend" aria-hidden="true">
              <span>
                <i style={{ color: "var(--actual)" }} /> Weekdays
              </span>
              <span>
                <i style={{ color: "var(--actual-after)" }} /> Weekends
              </span>
            </div>
          </div>
          <div className="card-body">
            <Block query={profile} what="the daily profile" placeholder={<Ph w="100%" h={260} />}>
              {(p) => <DayChart p={p} />}
            </Block>
          </div>
        </section>
        <section className="card" aria-labelledby="where-title">
          <div className="card-head">
            <h2 id="where-title">Most often congested</h2>
          </div>
          <Block query={sensors} what="the sensors" placeholder={<Ph w="100%" h={300} />}>
            {(list) => <CongestedSensors list={list} />}
          </Block>
        </section>
      </div>
    </>
  );
}

function PatternsFinding({ w }: { w: WeeklyCongestion }) {
  let best = { d: 0, h: 0, v: -1 };
  w.rate_pct.forEach((row, d) => row.forEach((v, h) => v != null && v > best.v && (best = { d, h, v })));
  const weekend = w.rate_pct.slice(5).flat().filter((v): v is number => v != null);
  const weekendMax = Math.max(...weekend);
  return (
    <p className="finding">
      <strong>
        {DAY_NAMES[best.d]} at {hh(best.h)} is the worst hour: {dec1(best.v)}% of readings congested.
      </strong>{" "}
      Weekends never pass {dec1(weekendMax)}%. Training period, 1 March to 23 May 2012.
    </p>
  );
}

function WeekGrid({ w }: { w: WeeklyCongestion }) {
  return (
    <>
      <div className="matrix" role="img" aria-label="Congested share of readings for each day of the week and hour of the day; the table below has the numbers.">
        <span />
        {Array.from({ length: 24 }, (_, h) => (
          <span key={h} className="hour num" aria-hidden="true">
            {h % 3 === 0 ? String(h).padStart(2, "0") : ""}
          </span>
        ))}
        {w.rate_pct.map((row, d) => (
          <div key={d} style={{ display: "contents" }}>
            <span className="day">{DAYS[d]}</span>
            {row.map((v, h) => (
              <span
                key={h}
                className="cell"
                title={`${DAY_NAMES[d]} ${hh(h)}: ${v == null ? "no readings" : `${dec1(v)}% congested`}`}
                style={{ background: binOf(v).color }}
              />
            ))}
          </div>
        ))}
      </div>
      <ChartTable
        caption="Congested share of readings by day and hour, training period"
        columns={["Hour", ...DAYS]}
        rows={Array.from({ length: 24 }, (_, h) => [hh(h), ...w.rate_pct.map((row) => `${dec1(row[h] ?? 0)}%`)])}
      />
    </>
  );
}

function DayChart({ p }: { p: DailyProfile }) {
  const pts = (arr: (number | null)[]) => arr.map((v, i) => [i * p.slot_minutes, v] as const);
  const weekday = pts(p.weekday);
  const min = weekday.reduce((a, b) => ((b[1] ?? Infinity) < (a[1] ?? Infinity) ? b : a));
  const clock = (m: number) => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
  return (
    <>
      <LineChart
        series={[
          { key: "weekend", label: "Weekends", color: "var(--actual-after)", points: pts(p.weekend) },
          { key: "weekday", label: "Weekdays", color: "var(--actual)", points: weekday },
        ]}
        height={300}
        xDomain={[0, 24 * 60 - p.slot_minutes]}
        xTicks={[0, 240, 480, 720, 960, 1200]}
        formatX={clock}
        summary={`Average speed across all sensors by time of day. On weekdays it falls lowest at ${clock(min[0])}, ${speed(min[1])} mph.`}
      />
      <ChartTable
        caption="Average network speed by hour, training period"
        columns={["Time", "Weekdays (mph)", "Weekends (mph)"]}
        rows={Array.from({ length: 24 }, (_, h) => [hh(h), speed(p.weekday[h * 12]), speed(p.weekend[h * 12])])}
      />
    </>
  );
}

function CongestedSensors({ list }: { list: Sensor[] }) {
  const r = useReplay();
  const top = [...list].sort((a, b) => (b.train_congested_pct ?? 0) - (a.train_congested_pct ?? 0)).slice(0, 6);
  const max = top[0]?.train_congested_pct ?? 1;
  return (
    <ol className="bar-list" aria-label="Sensors most often congested in training">
      {top.map((s) => (
        <li key={s.sensor_id}>
          <Link to={r.href(`/sensor/${s.sensor_id}`, { tab: "history" })}>
            <span className="bar-label">
              <span className="mono">{s.sensor_id}</span>
            </span>
            <span className="bar-track">
              <span className="bar" style={{ width: `${((s.train_congested_pct ?? 0) / max) * 82}%`, background: "var(--cong-2)" }} />
              <span className="num">{dec1(s.train_congested_pct ?? 0)}%</span>
            </span>
          </Link>
        </li>
      ))}
    </ol>
  );
}

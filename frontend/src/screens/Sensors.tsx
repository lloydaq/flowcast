// Sensors: "Which sensors are slowest, now and at the chosen horizon?" A sortable, filterable ranking.
import { ArrowDown, ArrowUp, Search } from "lucide-react";
import { useDeferredValue, useMemo, useState } from "react";
import { Link } from "react-router";
import { Block, Empty, Ph } from "../components/Block";
import { Segmented } from "../components/Segmented";
import { int, plural, speed } from "../lib/format";
import { summarize, type SensorOutlook } from "../lib/network";
import { useNetwork } from "../lib/queries";
import { HORIZONS, useReplay, type Horizon } from "../lib/replay";
import { BANDS, type BandKey } from "../lib/severity";
import { addMinutes, formatClock } from "../lib/time";

type SortKey = "rank" | "id" | "now" | "then" | "change";
type Row = { o: SensorOutlook; rank: number; value: number | null | undefined };
const HORIZON_OPTIONS = HORIZONS.map((h) => ({ value: h, label: h === 0 ? "Now" : `${h} min` }));

export function SensorsScreen() {
  const r = useReplay();
  const network = useNetwork(r.t, r.h === 0 ? 15 : r.h, r.model);
  const [query, setQuery] = useState("");
  const search = useDeferredValue(query.trim());
  const [state, setState] = useState<BandKey | "all">("all");
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "rank", asc: true });
  const target = r.t ? formatClock(addMinutes(r.t, r.h)) : "";

  const ranked = useMemo(() => {
    if (!network.data) return undefined;
    const value = (o: SensorOutlook) => (r.h === 0 ? o.point.speed_now : o.point.forecast);
    // rank 1 = slowest against its own free flow at the horizon; sensors without a value rank last
    const ratio = (o: SensorOutlook) => (value(o) ?? Infinity) / (o.point.free_flow ?? 1);
    return summarize(network.data, r.h)
      .sensors.slice()
      .sort((a, b) => ratio(a) - ratio(b))
      .map((o, i) => ({ o, rank: i + 1, value: value(o) }));
  }, [network.data, r.h]);

  const rows = useMemo(() => {
    if (!ranked) return [];
    const num = (v: number | null | undefined, asc: boolean) => v ?? (asc ? Infinity : -Infinity);
    const by: Record<SortKey, (x: Row) => number | string> = {
      rank: (x) => x.rank,
      id: (x) => x.o.point.sensor_id,
      now: (x) => num(x.o.point.speed_now, sort.asc),
      then: (x) => num(x.value, sort.asc),
      change: (x) => num(x.o.drop == null ? null : -x.o.drop, sort.asc),
    };
    const f = by[sort.key];
    return ranked
      .filter((x) => (state === "all" || x.o.then.key === state) && x.o.point.sensor_id.includes(search))
      .sort((a, b) => {
        const [x, y] = [f(a), f(b)];
        const d = typeof x === "string" ? x.localeCompare(String(y)) : x - Number(y);
        return sort.asc ? d : -d;
      });
  }, [ranked, sort, state, search]);

  const th = (key: SortKey, label: string, numeric = true) => (
    <th scope="col" className={numeric ? "num" : undefined} aria-sort={sort.key === key ? (sort.asc ? "ascending" : "descending") : "none"}>
      <button type="button" className="sort" onClick={() => setSort((s) => ({ key, asc: s.key === key ? !s.asc : true }))}>
        {label}
        {sort.key === key && (sort.asc ? <ArrowUp aria-hidden="true" /> : <ArrowDown aria-hidden="true" />)}
      </button>
    </th>
  );

  return (
    <>
      <div className="title-row">
        <div>
          <h1>Sensors</h1>
          <p className="finding">
            {ranked ? (
              <>
                All {int(ranked.length)} sensors ranked by speed against their own free flow, {r.h === 0 ? `at ${target}` : `forecast for ${target}`}.
              </>
            ) : (
              <Ph w={420} h={16} />
            )}
          </p>
        </div>
        <div className="controls">
          <Segmented<Horizon> label="Forecast horizon" options={HORIZON_OPTIONS} value={r.h} onChange={r.setHorizon} />
        </div>
      </div>

      <section className="card" aria-labelledby="rank-title">
        <div className="card-head">
          <h2 id="rank-title">Ranking</h2>
          <div className="controls">
            <label className="search inline">
              <Search aria-hidden="true" />
              <input aria-label="Filter by sensor ID" placeholder="Filter by ID" inputMode="numeric" value={query} onChange={(e) => setQuery(e.target.value)} />
            </label>
            <select className="select" aria-label="Filter by traffic state" value={state} onChange={(e) => setState(e.target.value as BandKey | "all")}>
              <option value="all">All states</option>
              {BANDS.map((b) => (
                <option key={b.key} value={b.key}>
                  {b.label}
                </option>
              ))}
            </select>
            <span className="sub num">{ranked ? `${plural(rows.length, "sensor")} shown` : ""}</span>
          </div>
        </div>
        <Block query={network} what="the sensor ranking" placeholder={<div style={{ padding: 20 }}><Ph w="100%" h={420} /></div>}>
          {() =>
            rows.length === 0 ? (
              <Empty>No sensor matches this filter.</Empty>
            ) : (
              <div className="table-scroll tall">
                <table className="data sticky">
                  <thead>
                    <tr>
                      {th("rank", "Rank")}
                      {th("id", "Sensor", false)}
                      <th scope="col">{r.h === 0 ? "State now" : `State by ${target}`}</th>
                      {th("now", "Now")}
                      {r.h > 0 && th("then", `In ${r.h} min`)}
                      {r.h > 0 && th("change", "Change")}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(({ o, rank, value }) => (
                      <tr key={o.point.sensor_id}>
                        <td className="num muted">{rank}</td>
                        <th scope="row">
                          <Link className="mono row-id" to={r.href(`/sensor/${o.point.sensor_id}`)}>
                            {o.point.sensor_id}
                          </Link>
                        </th>
                        <td>
                          <span className="state">
                            <i className={`swatch${o.then.key === "missing" ? " ring" : ""}`} style={{ background: o.then.color }} aria-hidden="true" />
                            {o.then.label}
                          </span>
                        </td>
                        <td className="num">
                          {speed(o.point.speed_now)}
                          {o.point.speed_now != null && <span className="unit">mph</span>}
                        </td>
                        {r.h > 0 && (
                          <td className="num">
                            {speed(value)}
                            {value != null && <span className="unit">mph</span>}
                          </td>
                        )}
                        {r.h > 0 && (
                          <td className="num">
                            {o.drop == null ? <span className="dash">–</span> : Math.round(-o.drop) === 0 ? "0" : `${-o.drop > 0 ? "+" : "−"}${Math.abs(Math.round(o.drop))}`}
                            {o.drop != null && <span className="unit">mph</span>}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          }
        </Block>
      </section>
    </>
  );
}

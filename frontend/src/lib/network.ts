// What the Network screen says about one forecast: counts, the finding and the "Slowing next" list.
import type { NetworkForecast, NetworkPoint } from "./api";
import { band, isCongested, severityRank, type Band } from "./severity";

export interface SensorOutlook {
  point: NetworkPoint;
  now: Band;
  then: Band; // at the chosen horizon (= now when the horizon is 0)
  drop: number | null; // mph lost from now to the forecast
}

export interface NetworkSummary {
  sensors: SensorOutlook[];
  total: number;
  reporting: number; // sensors with a reading at the as-of time
  forecasts: number; // sensors with a forecast at the horizon
  congestedNow: number;
  congestedThen: number;
  /** Sensors worth watching, worst first: slowing or worse at the horizon (or now, for horizon 0). */
  watch: SensorOutlook[];
}

export function summarize(fc: NetworkForecast, horizon: number): NetworkSummary {
  const sensors = fc.sensors.map((p): SensorOutlook => {
    const value = horizon === 0 ? p.speed_now : p.forecast;
    return {
      point: p,
      now: band(p.speed_now, p.free_flow),
      then: band(value, p.free_flow),
      drop: p.speed_now != null && value != null ? p.speed_now - value : null,
    };
  });
  const value = (s: SensorOutlook) => (horizon === 0 ? s.point.speed_now : s.point.forecast);
  const ratio = (s: SensorOutlook) => (value(s) ?? Infinity) / (s.point.free_flow ?? 1);
  const watch = sensors
    .filter((s) => severityRank(s.then) > 0)
    .sort((a, b) =>
      horizon === 0
        ? ratio(a) - ratio(b)
        : severityRank(b.then) - severityRank(a.then) || (b.drop ?? -Infinity) - (a.drop ?? -Infinity),
    );
  return {
    sensors,
    total: sensors.length,
    reporting: sensors.filter((s) => s.point.speed_now != null).length,
    forecasts: sensors.filter((s) => value(s) != null).length,
    congestedNow: sensors.filter((s) => isCongested(s.point.speed_now, s.point.free_flow)).length,
    congestedThen: sensors.filter((s) => isCongested(value(s), s.point.free_flow)).length,
    watch,
  };
}

/** The change in words: "19 more than now", "4 fewer than now", "the same as now". */
export function comparedWithNow(then: number, now: number): string {
  const d = then - now;
  if (d === 0) return "the same number as now";
  return `${Math.abs(d)} ${d > 0 ? "more" : "fewer"} than now`;
}

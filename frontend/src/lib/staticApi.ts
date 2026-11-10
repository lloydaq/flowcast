// The API, answered from static files: used by the GitHub Pages demo (VITE_STATIC=1), where no server
// runs. scripts/export_static.py writes the files. Small answers are the API's own JSON; forecasts and
// speeds are rebuilt here from per-day uint16 grids (hundredths of a mph, 65535 = missing) exactly as
// src/flowcast/api/store.py builds them. staticApi.test.ts checks the rebuild against real API answers.
import { ApiError } from "./api-error";

interface Grid {
  slots_per_day: number;
  missing: number;
  scale: number;
  sensor_ids: string[];
  free_flow: number[]; // unrounded, as the API compares
  horizons_min: number[];
  models: string[];
  speed_days: string[];
  forecast_days: string[];
}

type Params = Record<string, string | number | undefined>;
type Fetch = (url: string) => Promise<Response>;

const STEP_MS = 5 * 60_000;
const DAY_MS = 24 * 3_600_000;

export function createStaticApi(base: string, fetcher: Fetch = (u) => fetch(u)) {
  const cache = new Map<string, Promise<unknown>>();
  const load = <T>(file: string, binary = false): Promise<T> => {
    let p = cache.get(file) as Promise<T> | undefined;
    if (!p) {
      p = fetcher(base + file).then(async (res) => {
        if (!res.ok) throw new ApiError(`missing data file ${file}`, res.status);
        return (binary ? new Uint16Array(await res.arrayBuffer()) : await res.json()) as T;
      });
      p.catch(() => cache.delete(file)); // a failed load can be retried
      cache.set(file, p);
    }
    return p;
  };
  const grid = () => load<Grid>("grid.json");
  const health = () => load<{ replay_start: string; replay_end: string }>("health.json");

  // ---- validation: the same rules and messages as the API ----
  const echo = (v: string) => {
    const s = v.length <= 40 ? v : `${v.slice(0, 40)}…`;
    return `'${s.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
  };
  const pyList = (xs: (string | number)[]) =>
    `[${xs.map((x) => (typeof x === "string" ? `'${x}'` : String(x))).join(", ")}]`;
  const invalid = (message: string) => new ApiError(message, 422);
  const toMs = (t: string) => Date.parse(`${t.slice(0, 19).padEnd(19, ":00").slice(0, 19)}Z`);
  const iso = (ms: number) => new Date(ms).toISOString().slice(0, 19);
  const spaced = (t: string) => t.replace("T", " ");

  async function model(name: string | undefined): Promise<string> {
    const g = await grid();
    if (name === undefined) return g.models[0] as string;
    if (!g.models.includes(name)) throw invalid(`unknown model ${echo(name)}; expected one of ${pyList([...g.models].sort())}`);
    return name;
  }
  async function asOf(raw: string | undefined): Promise<number> {
    if (raw === undefined || Number.isNaN(toMs(raw))) throw invalid("as_of must be a date and time, e.g. 2012-06-04T08:00");
    if (/(Z|[+-]\d\d:?\d\d)$/.test(raw)) throw invalid("as_of must be a local Los Angeles time without a timezone, e.g. 2012-06-04T08:00");
    const ms = toMs(raw);
    if (ms % STEP_MS !== 0) throw invalid("as_of must fall on the 5-minute grid, e.g. 08:00 or 08:05");
    const h = await health();
    if (ms < toMs(h.replay_start) || ms > toMs(h.replay_end))
      throw invalid(`as_of must be between ${spaced(h.replay_start)} and ${spaced(h.replay_end)} (the test period)`);
    return ms;
  }
  async function horizonIndex(raw: string | number | undefined): Promise<number> {
    const g = await grid();
    const i = g.horizons_min.indexOf(Number(raw));
    if (i < 0) throw invalid(`horizon must be one of ${pyList(g.horizons_min)} minutes`);
    return i;
  }
  async function node(sensorId: string): Promise<number> {
    const i = (await grid()).sensor_ids.indexOf(sensorId);
    if (i < 0) throw new ApiError(`unknown sensor ${echo(sensorId)}`, 404);
    return i;
  }

  // ---- grids ----
  const value = (g: Grid, v: number | undefined) => (v === undefined || v === g.missing ? null : v / g.scale);
  const daySlot = (ms: number) => ({ day: iso(ms - (ms % DAY_MS)).slice(0, 10), slot: (ms % DAY_MS) / STEP_MS });
  async function speeds(ms: number): Promise<{ g: Grid; row: Uint16Array }> {
    const g = await grid();
    const { day, slot } = daySlot(ms);
    const all = await load<Uint16Array>(`speed/${day}.bin`, true);
    const n = g.sensor_ids.length;
    return { g, row: all.subarray(slot * n, (slot + 1) * n) };
  }
  async function forecasts(name: string, ms: number): Promise<Uint16Array> {
    const g = await grid();
    const { day, slot } = daySlot(ms);
    const all = await load<Uint16Array>(`forecast/${name}/${day}.bin`, true);
    const n = g.sensor_ids.length * g.horizons_min.length;
    return all.subarray(slot * n, (slot + 1) * n);
  }

  async function network(p: Params) {
    const [m, t, hi] = [await model(p.model as string | undefined), await asOf(p.as_of as string), await horizonIndex(p.horizon)];
    const g = await grid();
    const sensorsJson = await load<{ sensor_id: string; free_flow: number | null }[]>("sensors.json");
    const minutes = g.horizons_min[hi] as number;
    const [now, actual, fc] = await Promise.all([speeds(t), speeds(t + minutes * 60_000), forecasts(m, t)]);
    const H = g.horizons_min.length;
    return {
      model: m,
      as_of: iso(t),
      horizon_min: minutes,
      target_time: iso(t + minutes * 60_000),
      sensors: g.sensor_ids.map((sensor_id, n) => {
        const f = value(g, fc[n * H + hi]);
        return {
          sensor_id,
          speed_now: value(g, now.row[n]),
          forecast: f,
          actual: value(g, actual.row[n]),
          free_flow: sensorsJson[n]?.free_flow ?? null,
          congested: f === null ? null : f < 0.6 * (g.free_flow[n] as number),
        };
      }),
    };
  }

  async function sensorForecast(sensorId: string, p: Params) {
    const n = await node(sensorId);
    const [m, t] = [await model(p.model as string | undefined), await asOf(p.as_of as string)];
    const hours = p.history_hours === undefined ? 3 : Number(p.history_hours);
    if (!Number.isInteger(hours) || hours < 1 || hours > 24) throw invalid("history_hours must be a whole number from 1 to 24");
    const g = await grid();
    const detail = await load<{ free_flow: number | null }>(`sensor/${sensorId}.json`);
    const at = async (ms: number) => value(g, (await speeds(ms)).row[n]);
    const reading = async (ms: number) => ({ time: iso(ms), speed: await at(ms) });
    const steps = (hours * 60) / 5;
    const fc = await forecasts(m, t);
    const H = g.horizons_min.length;
    return {
      model: m,
      sensor_id: sensorId,
      as_of: iso(t),
      free_flow: detail.free_flow,
      history: await Promise.all(Array.from({ length: steps + 1 }, (_, i) => reading(t - (steps - i) * STEP_MS))),
      forecasts: await Promise.all(
        g.horizons_min.map(async (minutes, i) => ({
          horizon_min: minutes,
          target_time: iso(t + minutes * 60_000),
          forecast: value(g, fc[n * H + i]),
          actual: await at(t + minutes * 60_000),
        })),
      ),
      actual_after: await Promise.all(Array.from({ length: 12 }, (_, i) => reading(t + (i + 1) * STEP_MS))),
    };
  }

  /** The API's GET, answered from the static files. */
  return async function get(path: string, params: Params = {}): Promise<unknown> {
    const parts = path.replace(/^\/api\//, "").split("/").map(decodeURIComponent);
    const [a, b] = parts;
    if (a === "health" && !b) return health();
    if (a === "sensors" && !b) return load("sensors.json");
    if (a === "sensors" && b) return (await node(b), load(`sensor/${b}.json`));
    if (a === "models") return load("models.json");
    if (a === "metrics" && !b) return load("metrics.json");
    if (a === "metrics" && b === "sensors") {
      const metrics = await load<{ models: { model: string }[] }>("metrics.json");
      const name = String(params.model ?? "");
      const split = String(params.split ?? "test");
      if (!metrics.models.some((x) => x.model === name))
        throw invalid(`unknown evaluated model ${echo(name)}; expected one of ${pyList(metrics.models.map((x) => x.model))}`);
      if (split !== "validation" && split !== "test") throw invalid("split must be 'validation' or 'test'");
      const h = (await grid()).horizons_min[await horizonIndex(params.horizon)];
      return load(`sensor-errors/${encodeURIComponent(name)}/${split}/${h}.json`);
    }
    if (a === "analytics" && b === "daily-profile") return load("daily-profile.json");
    if (a === "analytics" && b === "weekly-congestion") return load("weekly-congestion.json");
    if (a === "forecast" && !b) return network(params);
    if (a === "forecast" && b) return sensorForecast(b, params);
    throw new ApiError("Not Found", 404);
  };
}

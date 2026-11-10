// Typed client for the FlowCast API. Types are generated from the backend's OpenAPI schema
// (python scripts/export_openapi.py && npm run api:types), so a backend change breaks the build.
import type { components } from "./api-types";

type S = components["schemas"];
export type Health = S["Health"];
export type Sensor = S["Sensor"];
export type SensorDetail = S["SensorDetail"];
export type NetworkForecast = S["NetworkForecast"];
export type NetworkPoint = S["NetworkPoint"];
export type SensorForecast = S["SensorForecast"];
export type Reading = S["Reading"];
export type ModelInfo = S["ModelInfo"];
export type Metrics = S["Metrics"];
export type ModelScores = S["ModelScores"];
export type HorizonScores = S["HorizonScores"];
export type Gap = S["Gap"];
export type SensorError = S["SensorError"];
export type DailyProfile = S["DailyProfile"];
export type WeeklyCongestion = S["WeeklyCongestion"];

const BASE: string = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function getJSON<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const query = new URLSearchParams();
  for (const [k, v] of Object.entries(params ?? {})) if (v !== undefined) query.set(k, String(v));
  const qs = query.toString();
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}${qs ? `?${qs}` : ""}`);
  } catch {
    throw new ApiError("The forecast service is not reachable", 0);
  }
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

// Congestion severity from relative speed (speed ÷ the sensor's free-flow speed), architect.md §10.3.
// Free flow is a quiet grey; only slowdowns get colour, and the colour gets stronger as it gets worse.

export type BandKey = "free" | "slowing" | "congested" | "heavy" | "stop" | "missing";

export interface Band {
  key: BandKey;
  label: string;
  color: string; // a CSS custom property reference
}

export const BANDS: readonly Band[] = [
  { key: "free", label: "Free flow", color: "var(--free-flow)" },
  { key: "slowing", label: "Slowing", color: "var(--cong-1)" },
  { key: "congested", label: "Congested", color: "var(--cong-2)" },
  { key: "heavy", label: "Heavy", color: "var(--cong-3)" },
  { key: "stop", label: "Stop-and-go", color: "var(--cong-4)" },
  { key: "missing", label: "No reading", color: "var(--surface)" },
];

const BY_KEY = new Map(BANDS.map((b) => [b.key, b]));
const get = (k: BandKey): Band => BY_KEY.get(k) as Band;

export function band(speed: number | null | undefined, freeFlow: number | null | undefined): Band {
  if (speed == null || freeFlow == null || freeFlow <= 0) return get("missing");
  const r = speed / freeFlow;
  if (r >= 0.8) return get("free");
  if (r >= 0.6) return get("slowing");
  if (r >= 0.4) return get("congested");
  if (r >= 0.25) return get("heavy");
  return get("stop");
}

/** The API's congestion definition: below 0.6 × free flow (congested, heavy or stop-and-go). */
export function isCongested(speed: number | null | undefined, freeFlow: number | null | undefined): boolean {
  if (speed == null || freeFlow == null || freeFlow <= 0) return false;
  return speed < 0.6 * freeFlow;
}

/** Ordinal rank of a band, for sorting (missing sorts last). */
export function severityRank(b: Band): number {
  return b.key === "missing" ? -1 : BANDS.indexOf(b);
}

/** Resolve a token like "var(--cong-2)" to its current colour (Leaflet paths need a literal value). */
export function resolveColor(token: string): string {
  const name = /var\((--[\w-]+)\)/.exec(token)?.[1];
  if (!name || typeof document === "undefined") return token;
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || token;
}

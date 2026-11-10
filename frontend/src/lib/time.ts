// Replay times are local Los Angeles wall-clock times with no zone ("2012-06-12T16:30"), exactly as
// the API takes them. They are handled as UTC instants internally so the viewer's own time zone and
// daylight saving can never shift them.

export const STEP_MIN = 5;
const STEP_MS = STEP_MIN * 60_000;

export function toMs(t: string): number {
  return Date.parse(`${t.slice(0, 16)}:00Z`);
}

export function fromMs(ms: number): string {
  return new Date(ms).toISOString().slice(0, 16);
}

export function addMinutes(t: string, minutes: number): string {
  return fromMs(toMs(t) + minutes * 60_000);
}

/** Snap to the 5-minute grid and keep inside [start, end]. */
export function clampToRange(t: string, start: string, end: string): string {
  const ms = Math.min(Math.max(toMs(t), toMs(start)), toMs(end));
  return fromMs(Math.round(ms / STEP_MS) * STEP_MS);
}

export function isValidTime(t: string | null | undefined): t is string {
  return t != null && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(t) && !Number.isNaN(toMs(t));
}

const fmt = (opts: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", ...opts });
const clock = fmt({ hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const dayParts = fmt({ weekday: "short", day: "numeric", month: "short", year: "numeric" });

const ms = (t: string | number) => (typeof t === "number" ? t : toMs(t));

/** "16:30" */
export const formatClock = (t: string | number) => clock.format(ms(t));
function dayText(t: string | number, withYear: boolean): string {
  const parts = Object.fromEntries(dayParts.formatToParts(ms(t)).map((p) => [p.type, p.value]));
  return [parts.weekday, parts.day, parts.month, withYear ? parts.year : ""].filter(Boolean).join(" ");
}

/** "Tue 12 Jun 2012, 16:30" */
export const formatLong = (t: string) => `${dayText(t, true)}, ${formatClock(t)}`;
/** "Tue 12 Jun" */
export const formatDay = (t: string | number) => dayText(t, false);

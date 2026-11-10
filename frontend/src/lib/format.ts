const n0 = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const n1 = new Intl.NumberFormat("en-GB", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const n2 = new Intl.NumberFormat("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/** A missing value: an en dash, shown at label size and in grey by the caller. */
export const MISSING = "–";

/** Speeds are shown to the whole mph: the sensors report about that precision. */
export const speed = (v: number | null | undefined): string => (v == null ? MISSING : n0.format(v));
export const int = (v: number): string => n0.format(v);
export const dec1 = (v: number): string => n1.format(v);
export const dec2 = (v: number): string => n2.format(v);
export const pct1 = (v: number | null | undefined): string => (v == null ? MISSING : `${n1.format(v)}%`);

/** A signed figure with a true minus sign: "+0.12", "−0.08", "0.00". */
export function signed(v: number, digits = 2): string {
  const s = new Intl.NumberFormat("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(
    Math.abs(v),
  );
  if (Number(s) === 0) return s;
  return `${v > 0 ? "+" : "−"}${s}`;
}

/** A change in speed in words: "steady", "12 mph slower", "3 mph faster". */
export function speedChange(from: number | null | undefined, to: number | null | undefined): string {
  if (from == null || to == null) return "no reading to compare";
  const d = Math.round(to) - Math.round(from);
  if (Math.abs(d) < 2) return "steady";
  return `${Math.abs(d)} mph ${d < 0 ? "slower" : "faster"}`;
}

export const plural = (n: number, one: string, many = `${one}s`): string => `${int(n)} ${n === 1 ? one : many}`;

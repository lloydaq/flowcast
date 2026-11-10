// A small SVG line chart drawn to architect.md §10.7: straight 2px lines, no point markers, 3-5 faint
// horizontal gridlines, no y-axis line, round ticks from 0, missing readings drawn as breaks with a
// "No data" band (never interpolated), a dashed forecast that unrolls from the "Now" line.
import { useMemo, useState } from "react";
import { useWidth } from "../useTheme";

export type Point = readonly [x: number, y: number | null];

export interface Series {
  key: string;
  label: string;
  color: string; // CSS colour or var(--token)
  points: readonly Point[];
  dashed?: boolean;
  width?: number;
  /** Changes when new data arrives; the series then unrolls from its first point. */
  unrollKey?: string;
  /** Label the last value at the line's end. */
  endLabel?: (y: number) => string;
  /** Draw a "No data" band where this series has no readings. */
  gapBands?: boolean;
  /** Leave out of the hover readout (e.g. a connector point duplicated from another series). */
  quiet?: boolean;
}

export interface LineChartProps {
  series: readonly Series[];
  height?: number;
  xDomain: readonly [number, number];
  xTicks: readonly number[];
  formatX: (x: number) => string;
  formatY?: (y: number) => string;
  yMax?: number; // minimum top of the y scale
  now?: { x: number; label: string };
  reference?: { y: number; label: string };
  marker?: { x: number; seriesKey: string };
  summary: string; // the chart in one sentence, for screen readers
}

const M = { l: 36, r: 64, t: 20, b: 26 };

/** Round, even ticks from 0 with the top tick above the maximum (3-5 lines). */
export function niceTicks(max: number): number[] {
  const target = Math.max(max, 1);
  for (const step of [1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500]) {
    const n = Math.ceil((target * 1.0001) / step);
    if (n <= 4 && step * n > target) return Array.from({ length: n + 1 }, (_, i) => i * step);
  }
  const step = 10 ** Math.ceil(Math.log10(target / 4));
  return Array.from({ length: Math.ceil(target / step) + 1 }, (_, i) => i * step);
}

function pathOf(points: readonly Point[], sx: (x: number) => number, sy: (y: number) => number): string {
  let d = "";
  let pen = false;
  for (const [x, y] of points) {
    if (y == null) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
    pen = true;
  }
  return d;
}

function gapRuns(points: readonly Point[]): [number, number][] {
  const runs: [number, number][] = [];
  let lastGood: number | null = null;
  let gapStart: number | null = null;
  for (const [x, y] of points) {
    if (y == null) {
      if (gapStart == null) gapStart = lastGood ?? x;
    } else {
      if (gapStart != null) runs.push([gapStart, x]);
      gapStart = null;
      lastGood = x;
    }
  }
  const last = points.at(-1);
  if (gapStart != null && last) runs.push([gapStart, last[0]]);
  return runs;
}

export function LineChart(props: LineChartProps) {
  const { series, height = 220, xDomain, xTicks, formatX, formatY = (y) => String(Math.round(y)) } = props;
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hoverX, setHoverX] = useState<number | null>(null);

  const geo = useMemo(() => {
    const ys = series.flatMap((s) => s.points.map((p) => p[1]).filter((y): y is number => y != null));
    const ticks = niceTicks(Math.max(props.yMax ?? 0, props.reference?.y ?? 0, ...ys));
    const top = ticks.at(-1) ?? 1;
    const w = Math.max(width - M.l - M.r, 10);
    const h = height - M.t - M.b;
    const [x0, x1] = xDomain;
    const sx = (x: number) => M.l + ((x - x0) / (x1 - x0 || 1)) * w;
    const sy = (y: number) => M.t + h - (y / top) * h;
    return { ticks, sx, sy, w, h };
  }, [series, width, height, xDomain, props.yMax, props.reference]);

  const { ticks, sx, sy, h } = geo;
  // keep every k-th x label so labels never collide (about 56px each); the step stays even
  const every = Math.max(1, Math.ceil((xTicks.length * 56) / Math.max(geo.w, 1)));
  const shownTicks = xTicks.filter((_, i) => i % every === 0);
  const allX = useMemo(
    () => [...new Set(series.filter((s) => !s.quiet).flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b),
    [series],
  );

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - box.left + M.l;
    let best: number | null = null;
    for (const x of allX) if (best == null || Math.abs(sx(x) - px) < Math.abs(sx(best) - px)) best = x;
    setHoverX(best);
  };
  const hoverRows =
    hoverX == null
      ? []
      : series
          .filter((s) => !s.quiet)
          .map((s) => ({ s, y: s.points.find((p) => p[0] === hoverX)?.[1] }))
          .filter((r) => r.y !== undefined);

  return (
    <div className="chart" ref={ref}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={props.summary}>
          {ticks.map((t, i) => (
            <g key={t}>
              <line x1={M.l} x2={width - M.r} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" strokeWidth={i === 0 ? 1.5 : 1} />
              <text x={M.l - 8} y={sy(t) + 4} textAnchor="end">
                {formatY(t)}
              </text>
            </g>
          ))}
          {shownTicks.map((x, i) => (
            <text
              key={x}
              x={sx(x)}
              y={height - 6}
              textAnchor={i === 0 ? "start" : i === shownTicks.length - 1 && sx(x) > width - M.r - 20 ? "end" : "middle"}
            >
              {formatX(x)}
            </text>
          ))}
          {series
            .filter((s) => s.gapBands)
            .flatMap((s) => gapRuns(s.points))
            .map(([a, b]) => (
              <g key={`gap-${a}`}>
                <rect x={sx(a)} y={M.t} width={Math.max(sx(b) - sx(a), 2)} height={h} fill="var(--raised)" />
                {sx(b) - sx(a) > 54 && (
                  <text x={(sx(a) + sx(b)) / 2} y={M.t + 14} textAnchor="middle">
                    No data
                  </text>
                )}
              </g>
            ))}
          {props.reference && (
            <g>
              <line
                x1={M.l}
                x2={width - M.r}
                y1={sy(props.reference.y)}
                y2={sy(props.reference.y)}
                stroke="var(--faint)"
                strokeDasharray="3 3"
              />
              <text x={width - M.r + 6} y={sy(props.reference.y) + 4}>
                {props.reference.label}
              </text>
            </g>
          )}
          {props.now && (
            <g>
              <line
                x1={sx(props.now.x)}
                x2={sx(props.now.x)}
                y1={M.t - 6}
                y2={M.t + h}
                stroke="var(--muted)"
                strokeDasharray="3 3"
              />
              <text x={sx(props.now.x) - 6} y={M.t - 8} textAnchor="end" className="label-ink">
                {props.now.label}
              </text>
            </g>
          )}
          {series.map((s) => {
            const d = pathOf(s.points, sx, sy);
            const last = [...s.points].reverse().find((p) => p[1] != null);
            return (
              <g key={`${s.key}-${s.unrollKey ?? ""}`} className={s.unrollKey ? "unroll" : undefined}>
                <path
                  d={d}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={s.width ?? 2}
                  strokeDasharray={s.dashed ? "6 4" : undefined}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                {s.endLabel && last && last[1] != null && (
                  <text x={sx(last[0]) + 8} y={sy(last[1]) + 4} className="label-ink">
                    {s.endLabel(last[1])}
                  </text>
                )}
              </g>
            );
          })}
          {props.marker &&
            (() => {
              const s = series.find((v) => v.key === props.marker?.seriesKey);
              const y = s?.points.find((p) => p[0] === props.marker?.x)?.[1];
              return y == null || !s ? null : (
                <circle cx={sx(props.marker.x)} cy={sy(y)} r={5} fill="var(--surface)" stroke={s.color} strokeWidth={2} />
              );
            })()}
          {hoverX != null && (
            <g pointerEvents="none">
              <line x1={sx(hoverX)} x2={sx(hoverX)} y1={M.t} y2={M.t + h} stroke="var(--control)" />
              {hoverRows.map(({ s, y }) =>
                y == null ? null : <circle key={s.key} cx={sx(hoverX)} cy={sy(y)} r={4} fill={s.color} />,
              )}
            </g>
          )}
          <rect
            x={M.l}
            y={M.t}
            width={Math.max(width - M.l - M.r, 0)}
            height={h}
            fill="transparent"
            onPointerMove={onMove}
            onPointerLeave={() => setHoverX(null)}
          />
        </svg>
      )}
      {hoverX != null && hoverRows.length > 0 && (
        <div
          className="hover-tip"
          style={{
            left: Math.min(sx(hoverX) + 12, Math.max(width - 190, 0)),
            top: M.t,
          }}
        >
          <div className="muted">{formatX(hoverX)}</div>
          {hoverRows.map(({ s, y }) => (
            <div key={s.key}>
              {s.label}: <span className="num">{y == null ? "no reading" : formatY(y)}</span>
            </div>
          ))}
        </div>
      )}
      {width === 0 && <div style={{ height }} />}
    </div>
  );
}

/** "View as table" for a chart (§10.10): the same numbers, readable without the picture. */
export function ChartTable({
  caption,
  columns,
  rows,
}: {
  caption: string;
  columns: readonly string[];
  rows: readonly (readonly string[])[];
}) {
  return (
    <details className="as-table">
      <summary>View as table</summary>
      <div className="table-scroll">
        <table className="data">
          <caption className="visually-hidden">{caption}</caption>
          <thead>
            <tr>
              {columns.map((c, i) => (
                <th key={c} className={i > 0 ? "num" : undefined} scope="col">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r[0]}>
                {r.map((cell, i) =>
                  i === 0 ? (
                    <th key={i} scope="row">
                      {cell}
                    </th>
                  ) : (
                    <td key={i} className="num">
                      {cell}
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

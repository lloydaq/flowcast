// The replay clock: the as-of time (t), the horizon (h) and the model (m) live in the URL, so every
// view is shareable and back/forward works. Play advances the clock one 5-minute step per second.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router";
import type { Health } from "./api";
import { useHealth } from "./queries";
import { addMinutes, clampToRange, isValidTime, toMs } from "./time";

/** The evening build-up on Tue 12 Jun 2012: 63 sensors congested, 82 forecast congested within 30 min. */
export const DEFAULT_TIME = "2012-06-12T16:30";
export const HORIZONS = [0, 15, 30, 45, 60] as const; // 0 = now
export type Horizon = (typeof HORIZONS)[number];
const DEFAULT_HORIZON: Horizon = 30;

export interface Replay {
  health: Health | undefined;
  t: string | undefined; // undefined until the replay range is known
  h: Horizon;
  model: string | undefined;
  start: string | undefined;
  end: string | undefined;
  playing: boolean;
  setTime: (t: string, replace?: boolean) => void;
  step: (steps: number) => void;
  setHorizon: (h: Horizon) => void;
  setModel: (m: string) => void;
  togglePlay: () => void;
  /** A link that keeps the replay state: href("/models") → "/models?t=…&h=…" */
  href: (path: string, extra?: Record<string, string>) => string;
}

const Ctx = createContext<Replay | null>(null);

export function ReplayProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams();
  const health = useHealth().data;
  const [playing, setPlaying] = useState(false);

  const start = health?.replay_start.slice(0, 16);
  const end = health?.replay_end.slice(0, 16);
  const rawT = params.get("t");
  const t = start && end ? clampToRange(isValidTime(rawT) ? rawT : DEFAULT_TIME, start, end) : undefined;
  const hParam = Number(params.get("h"));
  const h: Horizon = (HORIZONS as readonly number[]).includes(hParam) && params.has("h") ? (hParam as Horizon) : DEFAULT_HORIZON;
  const mParam = params.get("m");
  const model = health ? (mParam && health.models.includes(mParam) ? mParam : health.default_model) : undefined;

  const update = useCallback(
    (changes: Record<string, string>, replace = false) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const [k, v] of Object.entries(changes)) next.set(k, v);
          return next;
        },
        { replace },
      );
    },
    [setParams],
  );

  const setTime = useCallback(
    (next: string, replace = false) => {
      if (start && end) update({ t: clampToRange(next, start, end) }, replace);
    },
    [start, end, update],
  );
  const step = useCallback((n: number) => t && setTime(addMinutes(t, n * 5)), [t, setTime]);

  // play: one step per second, stopping at the end of the replay range
  useEffect(() => {
    if (!playing || !t || !end) return;
    if (toMs(t) >= toMs(end)) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setTime(addMinutes(t, 5), true), 1000);
    return () => window.clearTimeout(id);
  }, [playing, t, end, setTime]);

  const value = useMemo<Replay>(
    () => ({
      health,
      t,
      h,
      model,
      start,
      end,
      playing,
      setTime,
      step,
      setHorizon: (next) => update({ h: String(next) }, true),
      setModel: (m) => update({ m }, true),
      togglePlay: () => setPlaying((p) => !p),
      href: (path, extra = {}) => {
        const q = new URLSearchParams();
        for (const k of ["t", "h", "m"]) {
          const v = params.get(k);
          if (v) q.set(k, v);
        }
        for (const [k, v] of Object.entries(extra)) q.set(k, v);
        const s = q.toString();
        return s ? `${path}?${s}` : path;
      },
    }),
    [health, t, h, model, start, end, playing, setTime, step, update, params],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useReplay(): Replay {
  const v = useContext(Ctx);
  if (!v) throw new Error("useReplay outside ReplayProvider");
  return v;
}

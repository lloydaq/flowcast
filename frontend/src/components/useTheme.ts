import { useEffect, useState } from "react";
import type { Theme } from "../lib/theme";

/** The current data-theme on <html>, kept in sync (charts and the map re-read their colours from it). */
export function useTheme(): Theme {
  const read = (): Theme => (document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  const [theme, setTheme] = useState<Theme>(read);
  useEffect(() => {
    const mo = new MutationObserver(() => setTheme(read()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => mo.disconnect();
  }, []);
  return theme;
}

/** Width of an element, tracked with a ResizeObserver (charts are drawn to their real width). */
export function useWidth<T extends HTMLElement>(): [React.RefCallback<T>, number] {
  const [el, setEl] = useState<T | null>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!el) return;
    const ro = new ResizeObserver(([e]) => e && setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, w];
}

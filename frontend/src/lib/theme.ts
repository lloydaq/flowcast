// Light by default (the dealt style is a light framed sheet). The viewer's choice is remembered on this
// device, and ?theme=dark|light in the URL sets it (handy for sharing and for checking both themes).
export type Theme = "light" | "dark";
const KEY = "flowcast-theme";

export function storedTheme(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

export function initialTheme(): Theme {
  const fromUrl = new URLSearchParams(location.search).get("theme");
  if (fromUrl === "dark" || fromUrl === "light") {
    applyTheme(fromUrl, true);
    return fromUrl;
  }
  return storedTheme() ?? "light";
}

export function applyTheme(theme: Theme, remember = false): void {
  document.documentElement.dataset.theme = theme;
  if (!remember) return;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* private mode: the choice lasts this visit only */
  }
}

export function prefersReducedMotion(): boolean {
  return typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

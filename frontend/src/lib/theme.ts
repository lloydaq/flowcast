// Theme: dark, light or system. First visit follows the system; a manual choice is remembered on this
// device. index.html sets data-theme before the first paint (same key and rule as here), so the page
// never flashes the wrong theme. ?theme=dark|light in the URL forces a theme for that visit.
export type Theme = "light" | "dark";
export type ThemeChoice = Theme | "system";
const KEY = "flowcast-theme";
const dark = () => matchMedia("(prefers-color-scheme: dark)");

export function themeChoice(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

const resolve = (c: ThemeChoice): Theme => (c === "system" ? (dark().matches ? "dark" : "light") : c);

function paint(theme: Theme, animate: boolean): void {
  const root = document.documentElement;
  if (root.dataset.theme === theme) return;
  if (animate && !prefersReducedMotion()) {
    // colours cross-fade for one beat, then the class goes so normal transitions are unaffected
    root.classList.add("theme-switching");
    window.setTimeout(() => root.classList.remove("theme-switching"), 360);
  }
  root.dataset.theme = theme;
}

export function setThemeChoice(choice: ThemeChoice): void {
  try {
    if (choice === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, choice);
  } catch {
    /* private mode: the choice lasts this visit only */
  }
  paint(resolve(choice), true);
  window.dispatchEvent(new Event("flowcast-theme"));
}

/** Called once at startup: apply the theme and follow the system while the choice is "system". */
export function initTheme(): void {
  const forced = new URLSearchParams(location.search).get("theme");
  paint(forced === "dark" || forced === "light" ? forced : resolve(themeChoice()), false);
  dark().addEventListener("change", () => {
    if (themeChoice() === "system") paint(resolve("system"), true);
  });
}

export function prefersReducedMotion(): boolean {
  return typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

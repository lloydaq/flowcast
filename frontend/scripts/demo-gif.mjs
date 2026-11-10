// Record the README's demo GIFs from the running app: the replay playing through the evening build-up
// on Tue 12 Jun 2012, in both themes. Frames are real screenshots, one per 5-minute step.
//
// Usage (the app on :4173 and the API on :8000, or the static demo on :4174):
//   node scripts/demo-gif.mjs [baseURL]
// Writes docs/assets/flowcast-demo-{dark,light}.gif (frames assembled by scripts/assemble_gif.py).
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const BASE = process.argv[2] ?? "http://localhost:4173/";
const FRAMES = join(ROOT, "frontend", ".gif-frames");
const STEPS = 24; // 16:00 to 18:00
const python = join(ROOT, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");

const browser = await chromium.launch({ channel: "chrome" });
for (const theme of ["dark", "light"]) {
  const dir = join(FRAMES, theme);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 760 }, reducedMotion: "reduce" });
  await page.goto(new URL(`?t=2012-06-12T16:00&h=30&theme=${theme}`, BASE).href);
  await page.locator(".map path.fc-marker").nth(206).waitFor();
  await page.waitForTimeout(2500); // map tiles
  for (let i = 0; i <= STEPS; i++) {
    await page.screenshot({ path: join(dir, `${String(i).padStart(3, "0")}.png`) });
    if (i === STEPS) break;
    const before = await page.locator(".finding").textContent();
    await page.locator(".topbar").getByRole("button", { name: "Forward 5 minutes" }).click();
    await page.waitForFunction((prev) => document.querySelector(".finding")?.textContent !== prev, before);
    await page.waitForTimeout(150);
  }
  await page.close();
  execFileSync(python, [join(ROOT, "scripts", "assemble_gif.py"), dir, join(ROOT, "docs", "assets", `flowcast-demo-${theme}.gif`)], {
    stdio: "inherit",
  });
}
await browser.close();
rmSync(FRAMES, { recursive: true, force: true });

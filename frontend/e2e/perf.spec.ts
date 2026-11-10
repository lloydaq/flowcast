// Performance of the 207-marker map, measured in headless Chrome on this machine against the real API.
// Writes models/metrics/frontend_perf.json. Numbers are indicative: headless Chrome, one run, one laptop.
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "models", "metrics", "frontend_perf.json");
const AT = "t=2012-06-12T16:30&h=30";
const pct = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return Math.round((s[Math.min(s.length - 1, Math.floor((q / 100) * s.length))] ?? 0) * 10) / 10;
};

/** ms from pressing "Forward" until the finding shows the new time and the map has repainted. */
async function stepTimes(page: Page, n: number): Promise<number[]> {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const before = (await page.locator(".finding").textContent()) ?? "";
    const ms = await page.evaluate(async (prev) => {
      const t0 = performance.now();
      (document.querySelector('.topbar button[aria-label="Forward 5 minutes"]') as HTMLButtonElement).click();
      while ((document.querySelector(".finding")?.textContent ?? "") === prev) await new Promise((r) => setTimeout(r, 2));
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))); // the repaint after the update
      return performance.now() - t0;
    }, before);
    out.push(ms);
  }
  return out;
}

/** Frame intervals (ms) while the replay plays for `seconds`. */
async function playFrames(page: Page, seconds: number): Promise<number[]> {
  await page.locator(".topbar").getByRole("button", { name: /^Play replay/ }).click();
  const frames = await page.evaluate(async (s) => {
    const gaps: number[] = [];
    let last = performance.now();
    const end = last + s * 1000;
    await new Promise<void>((done) => {
      const tick = (now: number) => {
        gaps.push(now - last);
        last = now;
        if (now < end) requestAnimationFrame(tick);
        else done();
      };
      requestAnimationFrame(tick);
    });
    return gaps;
  }, seconds);
  await page.locator(".topbar").getByRole("button", { name: "Pause replay" }).click();
  return frames;
}

const frameSummary = (f: number[]) => ({
  frames: f.length,
  p50_ms: pct(f, 50),
  p95_ms: pct(f, 95),
  longest_ms: Math.round(Math.max(...f)),
  frames_over_50ms: f.filter((x) => x > 50).length,
});

test("207-marker map performance", async ({ page }) => {
  test.setTimeout(180_000);
  const loads: number[] = [];
  for (let i = 0; i < 3; i++) {
    await page.goto("about:blank");
    const t0 = Date.now();
    await page.goto(`/?${AT}&h=${30}`);
    await expect(page.locator(".map path.fc-marker")).toHaveCount(207);
    await expect(page.locator(".finding")).toContainText("forecast congested");
    loads.push(Date.now() - t0);
  }
  await page.waitForTimeout(1500);
  const normalSteps = await stepTimes(page, 12);
  const normalFrames = await playFrames(page, 6);

  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  const slowSteps = await stepTimes(page, 12);
  const slowFrames = await playFrames(page, 6);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 1 });

  const result = {
    setup:
      "Production build, headless Chrome (Playwright), 1440x900, real API on localhost; development laptop (Ryzen 7 7840HS). " +
      "Indicative only: one run, headless rendering, and map tiles from the network.",
    markers: 207,
    load_to_markers_ms: { runs: loads, median: pct(loads, 50) },
    replay_step_ms: { cpu: "normal", runs: normalSteps.length, p50: pct(normalSteps, 50), p95: pct(normalSteps, 95) },
    playing_frames: { cpu: "normal", ...frameSummary(normalFrames) },
    replay_step_ms_cpu_4x_slower: { runs: slowSteps.length, p50: pct(slowSteps, 50), p95: pct(slowSteps, 95) },
    playing_frames_cpu_4x_slower: frameSummary(slowFrames),
  };
  writeFileSync(OUT, `${JSON.stringify(result, null, 1)}\n`, "utf-8");
  console.log(JSON.stringify(result, null, 1));
  // the replay plays at one step per second: each step must finish well inside that, even on a slow CPU
  expect(result.replay_step_ms_cpu_4x_slower.p95).toBeLessThan(500);
});

// End-to-end tests against the real backend: the FastAPI app on :8000 and the production build on :4173.
// Both are started if they are not already running (they need data/processed/ and the saved predictions).
//   npm run e2e            all flows (Chrome, desktop + phone)
//   npm run e2e:static     the same flows against the GitHub Pages build (no API)
//   npm run e2e:perf       the map performance check; writes models/metrics/frontend_perf.json
import { defineConfig, devices } from "@playwright/test";

const python = process.platform === "win32" ? ".venv/Scripts/python" : ".venv/bin/python";

export default defineConfig({
  testDir: "e2e",
  fullyParallel: false, // one API process; keep the numbers in the perf check quiet
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: "http://localhost:4173/",
    channel: "chrome", // the installed Chrome: no separate browser download
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], channel: "chrome", viewport: { width: 1440, height: 900 } }, testIgnore: /perf|phone/ },
    { name: "phone", use: { ...devices["Pixel 7"], channel: "chrome" }, testMatch: /phone/ },
    // the GitHub Pages build: same flows, no API, served under /flowcast/
    { name: "static", use: { ...devices["Desktop Chrome"], channel: "chrome", viewport: { width: 1440, height: 900 }, baseURL: "http://localhost:4174/flowcast/" }, testMatch: /flows/ },
    { name: "static-phone", use: { ...devices["Pixel 7"], channel: "chrome", baseURL: "http://localhost:4174/flowcast/" }, testMatch: /phone/ },
    { name: "perf", use: { ...devices["Desktop Chrome"], channel: "chrome", viewport: { width: 1440, height: 900 } }, testMatch: /perf/ },
  ],
  webServer: [
    {
      command: `${python} -m uvicorn flowcast.api.app:app --port 8000 --log-level warning`,
      cwd: "..",
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: true,
      timeout: 60_000,
    },
    {
      command: "npm run build && npx vite preview --port 4173 --strictPort",
      url: "http://localhost:4173",
      reuseExistingServer: true,
      timeout: 120_000,
    },
    {
      // needs `python scripts/export_static.py` first (frontend/.static)
      command: "npm run build:static && npm run preview:static",
      url: "http://localhost:4174/flowcast/",
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
});

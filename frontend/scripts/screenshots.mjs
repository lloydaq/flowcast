// Capture the README screenshots from the running app, in both themes, and check each page for
// console errors and horizontal overflow (desktop and phone width).
//
// Usage (API on :8000 and `npm run preview` on :4173 must be running):
//   node scripts/screenshots.mjs
// Needs Node 22+ and Chrome (set CHROME_PATH if it is not in the default Windows location).
// Writes docs/assets/screens/<name>-<theme>.png. Nothing is mocked: these are the real screens.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const CHROME = process.env.CHROME_PATH ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const BASE = process.env.FLOWCAST_URL ?? "http://localhost:4173";
const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "docs", "assets", "screens");
const AT = "t=2012-06-12T16:30&h=30";
const PAGES = [
  ["network", `/?${AT}`],
  ["sensors", `/sensors?${AT}`],
  ["forecast", `/sensor/773939?${AT}`],
  ["models", `/models?${AT}`],
  ["patterns", `/patterns?${AT}`],
];
const PORT = 9334;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

mkdirSync(OUT, { recursive: true });
const chrome = spawn(CHROME, [
  "--headless=new",
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${mkdtempSync(join(tmpdir(), "fc-shots-"))}`,
  "--hide-scrollbars",
  "about:blank",
]);

let ws;
for (let i = 0; i < 60 && !ws; i++) {
  try {
    const page = (await (await fetch(`http://127.0.0.1:${PORT}/json`)).json()).find((t) => t.type === "page");
    if (page) ws = new WebSocket(page.webSocketDebuggerUrl);
  } catch {
    await sleep(200);
  }
}
if (!ws) throw new Error("Chrome did not start; set CHROME_PATH");
await new Promise((r) => ws.addEventListener("open", r));

let id = 0;
const pending = new Map();
const problems = [];
let current = "";
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) pending.get(m.id)(m);
  if (m.method === "Runtime.exceptionThrown") problems.push(`${current}: exception ${m.params.exceptionDetails.text}`);
  if (m.method === "Log.entryAdded" && m.params.entry.level === "error")
    problems.push(`${current}: ${m.params.entry.text} ${m.params.entry.url ?? ""}`);
});
const send = (method, params = {}) =>
  new Promise((resolve) => {
    pending.set(++id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true })).result.result?.value;
const size = (width, height) => send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width < 600 });

await send("Runtime.enable");
await send("Log.enable");
await send("Page.enable");

for (const theme of ["dark", "light"]) {
  for (const [name, path] of PAGES) {
    current = `${name}-${theme}`;
    await size(1440, 900);
    await send("Page.navigate", { url: `${BASE}${path}&theme=${theme}` });
    await sleep(name === "network" || name === "forecast" ? 4500 : 2500); // map tiles and entrance animations
    if ((await evaluate("document.querySelector('h1')?.textContent")) == null) problems.push(`${current}: no h1 rendered`);
    const shot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(join(OUT, `${current}.png`), Buffer.from(shot.result.data, "base64"));
    for (const width of [1440, 768, 390]) {
      await size(width, 900);
      await sleep(350);
      const over = await evaluate("document.documentElement.scrollWidth - window.innerWidth");
      if (over > 0) problems.push(`${current}: ${over}px horizontal overflow at ${width}px`);
    }
    console.log("captured", current);
  }
}
ws.close();
chrome.kill();
console.log(problems.length ? `PROBLEMS:\n  ${problems.join("\n  ")}` : "No console errors and no horizontal overflow on any page.");
process.exit(problems.length ? 1 : 0);

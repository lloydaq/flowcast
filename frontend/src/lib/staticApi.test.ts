// The static demo must answer exactly as the real API does. scripts/export_static.py records real API
// answers for sampled requests (.static-samples.json); this rebuilds each from the static files.
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { createStaticApi } from "./staticApi";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const SAMPLES = join(ROOT, ".static-samples.json");
const DATA = join(ROOT, ".static", "data");
const ready = existsSync(SAMPLES) && existsSync(DATA);

const fileFetch = async (url: string) => {
  const path = join(DATA, decodeURIComponent(url.replace(/^data\//, "")));
  if (!existsSync(path)) return new Response("", { status: 404 });
  return new Response(readFileSync(path));
};

describe.skipIf(!ready)("static demo data (run scripts/export_static.py first)", () => {
  const samples = ready ? (JSON.parse(readFileSync(SAMPLES, "utf-8")) as { url: string; response: unknown }[]) : [];
  const get = createStaticApi("data/", fileFetch);

  it.each(samples.map((s) => [s.url, s.response] as const))("%s matches the real API", async (url, response) => {
    const u = new URL(url, "http://x");
    expect(await get(u.pathname, Object.fromEntries(u.searchParams))).toEqual(response);
  });

  it("answers the small endpoints from the exported API JSON", async () => {
    expect(await get("/api/health")).toMatchObject({ sensors: 207, default_model: "dcrnn" });
    expect(((await get("/api/metrics/sensors", { model: "AR on residuals", horizon: 15, split: "test" })) as unknown[]).length).toBe(207);
  });

  it("rejects bad input with the API's messages", async () => {
    await expect(get("/api/sensors/999999")).rejects.toThrow("unknown sensor '999999'");
    await expect(get("/api/forecast", { as_of: "2012-06-12T16:32", horizon: 30 })).rejects.toThrow("5-minute grid");
    await expect(get("/api/forecast", { as_of: "2012-01-01T00:00", horizon: 30 })).rejects.toThrow("between 2012-06-04 04:10:00");
    await expect(get("/api/forecast", { as_of: "2012-06-12T16:30", horizon: 20 })).rejects.toThrow("horizon must be one of [15, 30, 45, 60]");
    await expect(get("/api/forecast", { as_of: "2012-06-12T16:30", horizon: 30, model: "gpt" })).rejects.toThrow(
      "unknown model 'gpt'; expected one of ['dcrnn', 'lightgbm']",
    );
  });
});

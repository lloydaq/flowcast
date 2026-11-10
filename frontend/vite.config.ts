import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

// The API runs on :8000 (uvicorn flowcast.api.app:app). The dev server and the preview of the
// production build both proxy /api to it, so the browser talks to one origin and needs no CORS.
const api = { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } };

/**
 * The GitHub Pages build (`vite build --mode static`). Pages cannot send headers, so the
 * Content-Security-Policy goes in a meta tag, with the hash of the one inline script (the theme
 * set-up that runs before first paint). Each route gets its own index.html so deep links load with HTTP 200.
 */
function pagesBuild(): Plugin {
  let outDir = "dist";
  return {
    name: "flowcast-pages",
    apply: "build",
    configResolved(c) {
      outDir = c.build.outDir;
    },
    closeBundle() {
      // hash the inline script exactly as it ships, after every other transform of index.html
      const file = join(outDir, "index.html");
      const html = readFileSync(file, "utf-8");
      const inline = /<script>([\s\S]*?)<\/script>/.exec(html)?.[1];
      if (inline === undefined) throw new Error("index.html: the inline theme script is missing");
      // browsers hash the script after turning CRLF into LF, as every HTML parser does
      const hash = createHash("sha256").update(inline.replace(/\r\n?/g, "\n")).digest("base64");
      const csp = [
        "default-src 'self'",
        `script-src 'self' 'sha256-${hash}'`,
        "style-src 'self' 'unsafe-inline'", // React style attributes and Leaflet's positioning
        "img-src 'self' data: https://tile.openstreetmap.org",
        "font-src 'self'",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
      ].join("; ");
      writeFileSync(file, html.replace("<head>", `<head>\n    <meta http-equiv="Content-Security-Policy" content="${csp}" />`));
      copyFileSync(file, join(outDir, "404.html")); // unknown paths: the app's own "Page not found"
      // every real route gets its own copy, so deep links load with HTTP 200, not GitHub's 404 fallback
      const grid = join(outDir, "data", "grid.json");
      const sensors = existsSync(grid) ? (JSON.parse(readFileSync(grid, "utf-8")).sensor_ids as string[]) : [];
      for (const route of ["sensors", "models", "patterns", "settings", ...sensors.map((id) => `sensor/${id}`)]) {
        mkdirSync(join(outDir, route), { recursive: true });
        copyFileSync(file, join(outDir, route, "index.html"));
      }
      writeFileSync(join(outDir, ".nojekyll"), "");
    },
  };
}

export default defineConfig(({ mode }) => ({
  plugins: [react(), ...(mode === "static" ? [pagesBuild()] : [])],
  // static: served from https://<user>.github.io/flowcast/, with the exported data as the public folder
  ...(mode === "static" ? { base: "/flowcast/", publicDir: ".static" } : {}),
  server: { port: 5173, proxy: api },
  preview: { port: 4173, proxy: api },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
}));

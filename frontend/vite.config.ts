import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The API runs on :8000 (uvicorn flowcast.api.app:app). The dev server and the preview of the
// production build both proxy /api to it, so the browser talks to one origin and needs no CORS.
const api = { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } };

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: api },
  preview: { port: 4173, proxy: api },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});

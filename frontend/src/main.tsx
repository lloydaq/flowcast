import "@fontsource-variable/inter";
import "@fontsource/ibm-plex-mono/400.css";
import "./styles/tokens.css";
import "./styles/app.css";
import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import { Shell } from "./components/Shell";
import { queryClient } from "./lib/queries";
import { ReplayProvider } from "./lib/replay";
import { initTheme } from "./lib/theme";
import { NetworkScreen } from "./screens/Network";
import { NotFound } from "./screens/NotFound";

initTheme();

// The Network screen ships in the first bundle; the other screens load when first opened.
const router = createBrowserRouter([
  {
    element: (
      <ReplayProvider>
        <Shell />
      </ReplayProvider>
    ),
    children: [
      { path: "/", element: <NetworkScreen /> },
      { path: "/sensors", lazy: () => import("./screens/Sensors").then((m) => ({ Component: m.SensorsScreen })) },
      { path: "/sensor/:id", lazy: () => import("./screens/Sensor").then((m) => ({ Component: m.SensorScreen })) },
      { path: "/models", lazy: () => import("./screens/Models").then((m) => ({ Component: m.ModelsScreen })) },
      { path: "/patterns", lazy: () => import("./screens/Patterns").then((m) => ({ Component: m.PatternsScreen })) },
      { path: "/settings", lazy: () => import("./screens/Settings").then((m) => ({ Component: m.SettingsScreen })) },
      { path: "*", element: <NotFound /> },
    ],
  },
], { basename: import.meta.env.BASE_URL.replace(/\/$/, "") || "/" }); // /flowcast on GitHub Pages

const root = document.getElementById("root");
if (!root) throw new Error("#root missing from index.html");
createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);

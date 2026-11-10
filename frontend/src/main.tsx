import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
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
import { applyTheme, initialTheme } from "./lib/theme";
import { ModelsScreen } from "./screens/Models";
import { NetworkScreen } from "./screens/Network";
import { NotFound } from "./screens/NotFound";
import { PatternsScreen } from "./screens/Patterns";
import { SensorScreen } from "./screens/Sensor";

applyTheme(initialTheme());

const router = createBrowserRouter([
  {
    element: (
      <ReplayProvider>
        <Shell />
      </ReplayProvider>
    ),
    children: [
      { path: "/", element: <NetworkScreen /> },
      { path: "/sensor/:id", element: <SensorScreen /> },
      { path: "/models", element: <ModelsScreen /> },
      { path: "/patterns", element: <PatternsScreen /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);

const root = document.getElementById("root");
if (!root) throw new Error("#root missing from index.html");
createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);

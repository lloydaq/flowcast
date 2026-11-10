// Settings: appearance and the default model. Each control saves itself.
import { useEffect, useState } from "react";
import { Segmented } from "../components/Segmented";
import { useModels } from "../lib/queries";
import { useReplay } from "../lib/replay";
import { setThemeChoice, themeChoice, type ThemeChoice } from "../lib/theme";

const THEMES = [
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
  { value: "system", label: "System" },
] as const;

export function SettingsScreen() {
  const r = useReplay();
  const models = useModels().data;
  const [choice, setChoice] = useState<ThemeChoice>(themeChoice);
  useEffect(() => {
    const sync = () => setChoice(themeChoice());
    window.addEventListener("flowcast-theme", sync);
    return () => window.removeEventListener("flowcast-theme", sync);
  }, []);

  return (
    <>
      <div className="title-row">
        <div>
          <h1>Settings</h1>
          <p className="finding">Appearance, and the model used for forecasts.</p>
        </div>
      </div>
      <section className="card settings" aria-label="Settings">
        <div className="setting">
          <div>
            <h2>Theme</h2>
            <p className="muted">System follows your device and switches with it.</p>
          </div>
          <Segmented<ThemeChoice> label="Theme" options={THEMES} value={choice} onChange={setThemeChoice} />
        </div>
        <div className="setting">
          <div>
            <h2>Forecasting model</h2>
            <p className="muted">Used on the Network, Sensors and sensor pages. Kept in the page address, so shared links show the same model.</p>
          </div>
          {models && r.model ? (
            <Segmented label="Forecasting model" options={models.map((m) => ({ value: m.name, label: m.label }))} value={r.model} onChange={r.setModel} />
          ) : (
            <span className="muted">Loading…</span>
          )}
        </div>
      </section>
    </>
  );
}

// The app frame: sidebar and top bar share the ground colour; the work sits on an inset sheet.
import { CalendarClock, ChartColumn, ChevronLeft, ChevronRight, List, Map, Menu, Moon, Pause, Play, Search, Settings, Sun, X } from "lucide-react";
import { useEffect, useId, useState, type FormEvent } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router";
import { useSensors } from "../lib/queries";
import { useReplay } from "../lib/replay";
import { setThemeChoice } from "../lib/theme";
import { formatLong, toMs } from "../lib/time";
import { useTheme } from "./useTheme";

/** The FlowCast mark: observed speed (solid), then the forecast (dashed) leaving the "now" point. */
export function Mark({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect x="0.5" y="0.5" width="31" height="31" rx="8" fill="#14213d" stroke="var(--border)" />
      <path d="M6 19h6l4-7 3 5" fill="none" stroke="#ece7d9" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M19 17l7-6" fill="none" stroke="#76a3da" strokeWidth="2.5" strokeLinecap="round" strokeDasharray="0.1 4.2" />
      <circle cx="19" cy="17" r="2.6" fill="#14213d" stroke="#ece7d9" strokeWidth="1.8" />
    </svg>
  );
}

// Only destinations with real data behind them.
const NAV = [
  { to: "/", label: "Network", icon: Map },
  { to: "/sensors", label: "Sensors", icon: List },
  { to: "/models", label: "Models", icon: ChartColumn },
  { to: "/patterns", label: "Patterns", icon: CalendarClock },
  { to: "/settings", label: "Settings", icon: Settings },
] as const;

function activeIndex(pathname: string): number {
  if (pathname.startsWith("/sensor")) return 1; // a sensor's page belongs to Sensors
  return NAV.findIndex((n) => n.to === pathname);
}

function ReplayClock({ compact = false }: { compact?: boolean }) {
  const r = useReplay();
  const atStart = !r.t || !r.start || toMs(r.t) <= toMs(r.start);
  const atEnd = !r.t || !r.end || toMs(r.t) >= toMs(r.end);
  return (
    <div className="replay">
      <button type="button" className="btn btn-ghost btn-icon" aria-label="Back 5 minutes" disabled={atStart} onClick={() => r.step(-1)}>
        <ChevronLeft aria-hidden="true" />
      </button>
      <button
        type="button"
        className="btn btn-ghost btn-icon"
        aria-label={r.playing ? "Pause replay" : "Play replay, 5 minutes per second"}
        aria-pressed={r.playing}
        disabled={!r.t || (atEnd && !r.playing)}
        onClick={r.togglePlay}
      >
        {r.playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
      </button>
      <button type="button" className="btn btn-ghost btn-icon" aria-label="Forward 5 minutes" disabled={atEnd} onClick={() => r.step(1)}>
        <ChevronRight aria-hidden="true" />
      </button>
      <span className="replay-time" aria-live="polite">
        {r.t ? (
          <>
            {compact ? "" : "Replaying "}
            <strong>
              <time dateTime={r.t}>{formatLong(r.t)}</time>
            </strong>
          </>
        ) : (
          <span className="muted">Loading the replay range…</span>
        )}
      </span>
    </div>
  );
}

function SensorSearch() {
  const sensors = useSensors().data;
  const navigate = useNavigate();
  const r = useReplay();
  const listId = useId();
  const [value, setValue] = useState("");
  const [error, setError] = useState(false);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    const id = value.trim();
    if (sensors?.some((s) => s.sensor_id === id)) {
      setError(false);
      setValue("");
      navigate(r.href(`/sensor/${id}`));
    } else setError(true);
  };
  return (
    <form className="search" role="search" onSubmit={submit}>
      <Search aria-hidden="true" />
      <input
        aria-label="Find a sensor by its ID"
        aria-invalid={error}
        placeholder={error ? "No sensor with that ID" : "Find sensor, e.g. 717453"}
        list={listId}
        value={value}
        inputMode="numeric"
        onChange={(e) => {
          setValue(e.target.value);
          setError(false);
        }}
      />
      <datalist id={listId}>
        {sensors?.map((s) => <option key={s.sensor_id} value={s.sensor_id} />)}
      </datalist>
    </form>
  );
}

export function Shell() {
  const r = useReplay();
  const theme = useTheme();
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setMenuOpen(false), [location.pathname]);

  const active = activeIndex(location.pathname);
  const title = location.pathname.startsWith("/sensor/") ? "Sensor" : (NAV[active]?.label ?? "FlowCast");

  return (
    <div className="frame">
      <a className="skip-link" href="#content">
        Skip to content
      </a>
      <aside className={`sidebar${menuOpen ? " open" : ""}`} aria-label="Main">
        <button type="button" className="btn btn-ghost btn-icon close-menu" aria-label="Close menu" onClick={() => setMenuOpen(false)}>
          <X aria-hidden="true" />
        </button>
        <Link className="brand" to={r.href("/")}>
          <Mark />
          FlowCast
        </Link>
        <p className="scope">Traffic forecasting · Los Angeles</p>
        <nav>
          <ul className="nav">
            {/* one indicator that slides to the active item */}
            {active >= 0 && <li className="nav-indicator" aria-hidden="true" style={{ transform: `translateY(${active * 34}px)` }} />}
            {NAV.map(({ to, label, icon: Icon }, i) => (
              <li key={to}>
                <NavLink to={r.href(to)} end={to === "/"} aria-current={i === active ? "page" : undefined}>
                  <Icon aria-hidden="true" />
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <div className="sidebar-foot">
          <p>
            <span className="status-dot" aria-hidden="true" /> METR-LA · {r.health ? `${r.health.sensors} sensors` : "connecting…"}
          </p>
          <p>A replay of 4–27 June 2012: each forecast is shown against what happened next.</p>
        </div>
      </aside>
      <header className="topbar">
        <button type="button" className="btn btn-ghost btn-icon menu-button" aria-label="Open menu" aria-expanded={menuOpen} onClick={() => setMenuOpen(true)}>
          <Menu aria-hidden="true" />
        </button>
        <span className="phone-title">{title}</span>
        <div className="topbar-replay">
          <ReplayClock />
        </div>
        <SensorSearch />
        <button
          type="button"
          className="btn btn-ghost btn-icon theme-toggle"
          aria-label={theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme"}
          onClick={() => setThemeChoice(theme === "dark" ? "light" : "dark")}
        >
          <span key={theme} className="icon-swap">
            {theme === "dark" ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
          </span>
        </button>
      </header>
      <div className="sheet">
        <div className="replay-bar">
          <ReplayClock compact />
        </div>
        <main id="content" tabIndex={-1} key={location.pathname} className="enter">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

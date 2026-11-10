// The framed sheet (architect.md §10.2): sidebar and top bar share one tinted ground with no rule
// between them; the work sits on a white sheet inset 8px from the right and bottom edges.
import { ChartColumn, ChevronLeft, ChevronRight, Map, Menu, Moon, Pause, Play, Search, Sun, X, CalendarClock } from "lucide-react";
import { useEffect, useId, useState, type FormEvent } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router";
import { useSensors } from "../lib/queries";
import { useReplay } from "../lib/replay";
import { applyTheme } from "../lib/theme";
import { formatLong, toMs } from "../lib/time";
import { useTheme } from "./useTheme";

export function Mark({ size = 22 }: { size?: number }) {
  // A road profile that dips (a slowdown) and recovers: the product's one subject, drawn once.
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="var(--text)" />
      <path
        d="M6 12h7l3 8 3-8h7"
        fill="none"
        stroke="var(--forecast)"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const NAV = [
  { to: "/", label: "Network", icon: Map },
  { to: "/models", label: "Models", icon: ChartColumn },
  { to: "/patterns", label: "Patterns", icon: CalendarClock },
] as const;

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

  const title =
    location.pathname.startsWith("/sensor/")
      ? "Sensor"
      : (NAV.find((n) => n.to === location.pathname)?.label ?? "FlowCast");

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
        <p className="scope">METR-LA · 207 sensors, Los Angeles</p>
        <nav>
          <ul className="nav">
            {NAV.map(({ to, label, icon: Icon }) => (
              <li key={to}>
                <NavLink to={r.href(to)} end={to === "/"}>
                  <Icon aria-hidden="true" />
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <div className="sidebar-foot">
          <button
            type="button"
            className="btn btn-ghost"
            style={{ alignSelf: "flex-start", marginLeft: -12 }}
            onClick={() => applyTheme(theme === "dark" ? "light" : "dark", true)}
          >
            {theme === "dark" ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
            {theme === "dark" ? "Light theme" : "Dark theme"}
          </button>
          <p>
            A replay of 4–27 June 2012: forecasts made at each moment, shown against what happened next.
          </p>
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
      </header>
      <div className="sheet">
        <div className="replay-bar">
          <ReplayClock compact />
        </div>
        <main id="content" tabIndex={-1}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}

// The sensor map: a grey basemap that leaves all colour to the data, and one SVG circle per sensor
// with a 1px halo so overlapping markers stay apart. The basemap is OpenStreetMap's standard tiles
// turned grey by a CSS filter (app.css .map): CARTO's Positron, the first choice, now needs an API key.
import "leaflet/dist/leaflet.css";
import type { LatLngBoundsExpression, Map as LeafletMap } from "leaflet";
import { useEffect, useMemo, useState } from "react";
import { CircleMarker, MapContainer, TileLayer, Tooltip } from "react-leaflet";
import { resolveColor } from "../../lib/severity";
import { prefersReducedMotion } from "../../lib/theme";
import { useTheme } from "../useTheme";

export interface MapPoint {
  id: string;
  lat: number;
  lon: number;
  color: string; // var(--token)
  hollow?: boolean; // no reading
  tip: string;
}

const TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

export function SensorMap({
  points,
  selected,
  onSelect,
  radius = 6,
  focus,
}: {
  points: readonly MapPoint[];
  selected?: string;
  onSelect?: (id: string) => void;
  radius?: number;
  /** Centre on this sensor (fly there unless motion is reduced). */
  focus?: string;
}) {
  const theme = useTheme();
  const [map, setMap] = useState<LeafletMap | null>(null);
  const bounds = useMemo<LatLngBoundsExpression>(() => {
    const lats = points.map((p) => p.lat);
    const lons = points.map((p) => p.lon);
    return [
      [Math.min(...lats), Math.min(...lons)],
      [Math.max(...lats), Math.max(...lons)],
    ];
  }, [points]);

  // Leaflet paths need literal colours: resolve the tokens for the current theme.
  const colors = useMemo(() => {
    const cache = new Map<string, string>();
    for (const t of [...points.map((p) => p.color), "var(--marker-halo)", "var(--control)", "var(--text)", "var(--surface)"])
      if (!cache.has(t)) cache.set(t, resolveColor(t));
    return cache;
  }, [points, theme]); // theme: the same tokens resolve to new colours
  const c = (t: string) => colors.get(t) ?? t;

  useEffect(() => {
    if (!map || !focus) return;
    const p = points.find((q) => q.id === focus);
    if (!p) return;
    if (prefersReducedMotion()) map.setView([p.lat, p.lon], 13);
    else map.flyTo([p.lat, p.lon], 13, { duration: 0.6 });
  }, [map, focus, points]);

  const sel = selected ? points.find((p) => p.id === selected) : undefined;

  return (
    <MapContainer
      bounds={bounds}
      boundsOptions={{ padding: [24, 24] }}
      scrollWheelZoom={false}
      ref={setMap}
      attributionControl
      keyboard={false}
    >
      <TileLayer url={TILES} attribution={ATTRIBUTION} maxZoom={18} />
      {points.map((p) => (
        <CircleMarker
          key={p.id}
          center={[p.lat, p.lon]}
          radius={radius}
          pathOptions={{
            color: p.hollow ? c("var(--control)") : c("var(--marker-halo)"),
            weight: p.hollow ? 1.5 : 1,
            fillColor: p.hollow ? c("var(--surface)") : c(p.color),
            fillOpacity: 1,
          }}
          eventHandlers={onSelect ? { click: () => onSelect(p.id) } : undefined}
        >
          <Tooltip className="fc-tip" direction="top" offset={[0, -radius]}>
            {p.tip}
          </Tooltip>
        </CircleMarker>
      ))}
      {sel && (
        <CircleMarker
          key={`sel-${sel.id}`}
          center={[sel.lat, sel.lon]}
          radius={radius + 4}
          interactive={false}
          pathOptions={{ color: c("var(--text)"), weight: 2, fill: false }}
        />
      )}
    </MapContainer>
  );
}

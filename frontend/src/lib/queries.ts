// Server state. All data is immutable (a replayed past), so nothing goes stale.
import { keepPreviousData, QueryClient, useQuery } from "@tanstack/react-query";
import {
  getJSON,
  type DailyProfile,
  type Health,
  type Metrics,
  type ModelInfo,
  type NetworkForecast,
  type Sensor,
  type SensorDetail,
  type SensorError,
  type SensorForecast,
  type WeeklyCongestion,
} from "./api";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: Infinity, gcTime: 10 * 60_000, retry: 1, refetchOnWindowFocus: false },
  },
});

export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: () => getJSON<Health>("/api/health") });

export const useSensors = () =>
  useQuery({ queryKey: ["sensors"], queryFn: () => getJSON<Sensor[]>("/api/sensors") });

export const useSensor = (id: string) =>
  useQuery({ queryKey: ["sensor", id], queryFn: () => getJSON<SensorDetail>(`/api/sensors/${id}`) });

/** The whole network at one horizon. While a new time loads, the previous map stays on screen. */
export const useNetwork = (t: string | undefined, horizon: number, model: string | undefined) =>
  useQuery({
    queryKey: ["network", t, horizon, model],
    queryFn: () => getJSON<NetworkForecast>("/api/forecast", { as_of: t, horizon, model }),
    enabled: !!t && !!model,
    placeholderData: keepPreviousData,
  });

export const useSensorForecast = (id: string, t: string | undefined, model: string | undefined, hours = 3) =>
  useQuery({
    queryKey: ["sensor-forecast", id, t, model, hours],
    queryFn: () => getJSON<SensorForecast>(`/api/forecast/${id}`, { as_of: t, model, history_hours: hours }),
    enabled: !!t && !!model,
    placeholderData: keepPreviousData,
  });

export const useModels = () =>
  useQuery({ queryKey: ["models"], queryFn: () => getJSON<ModelInfo[]>("/api/models") });

export const useMetrics = () => useQuery({ queryKey: ["metrics"], queryFn: () => getJSON<Metrics>("/api/metrics") });

export const useSensorErrors = (model: string, horizon: number, split: string) =>
  useQuery({
    queryKey: ["sensor-errors", model, horizon, split],
    queryFn: () => getJSON<SensorError[]>("/api/metrics/sensors", { model, horizon, split }),
    placeholderData: keepPreviousData,
  });

export const useDailyProfile = () =>
  useQuery({ queryKey: ["daily-profile"], queryFn: () => getJSON<DailyProfile>("/api/analytics/daily-profile") });

export const useWeeklyCongestion = () =>
  useQuery({
    queryKey: ["weekly-congestion"],
    queryFn: () => getJSON<WeeklyCongestion>("/api/analytics/weekly-congestion"),
  });

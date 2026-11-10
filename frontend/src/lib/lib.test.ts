import { describe, expect, it } from "vitest";
import type { NetworkForecast } from "./api";
import { signed, speed, speedChange } from "./format";
import { comparedWithNow, summarize } from "./network";
import { band, isCongested } from "./severity";
import { addMinutes, clampToRange, formatClock, formatLong, isValidTime } from "./time";
import { niceTicks } from "../components/charts/LineChart";

describe("severity bands (architect.md §10.3)", () => {
  it("uses relative speed with the documented edges", () => {
    expect(band(80, 100).key).toBe("free");
    expect(band(79.9, 100).key).toBe("slowing");
    expect(band(60, 100).key).toBe("slowing");
    expect(band(59.9, 100).key).toBe("congested");
    expect(band(40, 100).key).toBe("congested");
    expect(band(39.9, 100).key).toBe("heavy");
    expect(band(25, 100).key).toBe("heavy");
    expect(band(24.9, 100).key).toBe("stop");
    expect(band(null, 100).key).toBe("missing");
    expect(band(50, null).key).toBe("missing");
  });
  it("matches the API's congestion definition (below 0.6 x free flow)", () => {
    expect(isCongested(59.9, 100)).toBe(true);
    expect(isCongested(60, 100)).toBe(false);
    expect(isCongested(null, 100)).toBe(false);
  });
});

describe("replay time", () => {
  it("is zone-free wall-clock time, never shifted by the viewer's zone", () => {
    expect(addMinutes("2012-06-12T23:55", 5)).toBe("2012-06-13T00:00");
    expect(formatClock("2012-06-12T16:30")).toBe("16:30");
    expect(formatLong("2012-06-12T16:30")).toBe("Tue 12 Jun 2012, 16:30");
  });
  it("snaps to the 5-minute grid inside the replay range", () => {
    expect(clampToRange("2012-06-12T16:32", "2012-06-04T04:10", "2012-06-27T22:55")).toBe("2012-06-12T16:30");
    expect(clampToRange("2012-07-01T00:00", "2012-06-04T04:10", "2012-06-27T22:55")).toBe("2012-06-27T22:55");
    expect(clampToRange("2012-01-01T00:00", "2012-06-04T04:10", "2012-06-27T22:55")).toBe("2012-06-04T04:10");
  });
  it("rejects malformed times from the URL", () => {
    expect(isValidTime("2012-06-12T16:30")).toBe(true);
    expect(isValidTime("yesterday")).toBe(false);
    expect(isValidTime(null)).toBe(false);
  });
});

describe("formatting", () => {
  it("prints missing values as a dash and changes with a true minus", () => {
    expect(speed(null)).toBe("–");
    expect(speed(57.6)).toBe("58");
    expect(signed(-0.078)).toBe("−0.08");
    expect(signed(0.001)).toBe("0.00");
    expect(speedChange(61, 24)).toBe("37 mph slower");
    expect(speedChange(61, 62)).toBe("steady");
  });
  it("draws round ticks from 0 with the top above the maximum", () => {
    expect(niceTicks(67)).toEqual([0, 20, 40, 60, 80]);
    expect(niceTicks(3.4)).toEqual([0, 1, 2, 3, 4]);
    expect(niceTicks(12.2)).toEqual([0, 5, 10, 15]);
  });
});

describe("network summary", () => {
  const fc: NetworkForecast = {
    model: "dcrnn",
    as_of: "2012-06-12T16:30:00",
    horizon_min: 30,
    target_time: "2012-06-12T17:00:00",
    sensors: [
      { sensor_id: "1", speed_now: 65, forecast: 64, actual: 63, free_flow: 66, congested: false },
      { sensor_id: "2", speed_now: 60, forecast: 20, actual: 22, free_flow: 66, congested: true },
      { sensor_id: "3", speed_now: 30, forecast: 35, actual: null, free_flow: 66, congested: true },
      { sensor_id: "4", speed_now: null, forecast: null, actual: null, free_flow: 66, congested: null },
      { sensor_id: "5", speed_now: 55, forecast: 45, actual: 50, free_flow: 66, congested: false },
    ],
  };
  it("counts from the rows and ranks the worst first", () => {
    const s = summarize(fc, 30);
    expect([s.total, s.reporting, s.forecasts, s.congestedNow, s.congestedThen]).toEqual([5, 4, 4, 1, 2]);
    expect(s.watch.map((w) => w.point.sensor_id)).toEqual(["2", "3", "5"]);
  });
  it("uses current speeds for horizon 0", () => {
    const s = summarize(fc, 0);
    expect(s.congestedThen).toBe(s.congestedNow);
    expect(s.watch.map((w) => w.point.sensor_id)).toEqual(["3"]); // 55 of 66 mph is still free flow
  });
  it("words the change against now", () => {
    expect(comparedWithNow(82, 63)).toBe("19 more than now");
    expect(comparedWithNow(60, 63)).toBe("3 fewer than now");
    expect(comparedWithNow(5, 5)).toBe("the same number as now");
  });
});

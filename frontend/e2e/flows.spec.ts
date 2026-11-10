// User flows against the real API. Values asserted here are the real test-period data at the default
// replay time (Tue 12 Jun 2012, 16:30), so a change in the models or the data shows up as a failure.
import { expect, test } from "@playwright/test";

const AT = "t=2012-06-12T16:30&h=30";

test.describe("Network", () => {
  test("opens on the replay time with the forecast, the map and the figures", async ({ page }) => {
    await page.goto(`./?${AT}`);
    await expect(page.getByRole("heading", { level: 1, name: "Network" })).toBeVisible();
    await expect(page.locator(".finding")).toContainText("82 sensors forecast congested by 17:00, 19 more than now");
    await expect(page.locator(".map path.fc-marker")).toHaveCount(207);
    await expect(page.locator(".replay-time").first()).toContainText("Tue 12 Jun 2012, 16:30");
    await expect(page.getByLabel("Network overview")).toContainText("207");
  });

  test("changing the horizon rewrites the URL and the finding", async ({ page }) => {
    await page.goto(`./?${AT}`);
    await page.getByRole("group", { name: "Forecast horizon" }).getByRole("button", { name: "60 min" }).click();
    await expect(page).toHaveURL(/h=60/);
    await expect(page.locator(".finding")).toContainText("by 17:30");
    await page.getByRole("button", { name: "Now" }).click();
    await expect(page.locator(".finding")).toContainText("63 sensors congested at 16:30");
  });

  test("the replay clock steps and plays", async ({ page }) => {
    await page.goto(`./?${AT}`);
    const clock = page.locator(".topbar .replay-time");
    await page.locator(".topbar").getByRole("button", { name: "Forward 5 minutes" }).click();
    await expect(clock).toContainText("16:35");
    await page.locator(".topbar").getByRole("button", { name: /^Play replay/ }).click();
    await expect(clock).toContainText("16:45", { timeout: 5_000 });
    await page.locator(".topbar").getByRole("button", { name: "Pause replay" }).click();
    await expect(page).toHaveURL(/t=2012-06-12T16%3A4[05]/);
  });

  test("a marker opens its details, then the sensor's forecast", async ({ page }) => {
    await page.goto(`./?${AT}`);
    // markers overlap on dense stretches: click where a user would, and accept whichever marker is on top
    const box = await page.locator(".map path.fc-marker").nth(40).boundingBox();
    if (!box) throw new Error("marker not rendered");
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    const popup = page.locator(".fc-popup");
    await expect(popup).toContainText("Free-flow speed");
    await popup.getByRole("button", { name: "Open forecast" }).click();
    await expect(page).toHaveURL(/\/sensor\/\d+\?/);
    await expect(page.getByRole("tab", { name: "Forecast" }).or(page.getByRole("link", { name: "Forecast" }))).toBeVisible();
  });

  test("the slowing list links to a sensor and keeps the replay state", async ({ page }) => {
    await page.goto(`./?${AT}`);
    await page.locator(".rows .row-link").first().click();
    await expect(page).toHaveURL(/\/sensor\/773939\?t=2012-06-12T16%3A30&h=30/);
  });
});

test.describe("Sensor", () => {
  test("shows the forecast against what happened", async ({ page }) => {
    await page.goto(`./sensor/773939?${AT}`);
    await expect(page.getByRole("heading", { level: 1, name: "773939" })).toBeVisible();
    await expect(page.locator(".finding")).toContainText("forecast 14 mph by 17:00");
    await expect(page.locator(".finding")).toContainText("actually 22 mph");
    await expect(page.locator(".chart svg").first()).toBeVisible();
  });

  test("a horizon cell selects that horizon", async ({ page }) => {
    await page.goto(`./sensor/773939?${AT}`);
    await page.locator("button.kpi", { hasText: "+60 min" }).click();
    await expect(page).toHaveURL(/h=60/);
    await expect(page.locator("button.kpi", { hasText: "+60 min" })).toHaveAttribute("aria-pressed", "true");
  });

  test("the history and accuracy tabs load real data", async ({ page }) => {
    await page.goto(`./sensor/773939?${AT}`);
    await page.getByRole("link", { name: "Past 24 hours" }).click();
    await expect(page.getByRole("heading", { name: "The 24 hours up to now" })).toBeVisible();
    await page.getByRole("link", { name: "Accuracy" }).click();
    await expect(page.getByRole("heading", { name: "Typical error at this sensor" })).toBeVisible();
    await expect(page.locator("table.data tbody tr")).toHaveCount(3);
  });

  test("an unknown sensor shows the API's reason, not a blank page", async ({ page }) => {
    await page.goto(`./sensor/999999?${AT}`);
    await expect(page.locator(".finding")).toContainText("unknown sensor '999999'");
  });
});

test.describe("Sensors ranking", () => {
  test("ranks all 207, sorts and filters, with an empty state", async ({ page }) => {
    await page.goto(`./sensors?${AT}`);
    await expect(page.locator("tbody tr")).toHaveCount(207);
    await page.getByRole("button", { name: "Now" }).last().click(); // the column header, not the horizon
    await expect(page.locator("tbody tr").first().locator("td").nth(2)).toContainText("7");
    await page.getByLabel("Filter by traffic state").selectOption("stop");
    await expect(page.locator(".card-head .sub")).toHaveText("10 sensors shown");
    await page.getByLabel("Filter by sensor ID").fill("000000");
    await expect(page.getByText("No sensor matches this filter.")).toBeVisible();
  });
});

test.describe("Models and Patterns", () => {
  test("the score table switches metric and highlights the best", async ({ page }) => {
    await page.goto(`./models?${AT}`);
    await expect(page.locator(".finding")).toContainText("DCRNN is the most accurate up to 45 min");
    const dcrnn = page.locator("table.data tbody tr", { hasText: "DCRNN" }).first();
    await expect(dcrnn).toContainText("2.64");
    await page.getByRole("group", { name: "Metric" }).getByRole("button", { name: "RMSE" }).click();
    await expect(dcrnn).toContainText("5.08");
    await expect(page.getByText("No clear difference")).toBeVisible();
  });

  test("the patterns grid has a cell for every day and hour", async ({ page }) => {
    await page.goto(`./patterns?${AT}`);
    await expect(page.locator(".matrix .cell")).toHaveCount(168);
    await expect(page.locator(".finding")).toContainText("Thursday at 17:00");
  });
});

test.describe("States, routing and settings", () => {
  test("a failed request shows an in-block error, and Retry recovers", async ({ page }, info) => {
    test.skip(info.project.name.startsWith("static"), "the static demo makes no API calls");
    await page.route("**/api/forecast?**", (route) => route.abort());
    await page.goto(`./?${AT}`);
    await expect(page.getByRole("alert").first()).toContainText("Couldn’t load the network forecast");
    await expect(page.getByRole("heading", { level: 1, name: "Network" })).toBeVisible(); // the frame stays
    await page.unroute("**/api/forecast?**");
    await page.getByRole("button", { name: "Retry" }).first().click();
    await expect(page.locator(".finding")).toContainText("82 sensors forecast congested");
  });

  test("an API that is down is reported where the data would be", async ({ page }, info) => {
    test.skip(info.project.name.startsWith("static"), "the static demo makes no API calls");
    await page.route("**/api/**", (route) => route.fulfill({ status: 503, body: "" }));
    await page.goto(`./models?${AT}`);
    await expect(page.getByRole("alert").first()).toContainText("Couldn’t load");
  });

  test("a garbage replay time falls back to the default; an unknown page says so", async ({ page }) => {
    await page.goto("./?t=yesterday&h=7&m=gpt");
    await expect(page.locator(".topbar .replay-time")).toContainText("Tue 12 Jun 2012, 16:30");
    await expect(page.locator(".finding")).toContainText("forecast congested by 17:00"); // h falls back to 30
    await page.goto("./no-such-page");
    await expect(page.getByRole("heading", { name: "Page not found" })).toBeVisible();
  });

  test("the theme choice persists across reloads", async ({ page }) => {
    await page.goto("./settings");
    await page.getByRole("group", { name: "Theme" }).getByRole("button", { name: "Dark" }).click();
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.getByRole("group", { name: "Theme" }).getByRole("button", { name: "Light" }).click();
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  });

  test("keyboard: the skip link reaches the content", async ({ page }) => {
    await page.goto(`./?${AT}`);
    await page.keyboard.press("Tab");
    await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.locator("main#content")).toBeFocused();
  });

  test("no console errors on any screen", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    page.on("pageerror", (e) => errors.push(e.message));
    for (const path of ["/", "/sensors", "/sensor/773939", "/models", "/patterns", "/settings"]) {
      await page.goto(`.${path}?${AT}`);
      await expect(page.locator("h1")).toBeVisible();
      await page.waitForLoadState("networkidle");
    }
    expect(errors).toEqual([]);
  });
});

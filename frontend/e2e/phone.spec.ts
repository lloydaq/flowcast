// The phone layout (Pixel 7, 412 px): the menu folds away, the replay clock moves into the page.
import { expect, test } from "@playwright/test";

const AT = "t=2012-06-12T16:30&h=30";

test("the menu opens, navigates, and closes", async ({ page }) => {
  await page.goto(`./?${AT}`);
  await expect(page.locator(".sidebar")).toBeHidden();
  await page.getByRole("button", { name: "Open menu" }).click();
  await expect(page.locator(".sidebar")).toBeVisible();
  await page.locator(".nav").getByRole("link", { name: "Models" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Models" })).toBeVisible();
  await expect(page.locator(".sidebar")).toBeHidden();
  await expect(page).toHaveURL(/\/models\?t=2012-06-12T16%3A30/);
});

test("the replay clock is in the page and the map still works", async ({ page }) => {
  await page.goto(`./?${AT}`);
  await expect(page.locator(".replay-bar .replay-time")).toContainText("Tue 12 Jun 2012, 16:30");
  await page.locator(".replay-bar").getByRole("button", { name: "Forward 5 minutes" }).click();
  await expect(page.locator(".replay-bar .replay-time")).toContainText("16:35");
  await expect(page.locator(".map path.fc-marker")).toHaveCount(207);
});

test("no screen scrolls sideways", async ({ page }) => {
  for (const path of ["/", "/sensors", "/sensor/773939", "/models", "/patterns", "/settings"]) {
    await page.goto(`.${path}?${AT}`);
    await expect(page.locator("h1")).toBeVisible();
    await page.waitForLoadState("networkidle");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow, path).toBeLessThanOrEqual(0);
  }
});

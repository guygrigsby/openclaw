import { describe, expect, it, vi } from "vitest";
import { parseThemeSelection, resolveSystemTheme, resolveTheme } from "./theme.ts";

describe("resolveTheme", () => {
  it("resolves named theme families when mode is provided", () => {
    expect(resolveTheme("neon", "dark")).toBe("neon");
    expect(resolveTheme("turtle", "light")).toBe("turtle-light");
  });

  it("uses system preference when mode is system", () => {
    vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: true }));
    expect(resolveTheme("neon", "system")).toBe("neon-light");
    vi.unstubAllGlobals();
  });
});

describe("resolveSystemTheme", () => {
  it("mirrors the active preferred color scheme", () => {
    vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: true }));
    expect(resolveSystemTheme()).toBe("talon-light");
    vi.unstubAllGlobals();
  });
});

describe("parseThemeSelection", () => {
  it("maps legacy stored values onto theme + mode", () => {
    expect(parseThemeSelection("system", undefined)).toEqual({
      theme: "talon",
      mode: "system",
    });
    expect(parseThemeSelection("fieldmanual", undefined)).toEqual({
      theme: "turtle",
      mode: "dark",
    });
  });
});

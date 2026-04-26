// @vitest-environment node
import { describe, expect, it } from "vitest";
import { humanize, isAdvancedTagSet } from "./config-form.shared.ts";

describe("humanize", () => {
  it("handles simple camelCase", () => {
    expect(humanize("requireMention")).toBe("Require @mention");
    expect(humanize("dmScope")).toBe("DM Scope");
    expect(humanize("controlUi")).toBe("Control UI");
    expect(humanize("botToken")).toBe("Bot Token");
  });

  it("preserves known acronyms when they appear as words", () => {
    expect(humanize("apiUrl")).toBe("API URL");
    expect(humanize("httpServer")).toBe("HTTP Server");
    expect(humanize("openAiKey")).toMatch(/^Open\s+AI\s+Key$/);
  });

  it("falls back to humanized camelCase for unknown keys", () => {
    expect(humanize("someRandomThing")).toBe("Some Random Thing");
    expect(humanize("snake_case_key")).toBe("Snake Case Key");
  });

  it("splits letter-digit boundaries", () => {
    expect(humanize("v2Endpoint")).toBe("V 2 Endpoint");
  });
});

describe("isAdvancedTagSet", () => {
  it("detects advanced-marker tags", () => {
    expect(isAdvancedTagSet(["advanced"])).toBe(true);
    expect(isAdvancedTagSet(["Advanced"])).toBe(true);
    expect(isAdvancedTagSet(["common", "experimental"])).toBe(true);
    expect(isAdvancedTagSet(["common"])).toBe(false);
    expect(isAdvancedTagSet([])).toBe(false);
    expect(isAdvancedTagSet(undefined)).toBe(false);
  });
});

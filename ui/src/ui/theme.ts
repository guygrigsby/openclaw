export type ThemeName = "talon" | "neon" | "turtle" | "custom";
export type ThemeMode = "system" | "light" | "dark";
export type ResolvedTheme =
  | "talon"
  | "talon-light"
  | "neon"
  | "neon-light"
  | "turtle"
  | "turtle-light"
  | "custom"
  | "custom-light";

export const VALID_THEME_NAMES = new Set<ThemeName>(["talon", "neon", "turtle", "custom"]);
export const VALID_THEME_MODES = new Set<ThemeMode>(["system", "light", "dark"]);

type ThemeSelection = { theme: ThemeName; mode: ThemeMode };

// Migrate persisted values from earlier UI builds onto the current theme set.
const LEGACY_MAP: Record<string, ThemeSelection> = {
  defaultTheme: { theme: "talon", mode: "dark" },
  docsTheme: { theme: "talon", mode: "light" },
  lightTheme: { theme: "neon", mode: "dark" },
  landingTheme: { theme: "neon", mode: "dark" },
  newTheme: { theme: "neon", mode: "dark" },
  dark: { theme: "talon", mode: "dark" },
  light: { theme: "talon", mode: "light" },
  claw: { theme: "talon", mode: "dark" },
  knot: { theme: "neon", mode: "dark" },
  dash: { theme: "turtle", mode: "dark" },
  openknot: { theme: "neon", mode: "dark" },
  fieldmanual: { theme: "turtle", mode: "dark" },
  clawdash: { theme: "turtle", mode: "light" },
  system: { theme: "talon", mode: "system" },
};

function prefersLightScheme(): boolean {
  if (typeof globalThis.matchMedia !== "function") {
    return false;
  }
  return globalThis.matchMedia("(prefers-color-scheme: light)").matches;
}

export function resolveSystemTheme(): ResolvedTheme {
  return prefersLightScheme() ? "talon-light" : "talon";
}

export function parseThemeSelection(
  themeRaw: unknown,
  modeRaw: unknown,
): { theme: ThemeName; mode: ThemeMode } {
  const theme = typeof themeRaw === "string" ? themeRaw : "";
  const mode = typeof modeRaw === "string" ? modeRaw : "";

  const normalizedTheme = VALID_THEME_NAMES.has(theme as ThemeName)
    ? (theme as ThemeName)
    : (LEGACY_MAP[theme]?.theme ?? "talon");
  const normalizedMode = VALID_THEME_MODES.has(mode as ThemeMode)
    ? (mode as ThemeMode)
    : (LEGACY_MAP[theme]?.mode ?? "system");

  return { theme: normalizedTheme, mode: normalizedMode };
}

function resolveMode(mode: ThemeMode): "light" | "dark" {
  if (mode === "system") {
    return prefersLightScheme() ? "light" : "dark";
  }
  return mode;
}

export function resolveTheme(theme: ThemeName, mode: ThemeMode): ResolvedTheme {
  const resolvedMode = resolveMode(mode);
  if (theme === "talon") {
    return resolvedMode === "light" ? "talon-light" : "talon";
  }
  if (theme === "neon") {
    return resolvedMode === "light" ? "neon-light" : "neon";
  }
  if (theme === "turtle") {
    return resolvedMode === "light" ? "turtle-light" : "turtle";
  }
  return resolvedMode === "light" ? "custom-light" : "custom";
}

import { normalizeLowercaseStringOrEmpty } from "../string-coerce.ts";
import type { ConfigUiHint, ConfigUiHints } from "../types.ts";

export type JsonSchema = {
  type?: string | string[];
  title?: string;
  description?: string;
  tags?: string[];
  "x-tags"?: string[];
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema | JsonSchema[];
  additionalProperties?: JsonSchema | boolean;
  enum?: unknown[];
  const?: unknown;
  default?: unknown;
  anyOf?: JsonSchema[];
  oneOf?: JsonSchema[];
  allOf?: JsonSchema[];
  nullable?: boolean;
};

export function schemaType(schema: JsonSchema): string | undefined {
  if (!schema) {
    return undefined;
  }
  if (Array.isArray(schema.type)) {
    return schema.type.find((t) => t !== "null") ?? schema.type[0];
  }
  return schema.type;
}

export function defaultValue(schema?: JsonSchema): unknown {
  if (!schema) {
    return "";
  }
  if (schema.default !== undefined) {
    return schema.default;
  }
  const type = schemaType(schema);
  switch (type) {
    case "object":
      return {};
    case "array":
      return [];
    case "boolean":
      return false;
    case "number":
    case "integer":
      return 0;
    case "string":
      return "";
    default:
      return "";
  }
}

export function pathKey(path: Array<string | number>): string {
  return path.filter((segment) => typeof segment === "string").join(".");
}

export function hintForPath(path: Array<string | number>, hints: ConfigUiHints) {
  const key = pathKey(path);
  const direct = hints[key];
  if (direct) {
    return direct;
  }
  const segments = path.map(String);
  for (const [hintKey, hint] of Object.entries(hints)) {
    if (!hintKey.includes("*")) {
      continue;
    }
    const hintSegments = hintKey.split(".");
    if (hintSegments.length !== segments.length) {
      continue;
    }
    let match = true;
    for (let i = 0; i < segments.length; i += 1) {
      if (hintSegments[i] !== "*" && hintSegments[i] !== segments[i]) {
        match = false;
        break;
      }
    }
    if (match) {
      return hint;
    }
  }
  return undefined;
}

// Special-case labels for well-known config keys. Lookup is case-insensitive
// against the raw key (after underscore -> space normalization). When a key is
// not in this map, we fall back to humanize() with smarter acronym handling.
const SPECIAL_CASE_LABELS: Record<string, string> = {
  // Identity / auth
  bottoken: "Bot Token",
  apikey: "API Key",
  apibase: "API Base URL",
  apiurl: "API URL",
  baseurl: "Base URL",
  authtoken: "Auth Token",
  accesstoken: "Access Token",
  refreshtoken: "Refresh Token",
  clientid: "Client ID",
  clientsecret: "Client Secret",
  appid: "App ID",
  userid: "User ID",
  serviceaccount: "Service Account",
  serviceaccountref: "Service Account Ref",
  // Channels / messaging
  dmscope: "DM Scope",
  dmsallowed: "DMs Allowed",
  dmsenabled: "DMs Enabled",
  requiremention: "Require @mention",
  requirementions: "Require @mentions",
  requirementioned: "Require @mention",
  requiredm: "Require DM",
  controlui: "Control UI",
  webchat: "Web Chat",
  // Networking
  url: "URL",
  urls: "URLs",
  uri: "URI",
  host: "Host",
  port: "Port",
  path: "Path",
  ip: "IP",
  ipv4: "IPv4",
  ipv6: "IPv6",
  dns: "DNS",
  tls: "TLS",
  ssl: "SSL",
  ssh: "SSH",
  http: "HTTP",
  https: "HTTPS",
  ws: "WebSocket",
  wss: "WebSocket (TLS)",
  cors: "CORS",
  // Protocols / surfaces
  acp: "ACP",
  mcp: "MCP",
  llm: "LLM",
  rpc: "RPC",
  jsonrpc: "JSON-RPC",
  cli: "CLI",
  sdk: "SDK",
  ui: "UI",
  ux: "UX",
  os: "OS",
  ai: "AI",
  io: "I/O",
  fs: "Filesystem",
  pwa: "PWA",
  jwt: "JWT",
  json: "JSON",
  yaml: "YAML",
  toml: "TOML",
  xml: "XML",
  uuid: "UUID",
  ttl: "TTL",
  tts: "TTS",
  stt: "STT",
  // Token budgets
  maxtokens: "Max Tokens",
  maxinputtokens: "Max Input Tokens",
  maxoutputtokens: "Max Output Tokens",
  maxcompletiontokens: "Max Completion Tokens",
  contexttokens: "Context Tokens",
  totaltokens: "Total Tokens",
  tokencount: "Token Count",
  tokenlimit: "Token Limit",
  tokenbudget: "Token Budget",
};

// Acronyms that should remain uppercase when they appear as standalone words in
// the humanized output. Order does not matter; lookup is by lowercased word.
const ACRONYM_WORDS = new Set([
  "acp",
  "ai",
  "api",
  "cli",
  "cors",
  "cpu",
  "css",
  "db",
  "dm",
  "dns",
  "fs",
  "gpu",
  "html",
  "http",
  "https",
  "id",
  "io",
  "ip",
  "json",
  "jwt",
  "llm",
  "mcp",
  "oauth",
  "os",
  "pwa",
  "rpc",
  "sdk",
  "sms",
  "sql",
  "ssh",
  "ssl",
  "stt",
  "tcp",
  "tls",
  "toml",
  "tts",
  "ttl",
  "ui",
  "url",
  "urls",
  "ux",
  "uuid",
  "vm",
  "ws",
  "wss",
  "xml",
  "yaml",
]);

export function humanize(raw: string) {
  if (!raw) {
    return "";
  }
  // Try the special-case lookup first. Normalize by removing non-alphanumerics
  // and lowercasing so dmScope, dm_scope, DM-Scope all map identically.
  const normalized = raw.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
  const special = SPECIAL_CASE_LABELS[normalized];
  if (special) {
    return special;
  }

  // Insert spaces between camelCase, between letters and digits, and on
  // contiguous-acronym -> word boundaries (e.g. "DMScope" -> "DM Scope",
  // "HTTPServer" -> "HTTP Server").
  const spaced = raw
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/([a-zA-Z])([0-9])/g, "$1 $2")
    .replace(/([0-9])([a-zA-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim();

  // Capitalize each word, but keep known acronyms uppercase.
  return spaced
    .split(" ")
    .map((word) => {
      const lower = word.toLowerCase();
      if (ACRONYM_WORDS.has(lower)) {
        return lower.toUpperCase();
      }
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(" ");
}

// Tags that mark a field as "advanced" — rendered in a collapsed group below
// the common settings to reduce visual noise.
const ADVANCED_TAG_VALUES = new Set([
  "advanced",
  "expert",
  "experimental",
  "internal",
  "debug",
  "diagnostic",
  "diagnostics",
  "hidden",
  "deprecated",
]);

export function isAdvancedTagSet(tags: readonly string[] | undefined): boolean {
  if (!tags || tags.length === 0) {
    return false;
  }
  for (const tag of tags) {
    if (typeof tag !== "string") {
      continue;
    }
    if (ADVANCED_TAG_VALUES.has(tag.trim().toLowerCase())) {
      return true;
    }
  }
  return false;
}

const SENSITIVE_KEY_WHITELIST_SUFFIXES = [
  "maxtokens",
  "maxoutputtokens",
  "maxinputtokens",
  "maxcompletiontokens",
  "contexttokens",
  "totaltokens",
  "tokencount",
  "tokenlimit",
  "tokenbudget",
  "passwordfile",
] as const;

const SENSITIVE_PATTERNS = [
  /token$/i,
  /password/i,
  /secret/i,
  /api.?key/i,
  /serviceaccount(?:ref)?$/i,
];

const ENV_VAR_PLACEHOLDER_PATTERN = /^\$\{[^}]*\}$/;

export const REDACTED_PLACEHOLDER = "[redacted - click reveal to view]";

const MAX_SENSITIVE_SCAN_DEPTH = 64;
const MAX_SENSITIVE_SCAN_NODES = 20_000;

type SensitiveScanState = {
  visited: number;
};

function createSensitiveScanState(): SensitiveScanState {
  return { visited: 0 };
}

function enterSensitiveScanNode(state: SensitiveScanState, depth: number): boolean {
  if (depth > MAX_SENSITIVE_SCAN_DEPTH) {
    return false;
  }
  state.visited += 1;
  if (state.visited > MAX_SENSITIVE_SCAN_NODES) {
    return false;
  }
  return true;
}

function isEnvVarPlaceholder(value: string): boolean {
  return ENV_VAR_PLACEHOLDER_PATTERN.test(value.trim());
}

export function isSensitiveConfigPath(path: string): boolean {
  const lowerPath = normalizeLowercaseStringOrEmpty(path);
  const whitelisted = SENSITIVE_KEY_WHITELIST_SUFFIXES.some((suffix) => lowerPath.endsWith(suffix));
  return !whitelisted && SENSITIVE_PATTERNS.some((pattern) => pattern.test(path));
}

function isSensitiveLeafValue(value: unknown): boolean {
  if (typeof value === "string") {
    return value.trim().length > 0 && !isEnvVarPlaceholder(value);
  }
  return value !== undefined && value !== null;
}

function isHintSensitive(hint: ConfigUiHint | undefined): boolean {
  return hint?.sensitive ?? false;
}

export function hasSensitiveConfigData(
  value: unknown,
  path: Array<string | number>,
  hints: ConfigUiHints,
): boolean {
  return hasSensitiveConfigDataInner(value, path, hints, createSensitiveScanState(), 0);
}

function hasSensitiveConfigDataInner(
  value: unknown,
  path: Array<string | number>,
  hints: ConfigUiHints,
  scan: SensitiveScanState,
  depth: number,
): boolean {
  if (!enterSensitiveScanNode(scan, depth)) {
    return true;
  }

  const key = pathKey(path);
  const hint = hintForPath(path, hints);
  const pathIsSensitive = isHintSensitive(hint) || isSensitiveConfigPath(key);

  if (pathIsSensitive && isSensitiveLeafValue(value)) {
    return true;
  }

  if (Array.isArray(value)) {
    return value.some((item, index) =>
      hasSensitiveConfigDataInner(item, [...path, index], hints, scan, depth + 1),
    );
  }

  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).some(([childKey, childValue]) =>
      hasSensitiveConfigDataInner(childValue, [...path, childKey], hints, scan, depth + 1),
    );
  }

  return false;
}

export function countSensitiveConfigValues(
  value: unknown,
  path: Array<string | number>,
  hints: ConfigUiHints,
): number {
  return countSensitiveConfigValuesInner(value, path, hints, createSensitiveScanState(), 0);
}

function countSensitiveConfigValuesInner(
  value: unknown,
  path: Array<string | number>,
  hints: ConfigUiHints,
  scan: SensitiveScanState,
  depth: number,
): number {
  if (!enterSensitiveScanNode(scan, depth)) {
    return 1;
  }

  if (value == null) {
    return 0;
  }

  const key = pathKey(path);
  const hint = hintForPath(path, hints);
  const pathIsSensitive = isHintSensitive(hint) || isSensitiveConfigPath(key);

  if (pathIsSensitive && isSensitiveLeafValue(value)) {
    return 1;
  }

  if (Array.isArray(value)) {
    return value.reduce(
      (count, item, index) =>
        count + countSensitiveConfigValuesInner(item, [...path, index], hints, scan, depth + 1),
      0,
    );
  }

  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).reduce(
      (count, [childKey, childValue]) =>
        count +
        countSensitiveConfigValuesInner(childValue, [...path, childKey], hints, scan, depth + 1),
      0,
    );
  }

  return 0;
}

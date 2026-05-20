// AI & Agents → Overview section. Curated landing surface that
// front-loads the settings users actually want to change (default
// model, fallbacks) instead of making them dig through the schema-
// rendered tree under Agents → defaults → model → primary.
//
// Reads from props.formValue and writes via props.onFormPatch — same
// in-flight form state the deeper sections use, so changes here are
// staged identically and applied via the global Apply / Save buttons.

import { html, nothing } from "lit";
import type { TemplateResult } from "lit";

type FormPatchFn = (path: Array<string | number>, value: unknown) => void;

type AIOverviewProps = {
  formValue: Record<string, unknown> | null;
  onFormPatch: FormPatchFn;
  disabled: boolean;
};

// readPath walks a typed object path and returns the leaf value (or
// undefined). Mirrors what onFormPatch writes — keeps reads / writes
// symmetric.
function readPath(value: unknown, path: Array<string | number>): unknown {
  let cur: unknown = value;
  for (const seg of path) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string | number, unknown>)[seg];
  }
  return cur;
}

function readString(value: unknown, path: Array<string | number>): string {
  const v = readPath(value, path);
  return typeof v === "string" ? v : "";
}

function readStringArray(value: unknown, path: Array<string | number>): string[] {
  const v = readPath(value, path);
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

// configuredModelIds collects the model ids the user has actually
// touched: the primary + fallbacks, the agents.defaults.models map
// keys, and the per-agent agents.list[].model fields. Order matters —
// the primary gets surfaced first, then fallbacks (so the dropdown's
// natural order matches the user's intent), then everything else
// alphabetical so unconfigured-but-known models stay predictable.
function configuredModelIds(formValue: Record<string, unknown> | null): string[] {
  if (!formValue) return [];
  const out: string[] = [];
  const seen = new Set<string>();
  const add = (id: unknown) => {
    if (typeof id !== "string" || !id || seen.has(id)) return;
    seen.add(id);
    out.push(id);
  };
  add(readString(formValue, ["agents", "defaults", "model", "primary"]));
  for (const id of readStringArray(formValue, ["agents", "defaults", "model", "fallbacks"])) {
    add(id);
  }
  const defaultsModels = readPath(formValue, ["agents", "defaults", "models"]);
  if (defaultsModels && typeof defaultsModels === "object") {
    for (const id of Object.keys(defaultsModels as Record<string, unknown>)) {
      add(id);
    }
  }
  const list = readPath(formValue, ["agents", "list"]);
  if (Array.isArray(list)) {
    for (const entry of list) {
      if (entry && typeof entry === "object") {
        add((entry as Record<string, unknown>).model);
      }
    }
  } else if (list && typeof list === "object") {
    for (const entry of Object.values(list as Record<string, unknown>)) {
      if (entry && typeof entry === "object") {
        add((entry as Record<string, unknown>).model);
      }
    }
  }
  return out;
}

function countAgents(formValue: Record<string, unknown> | null): number {
  if (!formValue) return 0;
  const list = readPath(formValue, ["agents", "list"]);
  if (Array.isArray(list)) return list.length;
  if (list && typeof list === "object") return Object.keys(list).length;
  return 0;
}

export function renderAIOverviewSection(props: AIOverviewProps): TemplateResult {
  const primary = readString(props.formValue, ["agents", "defaults", "model", "primary"]);
  const fallbacks = readStringArray(props.formValue, ["agents", "defaults", "model", "fallbacks"]);
  const ids = configuredModelIds(props.formValue);
  const agentCount = countAgents(props.formValue);

  // The select is "configured + custom": the dropdown is populated
  // from configured ids so users can swap between models they've
  // already set up; the input below it accepts any model id (e.g.
  // newly published models the catalog hasn't picked up yet).
  const onSelectPrimary = (e: Event) => {
    const next = (e.target as HTMLSelectElement).value;
    if (next === "__custom__") return; // ignore the placeholder
    props.onFormPatch(["agents", "defaults", "model", "primary"], next);
  };
  const onPrimaryInput = (e: Event) => {
    const next = (e.target as HTMLInputElement).value;
    props.onFormPatch(["agents", "defaults", "model", "primary"], next);
  };

  // Fallbacks edited as a comma-separated string. Trade-off: simpler
  // than a chip editor; suitable for the small lists users actually
  // configure (3–5 models). Trimmed + de-empty before patching.
  const fallbackText = fallbacks.join(", ");
  const onFallbacksInput = (e: Event) => {
    const raw = (e.target as HTMLInputElement).value;
    const parsed = raw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    props.onFormPatch(["agents", "defaults", "model", "fallbacks"], parsed);
  };

  return html`
    <div class="settings-appearance">
      <div class="settings-appearance__section">
        <h3 class="settings-appearance__heading">Default model</h3>
        <p class="settings-appearance__hint">
          The model agents use when none is set per-agent. Provider-prefixed
          (<code>provider/model</code>) — for example <code>anthropic/claude-sonnet-4-6</code> or
          <code>deepseek/deepseek-chat</code>.
        </p>
        <div class="settings-info-grid" style="margin-top: 12px;">
          <label class="settings-info-row" style="display: grid; gap: 6px;">
            <span class="settings-info-row__label">Primary</span>
            <select
              class="settings-info-row__value"
              .value=${primary}
              ?disabled=${props.disabled}
              @change=${onSelectPrimary}
              style="min-width: 280px;"
            >
              ${ids.length === 0
                ? html`<option value="">(none configured)</option>`
                : ids.map(
                    (id) => html`<option value=${id} ?selected=${id === primary}>${id}</option>`,
                  )}
              ${primary && !ids.includes(primary)
                ? html`<option value=${primary} selected>${primary}</option>`
                : nothing}
            </select>
            <input
              type="text"
              .value=${primary}
              ?disabled=${props.disabled}
              @change=${onPrimaryInput}
              placeholder="provider/model-id"
              style="min-width: 280px; font-family: var(--font-mono, monospace);"
            />
          </label>
          <label class="settings-info-row" style="display: grid; gap: 6px;">
            <span class="settings-info-row__label">Fallbacks</span>
            <input
              type="text"
              .value=${fallbackText}
              ?disabled=${props.disabled}
              @change=${onFallbacksInput}
              placeholder="anthropic/claude-opus-4-7, openai/gpt-4o, ..."
              style="min-width: 280px; font-family: var(--font-mono, monospace);"
            />
            <span class="settings-appearance__hint" style="margin: 0;">
              Comma-separated list. Tried in order if the primary fails.
            </span>
          </label>
        </div>
      </div>

      <div class="settings-appearance__section">
        <h3 class="settings-appearance__heading">Agents</h3>
        <div class="settings-info-grid">
          <div class="settings-info-row">
            <span class="settings-info-row__label">Configured</span>
            <span class="settings-info-row__value">${agentCount}</span>
          </div>
          <div class="settings-info-row">
            <span class="settings-info-row__label">Models in use</span>
            <span class="settings-info-row__value">${ids.length}</span>
          </div>
        </div>
        <p class="settings-appearance__hint" style="margin-top: 10px;">
          Per-agent settings (workspace, tools, model override) live under
          <strong>Agents</strong>. Use this overview to change the global default; switch tabs for
          everything else.
        </p>
      </div>
    </div>
  `;
}

// Plugins tab — manage runtime dependencies for bundled openclaw
// extensions. Visual format mirrors the Skills tab so the page reads
// as a peer of that surface (same card wrapper, status tabs, filter,
// list-row layout, btn atoms). The bundled extensions don't have a
// ClawHub-style registry — install means "fetch the npm deps for an
// extension that's already on disk" — so the section after the
// status tabs is a simple list.

import { LitElement, html, nothing } from "lit";
import { customElement, property, state } from "lit/decorators.js";
import type { GatewayBrowserClient } from "../gateway.ts";

type StatusItem = {
  name: string;
  path: string;
  // Source is internal plumbing — talon vs openclaw vs bundled vs
  // builtin. We don't surface it directly to users; the UI converts
  // it to friendlier signals (Loaded / Bundled / In use).
  source: "talon" | "openclaw" | "bundled" | "builtin" | string;
  description?: string;
  version?: string;
  kind?: "channel" | "provider" | "plugin" | string;
  label?: string;
  hasPackageJson: boolean;
  depCount: number;
  installed: boolean;
  nodeModulesExists: boolean;
  inUse: boolean;
  uninstallable: boolean;
  loaded: boolean;
  error?: string;
};

type DetailResult = StatusItem & {
  dependencies?: Record<string, string>;
  blurb?: string;
  docsPath?: string;
  channelId?: string;
  packageName?: string;
};

type StatusResult = {
  items: StatusItem[];
  sources: { label: string; path: string }[];
  writeRoot: string;
};

type InstallResult = {
  ok: boolean;
  name?: string;
  status?: StatusItem;
  output?: string;
  error?: string;
  skipped?: string;
};

type InstallState =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "ok"; output?: string }
  | { kind: "error"; message: string; output?: string };

type StatusFilter = "all" | "loaded" | "native" | "ready" | "needs-install" | "error";

// Telegram setup wizard state. Threaded through three RPCs in order:
// channels.telegram.verify → captureSender → persist. The user sees the
// wizard as a single modal with the current step's UI rendered inline.
type TelegramBot = { id: number; username: string; firstName: string };
type TelegramSender = { chatId: number; senderId: number; displayName: string };
type WizardState =
  | { kind: "closed" }
  | { kind: "input"; token: string; error?: string }
  | { kind: "verifying"; token: string }
  | { kind: "verified"; token: string; bot: TelegramBot }
  | { kind: "waiting"; token: string; bot: TelegramBot; startedAt: number }
  | {
      kind: "captured";
      token: string;
      bot: TelegramBot;
      sender: TelegramSender;
    }
  | {
      kind: "persisting";
      token: string;
      bot: TelegramBot;
      sender: TelegramSender;
    }
  | {
      kind: "done";
      bot: TelegramBot;
      sender: TelegramSender;
      restartHint: string;
      confirmWarning?: string;
    }
  | { kind: "error"; step: string; message: string };

const STATUS_TABS: { id: StatusFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "loaded", label: "Loaded" },
  { id: "native", label: "Native" },
  { id: "ready", label: "Ready" },
  { id: "needs-install", label: "Needs Install" },
  { id: "error", label: "Error" },
];

// nativeBucket reports whether an item belongs in the "Native" tab.
// Distinct from the row's primary status (loaded / ready / needs-install)
// — a native plugin is also typically loaded, so the Native tab cuts
// across the others rather than replacing them.
function isNativePlugin(item: StatusItem): boolean {
  return item.source === "builtin";
}

function pluginStatusKind(item: StatusItem, install: InstallState): StatusFilter {
  if (item.error || install.kind === "error") return "error";
  if (item.loaded) return "loaded";
  if (item.depCount === 0 || item.installed) return "ready";
  return "needs-install";
}

function pluginDotClass(kind: StatusFilter): string {
  switch (kind) {
    case "loaded":
      return "ok";
    case "ready":
      return "ok";
    case "needs-install":
      return "warn";
    case "error":
      return "danger";
    default:
      return "muted";
  }
}

@customElement("talon-plugins-view")
export class TalonPluginsView extends LitElement {
  // Render to light DOM so openclaw's global stylesheet (.card,
  // .agent-tabs, .list-item, .btn, .statusDot, etc.) applies. With
  // shadow DOM these classes wouldn't pick up the global styles and
  // the page would look like a 1990s spreadsheet.
  protected override createRenderRoot() {
    return this;
  }

  @property({ attribute: false })
  client: GatewayBrowserClient | null = null;

  @state() private items: StatusItem[] = [];
  @state() sources: { label: string; path: string }[] = [];
  @state() writeRoot = "";
  @state() private loading = false;
  @state() private listError: string | null = null;
  @state() private install: Record<string, InstallState> = {};
  @state() private installAllRunning = false;
  @state() private statusFilter: StatusFilter = "all";
  @state() private filter = "";
  // Drill-down state. detailKey is the currently-expanded row name;
  // detailMap caches detail responses across expand/collapse cycles
  // so re-expanding doesn't re-fetch unless the user explicitly
  // refreshes.
  @state() private detailKey: string | null = null;
  @state() private detailMap: Record<string, DetailResult | { error: string } | "loading"> = {};

  // Telegram setup wizard. Single instance — only one channel wizard
  // open at a time. Closed by default.
  @state() private wizard: WizardState = { kind: "closed" };

  override updated() {
    if (this.client && !this.loading && this.items.length === 0 && !this.listError) {
      void this.refresh();
    }
  }

  private async refresh() {
    if (!this.client) return;
    this.loading = true;
    this.listError = null;
    try {
      const res = await this.client.request<StatusResult>("plugins.deps.status", {});
      this.items = res?.items ?? [];
      this.sources = res?.sources ?? [];
      this.writeRoot = res?.writeRoot ?? "";
    } catch (err) {
      this.listError = String(err);
    } finally {
      this.loading = false;
    }
  }

  private async installOne(name: string) {
    if (!this.client) return;
    this.install = { ...this.install, [name]: { kind: "running" } };
    try {
      const res = await this.client.request<InstallResult>("plugins.deps.install", { name });
      if (res?.ok) {
        this.install = { ...this.install, [name]: { kind: "ok", output: res.output } };
        if (res.status) {
          this.items = this.items.map((it) => (it.name === name ? res.status! : it));
        }
      } else {
        this.install = {
          ...this.install,
          [name]: {
            kind: "error",
            message: res?.error ?? "install failed",
            output: res?.output,
          },
        };
      }
    } catch (err) {
      this.install = {
        ...this.install,
        [name]: { kind: "error", message: String(err) },
      };
    }
  }

  private async uninstallOne(item: StatusItem) {
    if (!this.client) return;
    if (item.inUse) {
      const ok = window.confirm(
        `${item.label || item.name} is currently in use (${
          item.kind === "channel" ? "channel binding" : "plugins.entries"
        }). Uninstalling will break that integration until you reinstall. Continue?`,
      );
      if (!ok) return;
    }
    this.install = { ...this.install, [item.name]: { kind: "running" } };
    try {
      const res = await this.client.request<InstallResult>("plugins.deps.uninstall", {
        name: item.name,
      });
      if (res?.ok) {
        this.install = {
          ...this.install,
          [item.name]: { kind: "ok", output: res.output },
        };
        if (res.status) {
          this.items = this.items.map((it) => (it.name === item.name ? res.status! : it));
        } else {
          // Extension is gone from every layer — drop the row.
          this.items = this.items.filter((it) => it.name !== item.name);
        }
        // Drop any cached detail since the source layer changed.
        const next = { ...this.detailMap };
        delete next[item.name];
        this.detailMap = next;
      } else {
        this.install = {
          ...this.install,
          [item.name]: {
            kind: "error",
            message: res?.error ?? "uninstall failed",
            output: res?.output,
          },
        };
      }
    } catch (err) {
      this.install = {
        ...this.install,
        [item.name]: { kind: "error", message: String(err) },
      };
    }
  }

  // --- Telegram setup wizard --------------------------------------------

  private openTelegramWizard() {
    this.wizard = { kind: "input", token: "" };
  }

  private closeWizard = () => {
    this.wizard = { kind: "closed" };
  };

  private async wizardVerify() {
    if (this.wizard.kind !== "input" || !this.client) return;
    const token = this.wizard.token.trim();
    if (!token) {
      this.wizard = { ...this.wizard, error: "Bot token is required." };
      return;
    }
    this.wizard = { kind: "verifying", token };
    try {
      const res = await this.client.request<{ ok: boolean; bot: TelegramBot }>(
        "channels.telegram.verify",
        { token },
      );
      if (!res?.ok || !res.bot) {
        this.wizard = { kind: "error", step: "verify", message: "Token verification failed." };
        return;
      }
      this.wizard = { kind: "verified", token, bot: res.bot };
    } catch (err) {
      this.wizard = { kind: "error", step: "verify", message: String(err) };
    }
  }

  private async wizardCapture() {
    if (this.wizard.kind !== "verified" || !this.client) return;
    const { token, bot } = this.wizard;
    this.wizard = { kind: "waiting", token, bot, startedAt: Date.now() };
    try {
      const res = await this.client.request<TelegramSender>("channels.telegram.captureSender", {
        token,
        deadlineSec: 90,
      });
      if (!res?.senderId) {
        this.wizard = {
          kind: "error",
          step: "capture",
          message: "No message captured. Did you DM the bot?",
        };
        return;
      }
      this.wizard = { kind: "captured", token, bot, sender: res };
    } catch (err) {
      this.wizard = { kind: "error", step: "capture", message: String(err) };
    }
  }

  private async wizardPersist() {
    if (this.wizard.kind !== "captured" || !this.client) return;
    const { token, bot, sender } = this.wizard;
    this.wizard = { kind: "persisting", token, bot, sender };
    try {
      const res = await this.client.request<{
        ok: boolean;
        restartHint: string;
        confirmWarning?: string;
      }>("channels.telegram.persist", {
        token,
        senderId: sender.senderId,
        chatId: sender.chatId,
        agentId: "main",
      });
      if (!res?.ok) {
        this.wizard = { kind: "error", step: "persist", message: "Persist failed." };
        return;
      }
      this.wizard = {
        kind: "done",
        bot,
        sender,
        restartHint: res.restartHint,
        confirmWarning: res.confirmWarning,
      };
      // Refresh the plugins list so the telegram row picks up the
      // newly-loaded state next time the user reopens the page (or
      // after the gateway restart). No-op for the still-running list.
      void this.refresh();
    } catch (err) {
      this.wizard = { kind: "error", step: "persist", message: String(err) };
    }
  }

  private async toggleDetail(name: string) {
    if (this.detailKey === name) {
      this.detailKey = null;
      return;
    }
    this.detailKey = name;
    if (this.detailMap[name] && this.detailMap[name] !== "loading") {
      return;
    }
    if (!this.client) return;
    this.detailMap = { ...this.detailMap, [name]: "loading" };
    try {
      const res = await this.client.request<DetailResult>("plugins.deps.detail", { name });
      this.detailMap = { ...this.detailMap, [name]: res };
    } catch (err) {
      this.detailMap = { ...this.detailMap, [name]: { error: String(err) } };
    }
  }

  private async installAll(items: StatusItem[]) {
    if (this.installAllRunning) return;
    this.installAllRunning = true;
    try {
      // Sequence installs rather than parallelize: npm subprocesses
      // sharing a registry cache can race, and running 50 at once is
      // a network spike the user almost certainly doesn't want.
      for (const item of items) {
        if (item.installed || item.depCount === 0) continue;
        await this.installOne(item.name);
      }
    } finally {
      this.installAllRunning = false;
    }
  }

  private renderRow(item: StatusItem) {
    const installState = this.install[item.name] ?? { kind: "idle" };
    const kind = pluginStatusKind(item, installState);
    const dot = pluginDotClass(kind);
    const isRunning = installState.kind === "running";
    const showOutput = installState.kind === "error" || installState.kind === "ok";
    const outputBody =
      installState.kind === "error" || installState.kind === "ok"
        ? (installState.output ?? "")
        : "";
    const expanded = this.detailKey === item.name;

    // Compose the action group: install/reinstall on the left,
    // uninstall on the right when applicable. The user wanted clear
    // signals about WHICH plugins have which buttons, so we render
    // the reasoning explicitly when no button applies.
    const actions: unknown[] = [];
    if (isRunning) {
      actions.push(html`<button class="btn btn--sm" disabled>Working…</button>`);
    } else {
      if (item.depCount > 0) {
        actions.push(html`<button
          class=${"btn btn--sm" + (item.installed ? "" : " primary")}
          @click=${(e: Event) => {
            e.stopPropagation();
            this.installOne(item.name);
          }}
        >
          ${item.installed ? "Reinstall" : "Install"}
        </button>`);
      }
      if (item.uninstallable) {
        actions.push(html`<button
          class="btn btn--sm"
          @click=${(e: Event) => {
            e.stopPropagation();
            this.uninstallOne(item);
          }}
        >
          Uninstall
        </button>`);
      }
    }

    // When there's no actionable button, surface a one-line
    // explanation right where the button would have been. Avoids
    // the "why does this row have no button" mystery.
    //
    // Native plugins are intentionally button-less today — the
    // "Native" badge already says "compiled into talon, no install
    // step." Adding a "Read-only" reason on top would imply a
    // limitation (you wanted to remove it but can't), which isn't
    // the case — there's nothing to remove.
    if (actions.length === 0 && !isNativePlugin(item)) {
      let reason = "";
      if (!item.uninstallable) {
        reason = "Read-only";
      } else if (item.depCount === 0) {
        reason = "No deps to install";
      }
      if (reason) {
        actions.push(html`<span class="muted" style="font-size: 12px;">${reason}</span>`);
      }
    }

    // Sub line: prefer the package.json description; fall back to an
    // install-state message or the metadata one-liner.
    let sub = "";
    if (installState.kind === "error") {
      sub = installState.message;
    } else if (item.error) {
      sub = item.error;
    } else if (item.description) {
      sub = item.description;
    } else if (item.depCount === 0) {
      sub = "No runtime dependencies";
    } else {
      sub = `${item.depCount} ${item.depCount === 1 ? "dependency" : "dependencies"}`;
    }

    const kindBadge =
      item.kind && item.kind !== "plugin"
        ? html`<span
            class="muted"
            style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 2px 6px; border-radius: 4px; background: var(--bg-elev);"
            >${item.kind}</span
          >`
        : nothing;

    const inUseBadge = item.inUse
      ? html`<span
          style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 2px 6px; border-radius: 4px; background: var(--accent-subtle, rgba(99, 102, 241, 0.15)); color: var(--accent, rgb(67, 56, 202)); font-weight: 600;"
          title="Referenced by your active config — uninstalling will break this integration"
          >In use</span
        >`
      : nothing;

    const loadedBadge = item.loaded
      ? html`<span
          style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 2px 6px; border-radius: 4px; background: rgba(16, 185, 129, 0.18); color: rgb(6, 95, 70); font-weight: 600;"
          title="Currently running as a subprocess plugin"
          >Loaded</span
        >`
      : nothing;

    // "Native" surfaces the meaningful runtime distinction:
    // statically-compiled, zero-runtime-dep plugins vs the openclaw
    // shim's interpreted-extension path. The label is what users
    // actually care about — implementation language is hidden.
    const nativeBadge =
      item.source === "builtin"
        ? html`<span
            style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 2px 6px; border-radius: 4px; background: rgba(59, 130, 246, 0.15); color: rgb(30, 64, 175); font-weight: 600;"
            title="Native plugin: statically compiled, no runtime dependencies, no npm install required"
            >Native</span
          >`
        : nothing;

    return html`
      <div
        class="list-item list-item-clickable"
        @click=${() => this.toggleDetail(item.name)}
        style="cursor: pointer;"
      >
        <div class="list-main">
          <div
            class="list-title"
            style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;"
          >
            <span class="statusDot ${dot}"></span>
            <span>${item.label || item.name}</span>
            ${item.label && item.label !== item.name
              ? html`<span class="muted" style="font-size: 12px;">${item.name}</span>`
              : nothing}
            ${kindBadge} ${nativeBadge} ${loadedBadge} ${inUseBadge}
            ${item.version
              ? html`<span class="muted" style="font-size: 12px;">v${item.version}</span>`
              : nothing}
          </div>
          <div class="list-sub">${sub}</div>
          ${showOutput && outputBody
            ? html`<details style="margin-top: 6px;" @click=${(e: Event) => e.stopPropagation()}>
                <summary class="muted" style="cursor: pointer; font-size: 12px;">
                  npm output
                </summary>
                <pre
                  style="margin: 6px 0 0; padding: 8px; background: var(--bg-elev); border-radius: 4px; font-size: 12px; max-height: 12rem; overflow: auto; white-space: pre-wrap; word-break: break-word;"
                >
${outputBody}</pre
                >
              </details>`
            : nothing}
        </div>
        <div
          class="list-meta"
          style="display: flex; align-items: center; justify-content: flex-end; gap: 8px; flex-wrap: wrap;"
          @click=${(e: Event) => e.stopPropagation()}
        >
          ${actions}
        </div>
      </div>
      ${expanded ? this.renderDetail(item) : nothing}
    `;
  }

  private renderDetail(item: StatusItem) {
    const cached = this.detailMap[item.name];
    // The detail card is one drawer with two stacked sections:
    // (1) detail content (blurb / deps / metadata) and (2) the
    // per-row wizard slot. The wizard renders even while details
    // are still loading or errored, so users can run the telegram
    // setup independent of detail-fetch state.
    let detailBody: unknown = nothing;
    if (cached === "loading") {
      detailBody = html`<div class="muted">Loading details…</div>`;
    } else if (cached && "error" in cached) {
      detailBody = html`<div class="callout danger" style="margin: 0;">${cached.error}</div>`;
    } else if (cached) {
      const detail = cached;
      const deps = detail.dependencies ?? {};
      const depEntries = Object.entries(deps).sort(([a], [b]) => a.localeCompare(b));
      detailBody = html`
        ${detail.blurb
          ? html`<div style="font-size: 14px; line-height: 1.5;">${detail.blurb}</div>`
          : nothing}
        <div style="display: grid; gap: 4px; font-size: 12px;">
          ${detail.packageName
            ? html`<div class="muted">npm package: <code>${detail.packageName}</code></div>`
            : nothing}
          ${detail.channelId
            ? html`<div class="muted">channel id: <code>${detail.channelId}</code></div>`
            : nothing}
          ${detail.docsPath
            ? html`<div class="muted">docs: <code>${detail.docsPath}</code></div>`
            : nothing}
        </div>
        ${depEntries.length > 0
          ? html`
              <div>
                <div style="font-weight: 600; margin-bottom: 6px; font-size: 13px;">
                  Dependencies (${depEntries.length})
                </div>
                <div
                  style="display: grid; gap: 2px; font-size: 12px; font-family: var(--font-mono, monospace);"
                >
                  ${depEntries.map(
                    ([name, version]) => html`<div
                      style="display: flex; justify-content: space-between; padding: 2px 0;"
                    >
                      <span>${name}</span>
                      <span class="muted">${version}</span>
                    </div>`,
                  )}
                </div>
              </div>
            `
          : nothing}
      `;
    }
    const hasDetailBody = detailBody !== nothing;
    const hasWizardSlot = item.name === "telegram" && item.kind === "channel";
    // The drawer attaches to the bottom of the row above by consuming
    // the 8px grid-gap (.list grid) with a -8px top margin and using
    // matching radius. border-top: none + same border var as the row
    // make it read as a continuation of the row rather than a
    // floating panel — the bottom corners and border cleanly cap it
    // off so it doesn't bleed into the next section.
    return html`
      <div
        style="margin: -8px 0 16px; padding: 14px 16px; background: var(--bg-elev); border: 1px solid var(--border); border-top: none; border-radius: 0 0 var(--radius-md) var(--radius-md); display: grid; gap: 12px;"
      >
        ${detailBody}
        ${hasDetailBody && hasWizardSlot
          ? html`<div style="height: 1px; background: var(--border); margin: 0 -16px;"></div>`
          : nothing}
        ${this.renderInlineWizardSlot(item)}
      </div>
    `;
  }

  // renderInlineWizardSlot returns the per-row setup section that lives
  // inside the drawer. Today only telegram has a wizard; other rows
  // get nothing. Section composes with the rest of the drawer (deps,
  // blurb, etc.) — it's not a replacement.
  private renderInlineWizardSlot(item: StatusItem) {
    if (item.name !== "telegram" || item.kind !== "channel") return nothing;
    const w = this.wizard;
    return html`
      <div style="display: grid; gap: 10px;">
        <div
          style="display: flex; align-items: baseline; justify-content: space-between; gap: 12px;"
        >
          <div>
            <div style="font-weight: 600; font-size: 13px;">Telegram setup</div>
            <div class="muted" style="font-size: 12px;">${this.wizardSubtitle(w)}</div>
          </div>
          ${w.kind === "closed"
            ? html`<button class="btn btn--sm primary" @click=${() => this.openTelegramWizard()}>
                ${item.inUse ? "Reconfigure" : "Configure"}
              </button>`
            : w.kind !== "verifying" && w.kind !== "waiting" && w.kind !== "persisting"
              ? html`<button class="btn btn--sm" @click=${this.closeWizard}>Reset</button>`
              : nothing}
        </div>
        ${w.kind !== "closed"
          ? html`<div style="display: grid; gap: 12px;">${this.renderWizardBody(w)}</div>`
          : nothing}
      </div>
    `;
  }

  override render() {
    const counts: Record<StatusFilter, number> = {
      all: this.items.length,
      loaded: 0,
      native: 0,
      ready: 0,
      "needs-install": 0,
      error: 0,
    };
    for (const it of this.items) {
      const k = pluginStatusKind(it, this.install[it.name] ?? { kind: "idle" });
      counts[k]++;
      if (isNativePlugin(it)) counts.native++;
    }

    const afterStatus = (() => {
      if (this.statusFilter === "all") return this.items;
      if (this.statusFilter === "native") return this.items.filter(isNativePlugin);
      return this.items.filter(
        (it) =>
          pluginStatusKind(it, this.install[it.name] ?? { kind: "idle" }) === this.statusFilter,
      );
    })();

    const f = this.filter.trim().toLowerCase();
    const filtered = f
      ? afterStatus.filter((it) => [it.name, it.source].join(" ").toLowerCase().includes(f))
      : afterStatus;

    const installable = this.items.filter((it) => !it.installed && it.depCount > 0);

    return html`
      <section class="card">
        <div class="row" style="justify-content: space-between;">
          <div>
            <div class="card-title">Plugins</div>
            <div class="card-sub">
              Bundled openclaw extensions. Install runtime npm deps per extension here.
            </div>
          </div>
          <div style="display: flex; gap: 8px;">
            <button
              class="btn"
              ?disabled=${this.installAllRunning || installable.length === 0}
              @click=${() => this.installAll(this.items)}
            >
              ${this.installAllRunning
                ? `Installing… (${this.items.length - installable.length}/${this.items.length})`
                : `Install all (${installable.length})`}
            </button>
            <button
              class="btn"
              ?disabled=${this.loading || !this.client}
              @click=${() => this.refresh()}
            >
              ${this.loading ? "Loading…" : "Refresh"}
            </button>
          </div>
        </div>

        <div class="agent-tabs" style="margin-top: 14px;">
          ${STATUS_TABS.map(
            (tab) => html`
              <button
                class="agent-tab ${this.statusFilter === tab.id ? "active" : ""}"
                @click=${() => (this.statusFilter = tab.id)}
              >
                ${tab.label}<span class="agent-tab-count">${counts[tab.id]}</span>
              </button>
            `,
          )}
        </div>

        <div
          class="filters"
          style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 12px;"
        >
          <label class="field" style="flex: 1; min-width: 180px;">
            <input
              .value=${this.filter}
              @input=${(e: Event) => (this.filter = (e.target as HTMLInputElement).value)}
              placeholder="Filter bundled extensions"
              autocomplete="off"
              name="plugins-filter"
            />
          </label>
          <div class="muted">${filtered.length} shown</div>
        </div>

        ${this.listError
          ? html`<div class="callout danger" style="margin-top: 12px;">${this.listError}</div>`
          : nothing}

        <div style="margin-top: 12px;">
          ${filtered.length === 0
            ? html`<div class="muted" style="padding: 16px 0;">
                ${this.loading ? "Loading…" : "No plugins match the current filter."}
              </div>`
            : this.renderSectionedList(filtered)}
        </div>
      </section>
    `;
  }

  private wizardSubtitle(w: WizardState): string {
    switch (w.kind) {
      case "input":
        return "Step 1 of 3 — Bot token";
      case "verifying":
        return "Step 1 of 3 — Verifying token…";
      case "verified":
      case "waiting":
        return "Step 2 of 3 — Identify your Telegram account";
      case "captured":
      case "persisting":
        return "Step 3 of 3 — Save & confirm";
      case "done":
        return "✓ Configured";
      case "error":
        return "Setup failed";
      default:
        return "";
    }
  }

  private renderWizardBody(w: WizardState) {
    switch (w.kind) {
      case "input":
        return html`
          <div style="font-size: 13px; line-height: 1.5;">
            Get a bot token from
            <a href="https://t.me/BotFather" target="_blank" rel="noopener">@BotFather</a>:
            <code>/newbot</code> → copy the token.
          </div>
          <label class="field">
            <input
              type="text"
              .value=${w.token}
              @input=${(e: Event) => {
                const v = (e.target as HTMLInputElement).value;
                this.wizard = { kind: "input", token: v };
              }}
              placeholder="123456:ABC-..."
              autocomplete="off"
              spellcheck="false"
              style="font-family: var(--font-mono, monospace);"
            />
          </label>
          ${w.error
            ? html`<div class="callout danger" style="margin: 0;">${w.error}</div>`
            : nothing}
          <div class="row" style="justify-content: flex-end; gap: 8px;">
            <button class="btn primary" @click=${() => this.wizardVerify()}>Verify token</button>
          </div>
        `;
      case "verifying":
        return html`<div class="muted">Calling getMe…</div>`;
      case "verified":
        return html`
          <div style="font-size: 13px; line-height: 1.5;">
            Token verified. Bot:
            <strong>@${w.bot.username}</strong> (${w.bot.firstName})
          </div>
          <div style="font-size: 13px; line-height: 1.5;">
            Now open Telegram and <strong>DM @${w.bot.username}</strong> with
            <code>/start</code> (or any message). When you click Continue, talon will wait up to 90
            seconds for your message and capture your sender id.
          </div>
          <div class="row" style="justify-content: flex-end; gap: 8px;">
            <button class="btn primary" @click=${() => this.wizardCapture()}>Continue</button>
          </div>
        `;
      case "waiting": {
        const elapsed = Math.floor((Date.now() - w.startedAt) / 1000);
        return html`
          <div style="font-size: 13px; line-height: 1.5;">
            Waiting for a message from <strong>@${w.bot.username}</strong>… (${elapsed}s)
          </div>
          <div class="muted" style="font-size: 12px;">
            Open Telegram, DM the bot. We'll capture the first message that arrives. Cancel and
            reopen the wizard to retry.
          </div>
        `;
      }
      case "captured":
        return html`
          <div style="font-size: 13px; line-height: 1.5;">
            Captured sender <strong>${w.sender.displayName}</strong> (id
            <code>${w.sender.senderId}</code>).
          </div>
          <div style="font-size: 13px; line-height: 1.5;">
            Click Save to write the channel config and send a confirmation DM. The plugin will spawn
            after the next gateway restart.
          </div>
          <div class="row" style="justify-content: flex-end; gap: 8px;">
            <button class="btn primary" @click=${() => this.wizardPersist()}>Save</button>
          </div>
        `;
      case "persisting":
        return html`<div class="muted">Writing config and sending confirmation DM…</div>`;
      case "done":
        return html`
          <div class="callout" style="margin: 0;">
            ✓ talon is configured for <strong>@${w.bot.username}</strong>. Future replies in your
            chat are routed to the <code>main</code> agent.
          </div>
          ${w.confirmWarning
            ? html`<div class="callout danger" style="margin: 0;">
                Couldn't send the confirmation DM: ${w.confirmWarning}. Config was still written.
              </div>`
            : nothing}
          <div style="font-size: 13px; line-height: 1.5;">${w.restartHint}</div>
          <div class="row" style="justify-content: flex-end; gap: 8px;">
            <button class="btn primary" @click=${this.closeWizard}>Done</button>
          </div>
        `;
      case "error":
        return html`
          <div class="callout danger" style="margin: 0;">
            <strong>${w.step} failed:</strong> ${w.message}
          </div>
          <div class="row" style="justify-content: flex-end; gap: 8px;">
            <button class="btn" @click=${() => this.openTelegramWizard()}>Restart</button>
            <button class="btn" @click=${this.closeWizard}>Close</button>
          </div>
        `;
      default:
        return nothing;
    }
  }

  // renderSectionedList groups the filtered list into Native + Bundled
  // sections with labeled separators between them. Headers only render
  // when the section has rows, so an all-native or all-bundled filter
  // (e.g. the Native tab) doesn't show an empty header for the other.
  private renderSectionedList(filtered: StatusItem[]) {
    const native = filtered.filter(isNativePlugin);
    const bundled = filtered.filter((it) => !isNativePlugin(it));
    return html`
      ${native.length > 0
        ? html`
            ${this.renderSectionHeader(
              "Native plugins",
              `${native.length}`,
              "Statically compiled, no runtime dependencies. Always available.",
            )}
            <div class="list">${native.map((it) => this.renderRow(it))}</div>
          `
        : nothing}
      ${bundled.length > 0
        ? html`
            ${this.renderSectionHeader(
              "Bundled extensions",
              `${bundled.length}`,
              "openclaw extensions on disk. Some require an npm install before they can run.",
            )}
            <div class="list">${bundled.map((it) => this.renderRow(it))}</div>
          `
        : nothing}
    `;
  }

  private renderSectionHeader(label: string, count: string, sub: string) {
    return html`
      <div
        style="display: flex; align-items: baseline; gap: 8px; margin: 16px 0 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border);"
      >
        <div
          style="font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px;"
        >
          ${label}
        </div>
        <div class="muted" style="font-size: 12px;">${count}</div>
        <div
          class="muted"
          style="font-size: 12px; flex: 1; text-align: right; font-weight: normal;"
        >
          ${sub}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "talon-plugins-view": TalonPluginsView;
  }
}

export function renderPlugins(props: { client: GatewayBrowserClient | null }) {
  return html`<talon-plugins-view .client=${props.client}></talon-plugins-view>`;
}

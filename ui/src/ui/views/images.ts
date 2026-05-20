// Images tab — type a description, get a picture back; gallery shows
// recent renders. Backed by talon's images.{generate,fetch,list} RPCs.
//
// Generate flow:
//   1. submit prompt → images.generate → {runId}
//   2. listen for "images" events filtered by runId; show progress
//   3. on state="final", read the first ref and prepend it to the gallery
// Gallery flow:
//   1. on mount, call images.list (newest 50) → render thumbnail grid
//   2. each thumbnail loads via images.fetch with preview="webp;quality=70"
//      so the data URL is ~10× smaller than the full PNG
//   3. clicking a thumbnail loads the full-resolution image into a viewer
//
// Self-contained on purpose so a new tab doesn't have to thread state
// through the giant app-settings/app-render plumbing — the component
// takes the gateway client as a property and owns the rest.

import { LitElement, css, html, nothing } from "lit";
import { customElement, property, state } from "lit/decorators.js";
import type { GatewayBrowserClient, GatewayEventFrame } from "../gateway.ts";

type ImageRef = { filename: string; subfolder?: string; type?: string };

type GalleryItem = ImageRef & {
  promptId?: string;
  thumbDataUrl?: string;
  thumbLoading?: boolean;
  thumbError?: string;
};

type GenerateState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "running"; runId: string; progress?: { value: number; max: number } }
  | { kind: "fetching"; runId: string }
  | { kind: "error"; message: string };

@customElement("talon-images-view")
export class TalonImagesView extends LitElement {
  static override styles = css`
    :host {
      display: block;
      max-width: 960px;
      margin: 0 auto;
      padding: 1.5rem;
      color: var(--text-color, #222);
    }
    h1 {
      margin: 0 0 0.25rem;
      font-size: 1.4rem;
    }
    p.lead {
      margin: 0 0 1.25rem;
      opacity: 0.75;
      font-size: 0.95rem;
    }
    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 0.35rem;
      opacity: 0.85;
    }
    textarea {
      width: 100%;
      min-height: 70px;
      box-sizing: border-box;
      padding: 0.6rem 0.75rem;
      font: inherit;
      border-radius: 8px;
      border: 1px solid var(--border-color, #ccc);
      background: var(--input-bg, #fafafa);
      color: inherit;
      resize: vertical;
    }
    .row {
      display: flex;
      gap: 0.75rem;
      align-items: center;
      margin-top: 0.75rem;
      flex-wrap: wrap;
    }
    .row .progress-host {
      flex: 1 1 200px;
      min-width: 160px;
    }
    button {
      padding: 0.55rem 1rem;
      border-radius: 8px;
      border: 1px solid var(--accent, #4f46e5);
      background: var(--accent, #4f46e5);
      color: white;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    progress {
      width: 100%;
      height: 8px;
    }
    .error {
      margin-top: 0.75rem;
      padding: 0.6rem 0.8rem;
      background: rgba(220, 38, 38, 0.1);
      border-left: 3px solid rgb(220, 38, 38);
      border-radius: 6px;
      font-size: 0.9rem;
      white-space: pre-wrap;
    }
    .latest {
      margin-top: 1.5rem;
    }
    .latest img {
      max-width: 100%;
      border-radius: 12px;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
      display: block;
    }
    .latest .meta {
      margin-top: 0.5rem;
      font-size: 0.8rem;
      opacity: 0.65;
    }
    .gallery {
      margin-top: 1.75rem;
    }
    .gallery h2 {
      margin: 0 0 0.6rem;
      font-size: 1rem;
      opacity: 0.85;
    }
    .gallery-empty {
      font-size: 0.9rem;
      opacity: 0.55;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 0.6rem;
    }
    .thumb {
      position: relative;
      aspect-ratio: 1 / 1;
      overflow: hidden;
      border-radius: 8px;
      background: var(--input-bg, #f1f1f1);
      cursor: pointer;
      border: 1px solid transparent;
    }
    .thumb:hover {
      border-color: var(--accent, #4f46e5);
    }
    .thumb:hover .thumb-delete {
      opacity: 1;
    }
    .thumb img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .thumb-delete {
      position: absolute;
      top: 4px;
      right: 4px;
      width: 24px;
      height: 24px;
      padding: 0;
      border-radius: 50%;
      border: 0;
      background: rgba(0, 0, 0, 0.65);
      color: white;
      font: inherit;
      font-size: 0.85rem;
      line-height: 24px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 80ms ease-in;
    }
    .thumb-delete:hover {
      background: rgb(220, 38, 38);
    }
    .viewer-controls {
      position: absolute;
      top: 1rem;
      right: 1rem;
      display: flex;
      gap: 0.5rem;
    }
    .viewer-controls button {
      background: rgba(0, 0, 0, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.3);
      color: white;
      border-radius: 6px;
      padding: 0.4rem 0.7rem;
      cursor: pointer;
      font: inherit;
      font-size: 0.85rem;
    }
    .viewer-controls .danger:hover {
      background: rgb(220, 38, 38);
    }
    .thumb-fallback {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      opacity: 0.55;
      padding: 0.5rem;
      text-align: center;
    }
    .viewer-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.85);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      z-index: 9999;
    }
    .viewer-backdrop img {
      max-width: 100%;
      max-height: 100%;
      border-radius: 8px;
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
    }
    .viewer-meta {
      position: absolute;
      bottom: 0.75rem;
      left: 50%;
      transform: translateX(-50%);
      color: white;
      font-size: 0.85rem;
      background: rgba(0, 0, 0, 0.6);
      padding: 0.35rem 0.7rem;
      border-radius: 6px;
    }
  `;

  @property({ attribute: false })
  client: GatewayBrowserClient | null = null;

  @property({ attribute: false })
  sessionKey = "agent:main:main";

  @state() private prompt = "";
  @state() private negativePrompt = "";
  @state() private genState: GenerateState = { kind: "idle" };

  // Workflow / model selector. Loaded once via images.workflows.list
  // when the component first sees a client. workflowId == "" means
  // "use the user's config-driven default workflow"; non-empty ids
  // pick a builtin (e.g. dixar-character).
  @state() private workflows: Array<{
    id: string;
    label: string;
    description?: string;
    source: string;
  }> = [];
  @state() private workflowsLoaded = false;
  @state() private selectedWorkflowId = "";
  @state() private gallery: GalleryItem[] = [];
  @state() private galleryLoaded = false;
  @state() private galleryError: string | null = null;
  @state() private viewer: {
    ref: ImageRef;
    dataUrl: string | null;
    loading: boolean;
    error?: string;
  } | null = null;
  // Most recently generated image. Replaces on each successful run.
  // Distinct from the (currently disabled) gallery — this is the
  // single inline preview the user wanted back after we'd dropped
  // the giant inline display in favor of the prepend-to-gallery
  // flow that's no longer rendering.
  @state() private latest: {
    ref: ImageRef;
    dataUrl: string | null;
    loading: boolean;
    error?: string;
  } | null = null;

  private activeRunId: string | null = null;
  private detachListener: (() => void) | null = null;

  override disconnectedCallback() {
    super.disconnectedCallback();
    this.detachListener?.();
    this.detachListener = null;
  }

  // Gallery auto-load is intentionally disabled — see render(). The
  // loadGallery / loadThumb / deleteImage methods are kept around so
  // re-enabling later is a single edit (add the render call back).
  override updated(_changed: Map<string, unknown>) {
    if (this.client && !this.workflowsLoaded) {
      void this.loadWorkflows();
    }
  }

  private async loadWorkflows() {
    if (!this.client || this.workflowsLoaded) return;
    this.workflowsLoaded = true;
    try {
      const res = await this.client.request<{
        workflows?: Array<{ id: string; label: string; description?: string; source: string }>;
      }>("images.workflows.list", {});
      const list = Array.isArray(res?.workflows) ? res!.workflows! : [];
      this.workflows = list;
      // Pick a sensible default: prefer the user's configured row
      // (sentinel id ""), otherwise the first builtin so the user
      // gets a working preset out of the box.
      if (this.selectedWorkflowId === "") {
        const userRow = list.find((w) => w.source === "user");
        if (userRow) {
          this.selectedWorkflowId = "";
        } else if (list.length > 0) {
          this.selectedWorkflowId = list[0].id;
        }
      }
    } catch (err) {
      // Non-fatal — the dropdown just stays empty and the user falls
      // back to the existing default-workflow path on submit.
      console.warn("images.workflows.list failed:", err);
      this.workflows = [];
    }
  }

  private get isBusy(): boolean {
    return (
      this.genState.kind === "submitting" ||
      this.genState.kind === "running" ||
      this.genState.kind === "fetching"
    );
  }

  async loadGallery() {
    if (!this.client) return;
    this.galleryError = null;
    try {
      const res = await this.client.request<{ images: GalleryItem[] }>("images.list", {
        limit: 50,
      });
      // ComfyUI's history is unordered; treat the response as
      // newest-first since that matches current ComfyUI versions and
      // works correctly even if ordering is off (it's a flat grid).
      this.gallery = (res?.images ?? []).map((item) => ({ ...item }));
      this.galleryLoaded = true;
      // Kick off thumbnail loads in the background; updates re-render
      // each tile as its data URL arrives.
      for (const item of this.gallery) {
        void this.loadThumb(item);
      }
    } catch (err) {
      this.galleryError = String(err);
    }
  }

  private async loadThumb(item: GalleryItem) {
    if (!this.client || item.thumbDataUrl || item.thumbLoading) return;
    item.thumbLoading = true;
    this.gallery = [...this.gallery]; // trigger re-render of the loading state
    try {
      const res = await this.client.request<{ dataUrl: string }>("images.fetch", {
        filename: item.filename,
        subfolder: item.subfolder ?? "",
        type: item.type ?? "output",
        preview: "webp;quality=70",
      });
      item.thumbDataUrl = res?.dataUrl ?? "";
      item.thumbLoading = false;
      this.gallery = [...this.gallery];
    } catch (err) {
      item.thumbError = String(err);
      item.thumbLoading = false;
      this.gallery = [...this.gallery];
    }
  }

  private async submit() {
    if (!this.client) {
      this.genState = { kind: "error", message: "gateway client not connected" };
      return;
    }
    const prompt = this.prompt.trim();
    if (!prompt) {
      this.genState = { kind: "error", message: "prompt is required" };
      return;
    }

    this.genState = { kind: "submitting" };
    this.attachListener();

    try {
      const res = await this.client.request<{ runId: string }>("images.generate", {
        sessionKey: this.sessionKey,
        prompt,
        negativePrompt: this.negativePrompt.trim() || undefined,
        workflowId: this.selectedWorkflowId || undefined,
      });
      if (!res?.runId) {
        throw new Error("server returned no runId");
      }
      this.activeRunId = res.runId;
      this.genState = { kind: "running", runId: res.runId };
    } catch (err) {
      this.genState = { kind: "error", message: String(err) };
    }
  }

  private attachListener() {
    if (this.detachListener || !this.client) return;
    this.detachListener = this.client.addEventListener((evt: GatewayEventFrame) => {
      if (evt.event !== "images") return;
      this.handleImagesEvent(evt.payload);
    });
  }

  private handleImagesEvent(payload: unknown) {
    if (!payload || typeof payload !== "object") return;
    const p = payload as {
      runId?: string;
      state?: string;
      data?: Record<string, unknown> | null;
    };
    if (!p.runId || p.runId !== this.activeRunId) return;

    switch (p.state) {
      case "queued":
      case "running": {
        // Drop the node id — surface a single unified progress only.
        const prev = this.genState.kind === "running" ? this.genState.progress : undefined;
        this.genState = { kind: "running", runId: p.runId, progress: prev };
        return;
      }
      case "progress": {
        const value = Number(p.data?.value ?? 0);
        const max = Number(p.data?.max ?? 0);
        if (this.genState.kind === "running") {
          this.genState = { ...this.genState, progress: { value, max } };
        }
        return;
      }
      case "error": {
        const msg = String(p.data?.errorMessage ?? "image generation failed");
        this.genState = { kind: "error", message: msg };
        this.activeRunId = null;
        return;
      }
      case "final": {
        const images = (p.data?.images as ImageRef[] | undefined) ?? [];
        if (images.length === 0) {
          this.genState = { kind: "error", message: "run finished with no images" };
          this.activeRunId = null;
          return;
        }
        const ref = images[0];
        this.genState = { kind: "fetching", runId: p.runId };
        this.latest = { ref, dataUrl: null, loading: true };
        void this.fetchLatest(ref);
        this.activeRunId = null;
        return;
      }
    }
  }

  // fetchLatest pulls the bytes for the most recent generation as a
  // data URL and renders it inline. Mirrors openViewer but writes to
  // `this.latest` and isn't dismissable — the latest preview stays
  // until the next generation lands.
  private async fetchLatest(ref: ImageRef) {
    if (!this.client) return;
    try {
      const res = await this.client.request<{ dataUrl: string }>("images.fetch", {
        filename: ref.filename,
        subfolder: ref.subfolder ?? "",
        type: ref.type ?? "output",
      });
      if (!this.latest || this.latest.ref.filename !== ref.filename) return;
      this.latest = { ref, dataUrl: res?.dataUrl ?? "", loading: false };
      this.genState = { kind: "idle" };
    } catch (err) {
      if (!this.latest || this.latest.ref.filename !== ref.filename) return;
      this.latest = { ref, dataUrl: null, loading: false, error: String(err) };
      this.genState = { kind: "error", message: `fetch failed: ${String(err)}` };
    }
  }

  private async openViewer(item: GalleryItem) {
    if (!this.client) return;
    const ref: ImageRef = {
      filename: item.filename,
      subfolder: item.subfolder,
      type: item.type,
    };
    this.viewer = { ref, dataUrl: null, loading: true };
    try {
      const res = await this.client.request<{ dataUrl: string }>("images.fetch", {
        filename: ref.filename,
        subfolder: ref.subfolder ?? "",
        type: ref.type ?? "output",
      });
      if (!this.viewer || this.viewer.ref.filename !== ref.filename) return;
      this.viewer = { ref, dataUrl: res?.dataUrl ?? "", loading: false };
    } catch (err) {
      if (!this.viewer || this.viewer.ref.filename !== ref.filename) return;
      this.viewer = { ref, dataUrl: null, loading: false, error: String(err) };
    }
  }

  private closeViewer() {
    this.viewer = null;
  }

  // deleteImage removes an image from talon's persistent index. The
  // file on ComfyUI's disk is left in place — talon doesn't have
  // filesystem access to the LAN host's output dir. The image
  // disappears from the gallery; if it's still in ComfyUI's history
  // it'll reappear on the next load (until ComfyUI restarts), which
  // is mildly surprising but the alternative (silently failing on
  // history-only entries) is worse.
  private async deleteImage(item: GalleryItem, ev?: Event) {
    ev?.stopPropagation();
    if (!this.client) return;
    // Optimistic remove from UI; revert on failure.
    const before = this.gallery;
    this.gallery = this.gallery.filter(
      (g) => !(g.filename === item.filename && (g.subfolder ?? "") === (item.subfolder ?? "")),
    );
    try {
      await this.client.request("images.delete", {
        filename: item.filename,
        subfolder: item.subfolder ?? "",
      });
      // Close the viewer if it was showing the deleted image.
      if (this.viewer && this.viewer.ref.filename === item.filename) {
        this.viewer = null;
      }
    } catch (err) {
      this.galleryError = `delete failed: ${String(err)}`;
      this.gallery = before;
    }
  }

  private renderProgress() {
    switch (this.genState.kind) {
      case "submitting":
        return html`<progress></progress>`;
      case "running": {
        const progress = this.genState.progress;
        return progress
          ? html`<progress max=${progress.max} value=${progress.value}></progress>`
          : html`<progress></progress>`;
      }
      case "fetching":
        return html`<progress></progress>`;
      default:
        return nothing;
    }
  }

  renderGallery() {
    if (!this.galleryLoaded && !this.galleryError) {
      return html`<p class="gallery-empty">loading…</p>`;
    }
    if (this.galleryError) {
      return html`<div class="error">gallery: ${this.galleryError}</div>`;
    }
    if (this.gallery.length === 0) {
      return html`<p class="gallery-empty">No images yet — generate one above.</p>`;
    }
    return html`
      <div class="grid">
        ${this.gallery.map(
          (item) => html`
            <div class="thumb" @click=${() => this.openViewer(item)} title=${item.filename}>
              ${item.thumbDataUrl
                ? html`<img src=${item.thumbDataUrl} alt=${item.filename} loading="lazy" />`
                : item.thumbError
                  ? html`<div class="thumb-fallback">load failed</div>`
                  : html`<div class="thumb-fallback">…</div>`}
              <button
                class="thumb-delete"
                title="Remove from gallery"
                @click=${(ev: Event) => this.deleteImage(item, ev)}
              >
                ✕
              </button>
            </div>
          `,
        )}
      </div>
    `;
  }

  renderViewer() {
    if (!this.viewer) return nothing;
    const { ref, dataUrl, loading, error } = this.viewer;
    return html`
      <div class="viewer-backdrop" @click=${() => this.closeViewer()}>
        ${loading
          ? html`<progress style="width: 200px"></progress>`
          : error
            ? html`<div class="error">${error}</div>`
            : dataUrl
              ? html`<img src=${dataUrl} alt=${ref.filename} />`
              : nothing}
        <div class="viewer-controls" @click=${(e: Event) => e.stopPropagation()}>
          <button
            class="danger"
            @click=${() =>
              this.deleteImage({
                filename: ref.filename,
                subfolder: ref.subfolder,
                type: ref.type,
              })}
          >
            Delete
          </button>
          <button @click=${() => this.closeViewer()}>Close</button>
        </div>
        <div class="viewer-meta">${ref.filename}</div>
      </div>
    `;
  }

  override render() {
    const disabled = this.isBusy || !this.client;
    return html`
      <h1>Images</h1>
      <p class="lead">
        Generate an image with ComfyUI. Pick a shipped workflow below or configure your own under
        <code>images.providers.comfyui</code>.
      </p>

      ${this.renderWorkflowSelector()}

      <label for="prompt">Prompt</label>
      <textarea
        id="prompt"
        .value=${this.prompt}
        @input=${(e: Event) => (this.prompt = (e.target as HTMLTextAreaElement).value)}
        placeholder="a cute lobster wearing a tiny crown, watercolor"
      ></textarea>

      <label for="negative" style="margin-top: 0.85rem;">Negative prompt (optional)</label>
      <textarea
        id="negative"
        .value=${this.negativePrompt}
        @input=${(e: Event) => (this.negativePrompt = (e.target as HTMLTextAreaElement).value)}
      ></textarea>

      <div class="row">
        <button @click=${() => this.submit()} ?disabled=${disabled}>
          ${this.isBusy ? "generating…" : "Generate"}
        </button>
        <div class="progress-host">${this.renderProgress()}</div>
      </div>

      ${this.genState.kind === "error"
        ? html`<div class="error">${this.genState.message}</div>`
        : nothing}
      ${this.renderLatest()}
    `;
  }

  private renderWorkflowSelector() {
    if (this.workflows.length === 0) return nothing;
    const selected = this.workflows.find((w) => w.id === this.selectedWorkflowId);
    return html`
      <label for="workflow" style="margin-top: 0.25rem;">Workflow / model</label>
      <select
        id="workflow"
        .value=${this.selectedWorkflowId}
        ?disabled=${this.isBusy || !this.client}
        @change=${(e: Event) => {
          this.selectedWorkflowId = (e.target as HTMLSelectElement).value;
        }}
        style="width: 100%; padding: 0.55rem 0.7rem; border-radius: 8px; border: 1px solid var(--border-color, #ccc); background: var(--input-bg, #fafafa); color: inherit; font: inherit;"
      >
        ${this.workflows.map(
          (w) => html`<option value=${w.id} ?selected=${w.id === this.selectedWorkflowId}>
            ${w.label}${w.source === "builtin" ? " · shipped" : ""}
          </option>`,
        )}
      </select>
      ${selected?.description
        ? html`<div style="margin-top: 0.35rem; font-size: 0.8rem; opacity: 0.7;">
            ${selected.description}
          </div>`
        : nothing}
    `;
  }

  private renderLatest() {
    if (!this.latest) return nothing;
    const { ref, dataUrl, loading, error } = this.latest;
    return html`
      <div class="latest">
        ${loading
          ? html`<progress style="width: 200px"></progress>`
          : error
            ? html`<div class="error">${error}</div>`
            : dataUrl
              ? html`<img src=${dataUrl} alt=${this.prompt || ref.filename} />`
              : nothing}
        <div class="meta">${ref.filename}</div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "talon-images-view": TalonImagesView;
  }
}

// renderImages is the entry point app-render.ts calls. Returns a Lit
// template that mounts the custom element and passes through the
// connected gateway client.
export function renderImages(props: { client: GatewayBrowserClient | null; sessionKey: string }) {
  return html`<talon-images-view
    .client=${props.client}
    .sessionKey=${props.sessionKey}
  ></talon-images-view>`;
}

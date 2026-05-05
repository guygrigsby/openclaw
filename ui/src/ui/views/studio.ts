// Studio tab — native Lit view that drives talon's bundled ComfyUI
// workflows with every knob exposed (KSampler steps/cfg/sampler/
// scheduler, LoRA strengths, IPAdapter weights, plus prompt/seed).
//
// Backend: talon's existing images.{workflows.list,workflows.get,
// generate} RPCs. workflows.get returns the parsed workflow graph so
// we can introspect node types and current input values to render
// per-workflow controls; generate accepts a nodeOverrides map that
// applies arbitrary input patches before submission.
//
// Naming history: started as "Grado" (Gradio playground sidecar
// running at :7860), got promoted to a native Lit view and kept the
// name briefly. Renamed to "Studio" once the Gradio etymology no
// longer applied — the tab is the full-knobs studio for ComfyUI
// workflows next to the simpler Images tab.

import { LitElement, css, html, nothing } from "lit";
import { customElement, property, state } from "lit/decorators.js";
import type { GatewayBrowserClient, GatewayEventFrame } from "../gateway.ts";

// --- types ---------------------------------------------------------------

type ImageRef = { filename: string; subfolder?: string; type?: string };

type WorkflowEntry = {
  id: string;
  label: string;
  description?: string;
  source: string;
};

type WorkflowGraph = Record<
  string,
  { class_type?: string; inputs?: Record<string, unknown>; _meta?: { title?: string } }
>;

type LoraSlot = { nodeId: string; label: string; value: number };
type IpaSlot = { nodeId: string; label: string; value: number };

// Params extracted from a workflow graph — used by both the Studio
// dropdown-change path (apply defaults from the just-picked workflow)
// and Replay (apply values pulled from a generated image's embedded
// PNG metadata).
type ExtractedParams = {
  positivePrompt: string | null;
  negativePrompt: string | null;
  seed: number | null;
  steps: number | null;
  cfg: number | null;
  sampler: string | null;
  scheduler: string | null;
  loras: LoraSlot[];
  ipas: IpaSlot[];
};

// Embedded PNG metadata: ComfyUI writes `prompt` (API graph) and
// optionally `workflow` (editor graph) as tEXt chunks. raw holds any
// other tEXt entries we didn't parse — useful for debugging.
type ParsedMeta = {
  prompt: WorkflowGraph | null;
  workflow: WorkflowGraph | null;
  raw: Record<string, string>;
};
type KSamplerKnob = {
  nodeId: string;
  title: string;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  seed: number;
};

type GenState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "running"; runId: string; progress?: { value: number; max: number } }
  | { kind: "fetching"; runId: string }
  | { kind: "error"; message: string };

// --- constants -----------------------------------------------------------

// Sampler / scheduler lists are the union of what talon's bundled
// workflows reference. Both fields accept arbitrary strings so a
// custom sampler in the user's ComfyUI install still works — just
// type it in.
const SAMPLERS = [
  "euler",
  "euler_ancestral",
  "dpmpp_2m",
  "dpmpp_2m_sde",
  "dpmpp_3m_sde",
  "dpm_2",
  "deis",
  "uni_pc",
  "lcm",
];
const SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"];

// LoRA strengths beyond ±2 are almost always wrong in practice;
// IPAdapter weights typically cap at 2. Picking ranges the user can
// drive without fat-fingering past sane values.
const LORA_RANGE: [number, number] = [-2, 2];
const IPA_RANGE: [number, number] = [0, 2];

// --- helpers (module scope) ---------------------------------------------

// extractParamsFromGraph walks a ComfyUI API-format graph and pulls
// every knob the Studio surface exposes: user prompts, KSampler
// settings, LoRA strengths, IPAdapter weights. Used by both the
// dropdown-change introspection (apply template defaults) and
// Replay (apply values pulled from a generated image's PNG).
//
// Conventions match the bundled workflows:
//   - 'Positive (user)'-titled CLIPTextEncode → positive prompt
//   - 'Negative (user)'-titled CLIPTextEncode → negative prompt
//   - First KSampler/KSamplerAdvanced → seed/steps/cfg/sampler/scheduler
//   - Every LoraLoader → one slot
//   - Every IPAdapter* (except IPAdapterUnifiedLoader) → one slot
function extractParamsFromGraph(graph: WorkflowGraph): ExtractedParams {
  const out: ExtractedParams = {
    positivePrompt: null,
    negativePrompt: null,
    seed: null,
    steps: null,
    cfg: null,
    sampler: null,
    scheduler: null,
    loras: [],
    ipas: [],
  };
  let seenKsampler = false;
  // Sort keys for stable slot order (workflows reuse small int ids).
  const ids = Object.keys(graph).sort((a, b) => a.length - b.length || a.localeCompare(b));
  for (const nid of ids) {
    const node = graph[nid];
    const cls = node?.class_type ?? "";
    const title = node?._meta?.title ?? "";
    const titleLower = title.toLowerCase();
    const inputs = node?.inputs ?? {};
    if (cls === "CLIPTextEncode" && titleLower.includes("user")) {
      const text = typeof inputs["text"] === "string" ? (inputs["text"] as string) : null;
      if (titleLower.includes("positive") && text != null) out.positivePrompt = text;
      else if (titleLower.includes("negative") && text != null) out.negativePrompt = text;
    } else if (cls === "LoraLoader") {
      const loraName = String(inputs["lora_name"] ?? "");
      const v = Number(inputs["strength_model"] ?? 0);
      out.loras.push({
        nodeId: nid,
        label: title || loraName.replace(/\.safetensors$/, "") || "LoRA",
        value: Number.isFinite(v) ? v : 0,
      });
    } else if (cls.startsWith("IPAdapter") && cls !== "IPAdapterUnifiedLoader") {
      const w = inputs["weight"];
      if (typeof w === "number") {
        out.ipas.push({ nodeId: nid, label: title || cls, value: w });
      }
    } else if ((cls === "KSampler" || cls === "KSamplerAdvanced") && !seenKsampler) {
      seenKsampler = true;
      const seedKey = cls === "KSamplerAdvanced" ? "noise_seed" : "seed";
      const seedRaw = inputs[seedKey];
      out.seed = typeof seedRaw === "number" ? seedRaw : Number(seedRaw ?? NaN);
      if (!Number.isFinite(out.seed)) out.seed = null;
      const steps = Number(inputs["steps"]);
      out.steps = Number.isFinite(steps) ? steps : null;
      const cfg = Number(inputs["cfg"]);
      out.cfg = Number.isFinite(cfg) ? cfg : null;
      out.sampler = inputs["sampler_name"] != null ? String(inputs["sampler_name"]) : null;
      out.scheduler = inputs["scheduler"] != null ? String(inputs["scheduler"]) : null;
    }
  }
  return out;
}

// graphSignature reduces a workflow graph to a comparable shape for
// matching against shipped workflow templates. Two workflows match
// if they have the same set of (node id → class_type) pairs.
// Whitespace-cheap and fast; resolves the embedded prompt back to a
// known workflowId so Replay knows which template to select.
function graphSignature(graph: WorkflowGraph): string {
  return Object.keys(graph)
    .sort()
    .map((nid) => `${nid}:${graph[nid]?.class_type ?? ""}`)
    .join("|");
}

// parsePngTextChunks decodes a base64 data URL into PNG tEXt chunks.
// Returns a Record<keyword, text>. Handles only tEXt (Latin-1)
// because that's what ComfyUI emits; iTXt support is a follow-up if
// any source ever needs it.
function parsePngTextChunks(dataUrl: string): Record<string, string> {
  const out: Record<string, string> = {};
  const comma = dataUrl.indexOf(",");
  if (comma < 0) return out;
  let bytes: Uint8Array;
  try {
    const bin = atob(dataUrl.slice(comma + 1));
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  } catch {
    return out;
  }
  // PNG signature: 89 50 4E 47 0D 0A 1A 0A
  if (
    bytes.length < 8 ||
    bytes[0] !== 0x89 ||
    bytes[1] !== 0x50 ||
    bytes[2] !== 0x4e ||
    bytes[3] !== 0x47
  ) {
    return out;
  }
  let offset = 8;
  const decoder = new TextDecoder("latin1");
  while (offset + 8 <= bytes.length) {
    const length =
      (bytes[offset] << 24) |
      (bytes[offset + 1] << 16) |
      (bytes[offset + 2] << 8) |
      bytes[offset + 3];
    const type = decoder.decode(bytes.subarray(offset + 4, offset + 8));
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > bytes.length) break;
    if (type === "tEXt") {
      const chunk = bytes.subarray(dataStart, dataEnd);
      const nul = chunk.indexOf(0);
      if (nul > 0) {
        const keyword = decoder.decode(chunk.subarray(0, nul));
        const text = decoder.decode(chunk.subarray(nul + 1));
        out[keyword] = text;
      }
    }
    if (type === "IEND") break;
    offset = dataEnd + 4; // skip CRC
  }
  return out;
}

// parseEmbeddedMetadata pulls the JSON-shaped tEXt chunks ComfyUI
// writes (`prompt`, `workflow`) and returns them as parsed graphs.
// Failed JSON parses are treated as missing — callers see null and
// render "no metadata."
function parseEmbeddedMetadata(dataUrl: string): ParsedMeta {
  const raw = parsePngTextChunks(dataUrl);
  const tryParse = (key: string): WorkflowGraph | null => {
    const s = raw[key];
    if (!s) return null;
    try {
      const v = JSON.parse(s);
      return v && typeof v === "object" ? (v as WorkflowGraph) : null;
    } catch {
      return null;
    }
  };
  return { prompt: tryParse("prompt"), workflow: tryParse("workflow"), raw };
}

// --- component -----------------------------------------------------------

@customElement("studio-view")
export class StudioView extends LitElement {
  @property({ attribute: false }) client: GatewayBrowserClient | null = null;
  @property({ type: String }) sessionKey = "agent:main:main";

  // Workflows + selection
  @state() private workflows: WorkflowEntry[] = [];
  @state() private workflowsLoading = false;
  @state() private workflowsError: string | null = null;
  @state() private selectedWorkflowId = "";
  @state() private currentGraph: WorkflowGraph | null = null;

  // Prompt + neg + seed
  @state() private prompt = "";
  @state() private negativePrompt = "";
  @state() private seed = 42;
  @state() private randomizeSeed = true;

  // Common KSampler knobs (apply to every KSampler in the workflow)
  @state() private steps = 30;
  @state() private cfg = 7.5;
  @state() private sampler = "dpmpp_2m";
  @state() private scheduler = "karras";

  // Batch
  @state() private numImages = 4;

  // Dynamic LoRA + IPAdapter slot data (recomputed on workflow change)
  @state() private loraSlots: LoraSlot[] = [];
  @state() private ipaSlots: IpaSlot[] = [];

  // Generation state + result gallery
  @state() private genState: GenState = { kind: "idle" };
  @state() private gallery: { ref: ImageRef; dataUrl: string | null }[] = [];
  @state() private activeRunIds = new Set<string>();

  // Lightbox: clicking a thumbnail opens the full-resolution image
  // in a modal. dataUrl starts null while images.fetch is in flight.
  // Mirrors the viewer pattern in images.ts so behavior is consistent
  // across the two image tabs. metadata is populated once the dataUrl
  // arrives by parsing the PNG's tEXt chunks (ComfyUI embeds prompt +
  // workflow there); null when the image lacks embedded metadata.
  @state() private viewer: {
    ref: ImageRef;
    dataUrl: string | null;
    loading: boolean;
    error?: string;
    metadata: ParsedMeta | null;
  } | null = null;

  // Bulk-download progress. {done, total} while a download-all run is
  // in flight; null otherwise. Lets the button label show "3/12…" and
  // disables itself until the run finishes.
  @state() private downloading: { done: number; total: number } | null = null;

  private detachListener: (() => void) | null = null;

  // --- lifecycle ---------------------------------------------------------
  //
  // connectedCallback / disconnectedCallback are defined later (next
  // to the openViewer helpers) so the keyboard listener for ESC-to-
  // close-lightbox lives next to the viewer state it operates on.

  override updated(changed: Map<string, unknown>) {
    // Re-attach the event listener whenever the gateway client
    // identity changes (reconnect, session swap, etc.).
    if (changed.has("client") && this.client) {
      this.attachListener();
    }
  }

  // --- RPC: list + get workflows ----------------------------------------

  private async loadWorkflows() {
    if (!this.client) return;
    this.workflowsLoading = true;
    this.workflowsError = null;
    try {
      const res = await this.client.request<{ workflows: WorkflowEntry[] }>(
        "images.workflows.list",
        {},
      );
      this.workflows = res?.workflows ?? [];
      // Default to vyx when present (the user's headline workflow);
      // otherwise pick the first builtin so the UI has *something*
      // selected on first paint.
      const vyx = this.workflows.find((w) => w.id === "vyx");
      const initial =
        vyx ?? this.workflows.find((w) => w.source === "builtin") ?? this.workflows[0];
      if (initial) {
        this.selectedWorkflowId = initial.id;
        await this.loadWorkflowGraph(initial.id);
      }
    } catch (err) {
      this.workflowsError = String(err);
    } finally {
      this.workflowsLoading = false;
    }
  }

  private async loadWorkflowGraph(id: string) {
    if (!this.client) return;
    try {
      const res = await this.client.request<{ id: string; graph: WorkflowGraph }>(
        "images.workflows.get",
        { id },
      );
      this.currentGraph = res?.graph ?? null;
      this.applyGraphIntrospection(this.currentGraph);
    } catch (err) {
      this.workflowsError = `failed to load workflow: ${err}`;
      this.currentGraph = null;
      this.loraSlots = [];
      this.ipaSlots = [];
    }
  }

  private applyGraphIntrospection(graph: WorkflowGraph | null) {
    if (!graph) {
      this.loraSlots = [];
      this.ipaSlots = [];
      return;
    }
    const p = extractParamsFromGraph(graph);
    this.loraSlots = p.loras;
    this.ipaSlots = p.ipas;
    // Seed common-knob defaults from the workflow so the visible
    // values match what the template was tuned for. Don't override
    // the prompts here — those come from the user, not the template.
    if (p.steps != null) this.steps = p.steps;
    if (p.cfg != null) this.cfg = p.cfg;
    if (p.sampler) this.sampler = p.sampler;
    if (p.scheduler) this.scheduler = p.scheduler;
  }

  // --- generate flow -----------------------------------------------------

  private buildNodeOverrides(seedForRun: number): Record<string, Record<string, unknown>> {
    const overrides: Record<string, Record<string, unknown>> = {};
    if (!this.currentGraph) return overrides;

    // KSampler patches: apply common knobs to EVERY KSampler in the
    // workflow. Seed flows through the existing seed/extraSeed path
    // on the server, but it's cheap to also write it here so this
    // map captures the full intent.
    for (const [nid, node] of Object.entries(this.currentGraph)) {
      const cls = node?.class_type ?? "";
      if (cls === "KSampler" || cls === "KSamplerAdvanced") {
        const seedKey = cls === "KSamplerAdvanced" ? "noise_seed" : "seed";
        overrides[nid] = {
          steps: this.steps,
          cfg: this.cfg,
          sampler_name: this.sampler,
          scheduler: this.scheduler,
          [seedKey]: seedForRun,
        };
      }
    }
    // LoRA strengths: lockstep model+clip per slider value.
    for (const slot of this.loraSlots) {
      overrides[slot.nodeId] = {
        ...(overrides[slot.nodeId] ?? {}),
        strength_model: slot.value,
        strength_clip: slot.value,
      };
    }
    // IPAdapter weights.
    for (const slot of this.ipaSlots) {
      overrides[slot.nodeId] = {
        ...(overrides[slot.nodeId] ?? {}),
        weight: slot.value,
      };
    }
    return overrides;
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

    // Submit the batch. Each call gets its own seed (random or
    // sequential); the server creates a runId per call and we listen
    // for events on all of them simultaneously.
    const count = Math.max(1, Math.floor(this.numImages));
    for (let i = 0; i < count; i++) {
      const s = this.randomizeSeed
        ? Math.floor(Math.random() * 0x7fffffff)
        : Math.floor(this.seed) + i;
      const overrides = this.buildNodeOverrides(s);
      try {
        const res = await this.client.request<{ runId: string }>("images.generate", {
          sessionKey: this.sessionKey,
          prompt,
          negativePrompt: this.negativePrompt.trim() || undefined,
          seed: s,
          workflowId: this.selectedWorkflowId || undefined,
          nodeOverrides: overrides,
        });
        if (res?.runId) {
          this.activeRunIds.add(res.runId);
          this.genState = { kind: "running", runId: res.runId };
        }
      } catch (err) {
        this.genState = { kind: "error", message: String(err) };
        return;
      }
    }
  }

  private attachListener() {
    if (this.detachListener || !this.client) return;
    this.detachListener = this.client.addEventListener((evt: GatewayEventFrame) => {
      if (evt.event !== "images") return;
      this.handleImagesEvent(evt.payload);
    });
  }

  private async handleImagesEvent(payload: unknown) {
    if (!payload || typeof payload !== "object") return;
    const p = payload as {
      runId?: string;
      state?: string;
      data?: Record<string, unknown> | null;
    };
    if (!p.runId || !this.activeRunIds.has(p.runId)) return;

    switch (p.state) {
      case "queued":
      case "running":
        this.genState = { kind: "running", runId: p.runId };
        return;
      case "progress": {
        const value = Number(p.data?.value ?? 0);
        const max = Number(p.data?.max ?? 0);
        this.genState = { kind: "running", runId: p.runId, progress: { value, max } };
        return;
      }
      case "error":
        this.genState = {
          kind: "error",
          message: String(p.data?.errorMessage ?? "image generation failed"),
        };
        this.activeRunIds.delete(p.runId);
        return;
      case "final": {
        const images = (p.data?.images as ImageRef[] | undefined) ?? [];
        for (const ref of images) {
          this.gallery = [{ ref, dataUrl: null }, ...this.gallery].slice(0, 50);
          void this.fetchThumb(ref);
        }
        this.activeRunIds.delete(p.runId);
        if (this.activeRunIds.size === 0) {
          this.genState = { kind: "idle" };
        }
        return;
      }
    }
  }

  // downloadAll fetches the full-resolution image for every gallery
  // item and triggers a browser download per file. No zip — keeps
  // the implementation dependency-free; modern browsers batch the
  // anchor clicks behind a single "allow multiple downloads" prompt.
  // Sequential with a small inter-file pause so a 50-image batch
  // doesn't slam the gateway with parallel images.fetch calls.
  private async downloadAll() {
    if (!this.client || this.gallery.length === 0 || this.downloading) return;
    const items = [...this.gallery];
    this.downloading = { done: 0, total: items.length };
    for (let i = 0; i < items.length; i++) {
      const ref = items[i].ref;
      try {
        const res = await this.client.request<{ dataUrl: string }>("images.fetch", {
          filename: ref.filename,
          subfolder: ref.subfolder ?? "",
          type: ref.type ?? "output",
        });
        if (res?.dataUrl) {
          this.triggerDownload(res.dataUrl, ref.filename);
        }
      } catch (err) {
        // Don't abort the whole batch on a single failure — log and
        // keep going so the user gets whatever's still fetchable.
        console.warn("studio: download failed", ref.filename, err);
      }
      this.downloading = { done: i + 1, total: items.length };
      // Stagger so the browser's download queue doesn't drop entries.
      // ~80ms is enough on Chromium/Firefox without dragging out big
      // batches.
      await new Promise((resolve) => setTimeout(resolve, 80));
    }
    this.downloading = null;
  }

  private triggerDownload(dataUrl: string, filename: string) {
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = filename;
    // Append → click → remove keeps the DOM clean and works in
    // browsers that ignore detached-anchor clicks.
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  private async openViewer(item: { ref: ImageRef; dataUrl: string | null }) {
    if (!this.client) return;
    // Optimistic display: if we already have a thumb data URL, show
    // it immediately while the full-res fetch runs in the background.
    this.viewer = { ref: item.ref, dataUrl: item.dataUrl, loading: true, metadata: null };
    try {
      const res = await this.client.request<{ dataUrl: string }>("images.fetch", {
        filename: item.ref.filename,
        subfolder: item.ref.subfolder ?? "",
        type: item.ref.type ?? "output",
      });
      // Skip the swap if the user already closed or opened a different
      // image by the time the fetch returned.
      if (!this.viewer || this.viewer.ref.filename !== item.ref.filename) return;
      const url = res?.dataUrl ?? "";
      // Parse PNG tEXt chunks once on arrival. parseEmbeddedMetadata
      // tolerates non-PNGs and absent chunks (returns nulls), so the
      // lightbox always has a stable shape to render against.
      const metadata = url ? parseEmbeddedMetadata(url) : null;
      this.viewer = { ref: item.ref, dataUrl: url, loading: false, metadata };
    } catch (err) {
      if (!this.viewer || this.viewer.ref.filename !== item.ref.filename) return;
      this.viewer = {
        ref: item.ref,
        dataUrl: this.viewer.dataUrl,
        loading: false,
        error: String(err),
        metadata: this.viewer.metadata,
      };
    }
  }

  // replayFromViewer pulls every knob the Studio surface exposes out
  // of the embedded prompt graph and pushes them into the form. The
  // user reviews and clicks Generate themselves — we don't auto-fire
  // a run because the human almost always wants to tweak something
  // before re-rolling.
  private replayFromViewer() {
    if (!this.viewer?.metadata?.prompt) return;
    const graph = this.viewer.metadata.prompt;
    const p = extractParamsFromGraph(graph);

    // Match the embedded graph's structural signature against the
    // shipped workflows so we can pick the right template entry.
    // Fall back to keeping the current selection silently — the
    // params still apply, but the dropdown stays where it is.
    void this.findAndSelectWorkflow(graphSignature(graph));

    if (p.positivePrompt != null) this.prompt = p.positivePrompt;
    if (p.negativePrompt != null) this.negativePrompt = p.negativePrompt;
    if (p.seed != null) {
      this.seed = p.seed;
      this.randomizeSeed = false; // exact-seed replay is the point
    }
    if (p.steps != null) this.steps = p.steps;
    if (p.cfg != null) this.cfg = p.cfg;
    if (p.sampler) this.sampler = p.sampler;
    if (p.scheduler) this.scheduler = p.scheduler;
    // LoRA + IPAdapter: only apply values to slots that already exist
    // in the current workflow's introspected layout (matching by
    // node id). If the workflow is the same as the source, every
    // slot maps. If it's a different template, only matching nodes
    // get applied; the rest keep their current values.
    if (p.loras.length > 0) {
      const byId = new Map(p.loras.map((l) => [l.nodeId, l.value]));
      this.loraSlots = this.loraSlots.map((s) =>
        byId.has(s.nodeId) ? { ...s, value: byId.get(s.nodeId) as number } : s,
      );
    }
    if (p.ipas.length > 0) {
      const byId = new Map(p.ipas.map((l) => [l.nodeId, l.value]));
      this.ipaSlots = this.ipaSlots.map((s) =>
        byId.has(s.nodeId) ? { ...s, value: byId.get(s.nodeId) as number } : s,
      );
    }
    this.closeViewer();
  }

  // findAndSelectWorkflow picks the first workflow whose graph
  // structure matches the embedded prompt's signature. Async because
  // it needs to fetch each candidate's graph; only fires when the
  // current selection doesn't already match.
  private async findAndSelectWorkflow(targetSig: string) {
    if (!this.client) return;
    // Fast path: if the currently-selected graph already matches,
    // don't probe others.
    if (this.currentGraph && graphSignature(this.currentGraph) === targetSig) {
      return;
    }
    for (const wf of this.workflows) {
      if (!wf.id) continue; // skip the user's default (no id)
      try {
        const res = await this.client.request<{ id: string; graph: WorkflowGraph }>(
          "images.workflows.get",
          { id: wf.id },
        );
        if (res?.graph && graphSignature(res.graph) === targetSig) {
          this.selectedWorkflowId = wf.id;
          this.currentGraph = res.graph;
          this.applyGraphIntrospection(res.graph);
          return;
        }
      } catch {
        // skip; next candidate
      }
    }
  }

  private closeViewer = () => {
    this.viewer = null;
  };

  // ESC closes the lightbox. Bound on connect so a keyboard user
  // doesn't have to mouse over to the close button.
  private onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape" && this.viewer) {
      e.preventDefault();
      this.closeViewer();
    }
  };

  override connectedCallback() {
    super.connectedCallback();
    void this.loadWorkflows();
    window.addEventListener("keydown", this.onKeyDown);
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener("keydown", this.onKeyDown);
    if (this.detachListener) {
      this.detachListener();
      this.detachListener = null;
    }
  }

  private async fetchThumb(ref: ImageRef) {
    if (!this.client) return;
    try {
      const res = await this.client.request<{ dataUrl: string }>("images.fetch", {
        filename: ref.filename,
        subfolder: ref.subfolder ?? "",
        type: ref.type ?? "output",
        preview: "webp;quality=70",
      });
      this.gallery = this.gallery.map((g) =>
        g.ref.filename === ref.filename ? { ref: g.ref, dataUrl: res?.dataUrl ?? null } : g,
      );
    } catch (err) {
      // Thumbs that fail to load just stay on the placeholder; don't
      // block the gallery on a single bad fetch.
      console.warn("studio: thumb fetch failed", ref.filename, err);
    }
  }

  // --- render ------------------------------------------------------------

  // Theme tokens used here come from openclaw/ui/src/styles/base.css —
  // var names match the rest of the control-ui so light/dark mode
  // and any future theme-swap flows through automatically. Fallback
  // values mirror the dark-mode defaults so a stylesheet load failure
  // still produces a readable view.
  static styles = css`
    :host {
      display: block;
      width: 100%;
      color: var(--text, #d6dce8);
    }
    .wrap {
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      gap: 1rem;
      padding: 1rem;
      align-items: start;
    }
    .controls {
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    label {
      font-size: 0.85rem;
      color: var(--muted, #7a8398);
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    label .meta {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }
    label .value {
      font-variant-numeric: tabular-nums;
      color: var(--text-strong, #f4f7fb);
      font-size: 0.8rem;
    }
    input[type="number"],
    input[type="text"],
    select,
    textarea {
      width: 100%;
      box-sizing: border-box;
      background: var(--input, #1a1f2c);
      color: var(--text, #d6dce8);
      border: 1px solid var(--border, #1a1f2c);
      border-radius: 0.35rem;
      padding: 0.4rem 0.55rem;
      font: inherit;
    }
    input[type="number"]:focus,
    input[type="text"]:focus,
    select:focus,
    textarea:focus,
    input[type="range"]:focus {
      outline: 0;
      border-color: var(--ring, #5fa8ff);
      box-shadow:
        0 0 0 2px var(--bg, #07080c),
        0 0 0 3px var(--ring, #5fa8ff);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent, #5fa8ff);
    }
    input[type="checkbox"] {
      accent-color: var(--accent, #5fa8ff);
    }
    textarea {
      resize: vertical;
      font-family: ui-monospace, monospace;
      min-height: 4rem;
    }
    .group {
      border: 1px solid var(--border, #1a1f2c);
      background: var(--bg-elevated, #14171f);
      border-radius: 0.5rem;
      padding: 0.65rem 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .group h3 {
      margin: 0;
      font-size: 0.8rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--muted, #7a8398);
    }
    button.generate {
      padding: 0.65rem 1rem;
      font-weight: 600;
      background: var(--accent, #5fa8ff);
      color: var(--accent-foreground, #07080c);
      border: 0;
      border-radius: 0.4rem;
      cursor: pointer;
    }
    button.generate:hover:not(:disabled) {
      background: var(--accent-hover, #7cb8ff);
    }
    button.generate:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .status {
      font-size: 0.8rem;
      color: var(--muted, #7a8398);
      min-height: 1.25rem;
    }
    .status.error {
      color: var(--destructive, #ef4444);
    }
    .gallery-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    .gallery-header h3 {
      margin: 0;
      font-size: 0.8rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--muted, #7a8398);
    }
    button.download {
      background: transparent;
      color: var(--text, #d6dce8);
      border: 1px solid var(--border, #1a1f2c);
      border-radius: 0.35rem;
      padding: 0.35rem 0.7rem;
      font: inherit;
      font-size: 0.8rem;
      cursor: pointer;
    }
    button.download:hover:not(:disabled) {
      border-color: var(--border-hover, #3c4564);
    }
    button.download:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 0.5rem;
      align-content: start;
    }
    .gallery .tile {
      aspect-ratio: 1 / 1;
      background: var(--bg-elevated, #14171f);
      border: 1px solid var(--border, #1a1f2c);
      border-radius: 0.4rem;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      cursor: pointer;
      transition: border-color 80ms ease;
    }
    .gallery .tile:hover {
      border-color: var(--border-hover, #3c4564);
    }
    .gallery .tile img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .gallery .tile .placeholder {
      color: var(--muted, #7a8398);
      font-size: 0.7rem;
    }
    /* Lightbox: full-screen backdrop + centered image. Click the
       backdrop or hit ESC to close; controls live in a corner so
       they don't fight with the image. */
    .viewer-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.85);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      z-index: 9999;
      cursor: zoom-out;
    }
    .viewer-backdrop img {
      max-width: 100%;
      max-height: 100%;
      border-radius: 8px;
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
      cursor: default;
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
      max-width: min(80vw, 720px);
      max-height: 60vh;
      overflow: auto;
      cursor: default;
    }
    .viewer-loading {
      color: white;
      font-size: 0.9rem;
    }
    .viewer-metadata {
      margin-top: 0.5rem;
      font-size: 0.8rem;
      cursor: default;
    }
    .viewer-metadata summary {
      cursor: pointer;
      color: rgba(255, 255, 255, 0.75);
      list-style: revert;
      user-select: none;
    }
    .viewer-metadata .meta-summary {
      margin-top: 0.4rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    .viewer-metadata .meta-summary .k {
      display: inline-block;
      width: 5rem;
      color: rgba(255, 255, 255, 0.6);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.7rem;
      vertical-align: top;
    }
    .viewer-metadata .meta-summary .v {
      color: white;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .viewer-metadata .meta-raw {
      margin-top: 0.5rem;
      border-top: 1px solid rgba(255, 255, 255, 0.15);
      padding-top: 0.5rem;
    }
    .viewer-metadata .meta-raw .copy-raw {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.25);
      color: white;
      border-radius: 4px;
      padding: 0.2rem 0.5rem;
      font: inherit;
      font-size: 0.75rem;
      cursor: pointer;
      margin-bottom: 0.4rem;
    }
    .viewer-metadata .meta-raw pre {
      margin: 0;
      max-height: 30vh;
      overflow: auto;
      font-family: ui-monospace, monospace;
      font-size: 0.7rem;
      color: rgba(255, 255, 255, 0.85);
      background: rgba(0, 0, 0, 0.4);
      padding: 0.5rem;
      border-radius: 4px;
      white-space: pre-wrap;
      word-break: break-word;
    }
  `;

  override render() {
    if (!this.client) {
      return html`<div class="wrap"><div>gateway not connected</div></div>`;
    }
    return html`
      <div class="wrap">
        <div class="controls">${this.renderControls()}</div>
        <div>${this.renderGallery()}</div>
      </div>
    `;
  }

  private renderControls() {
    const generating =
      this.genState.kind === "submitting" ||
      this.genState.kind === "running" ||
      this.genState.kind === "fetching";
    return html`
      <label>
        <span>Workflow</span>
        <select
          .value=${this.selectedWorkflowId}
          @change=${(e: Event) => {
            this.selectedWorkflowId = (e.target as HTMLSelectElement).value;
            void this.loadWorkflowGraph(this.selectedWorkflowId);
          }}
        >
          ${this.workflowsLoading ? html`<option>loading…</option>` : nothing}
          ${this.workflows.map(
            (w) =>
              html`<option .value=${w.id} ?selected=${w.id === this.selectedWorkflowId}>
                ${w.label}
              </option>`,
          )}
        </select>
      </label>

      <label>
        <span>Prompt</span>
        <textarea
          rows="4"
          .value=${this.prompt}
          @input=${(e: Event) => (this.prompt = (e.target as HTMLTextAreaElement).value)}
        ></textarea>
      </label>

      <label>
        <span>Negative Prompt</span>
        <textarea
          rows="2"
          .value=${this.negativePrompt}
          @input=${(e: Event) => (this.negativePrompt = (e.target as HTMLTextAreaElement).value)}
        ></textarea>
      </label>

      <div class="row">
        <label style="flex:1">
          <span>Seed</span>
          <input
            type="number"
            .value=${String(this.seed)}
            @input=${(e: Event) => (this.seed = Number((e.target as HTMLInputElement).value))}
          />
        </label>
        <label style="flex:0 0 auto; align-self:end; flex-direction:row; gap:0.35rem;">
          <input
            type="checkbox"
            .checked=${this.randomizeSeed}
            @change=${(e: Event) => (this.randomizeSeed = (e.target as HTMLInputElement).checked)}
          />
          <span style="color:var(--text-strong,#f4f7fb)">Randomize</span>
        </label>
      </div>

      ${this.renderSliderRow("Steps", this.steps, 8, 100, 1, (v) => (this.steps = v))}
      ${this.renderSliderRow("CFG Weight", this.cfg, 1, 20, 0.1, (v) => (this.cfg = v))}

      <label>
        <span>Sampler</span>
        <input
          type="text"
          list="studio-samplers"
          .value=${this.sampler}
          @change=${(e: Event) => (this.sampler = (e.target as HTMLInputElement).value)}
        />
        <datalist id="studio-samplers">
          ${SAMPLERS.map((s) => html`<option .value=${s}></option>`)}
        </datalist>
      </label>

      <label>
        <span>Scheduler</span>
        <input
          type="text"
          list="studio-schedulers"
          .value=${this.scheduler}
          @change=${(e: Event) => (this.scheduler = (e.target as HTMLInputElement).value)}
        />
        <datalist id="studio-schedulers">
          ${SCHEDULERS.map((s) => html`<option .value=${s}></option>`)}
        </datalist>
      </label>

      <label>
        <span>Number of Images</span>
        <input
          type="number"
          min="1"
          max="100"
          step="1"
          .value=${String(this.numImages)}
          @input=${(e: Event) => {
            const n = Number((e.target as HTMLInputElement).value);
            // Clamp to a sane floor; no upper cap beyond what the
            // input's max attribute already enforces. The agent can
            // still enter 100 if they really want a long batch.
            this.numImages = Number.isFinite(n) && n > 0 ? Math.floor(n) : 1;
          }}
        />
      </label>

      ${this.loraSlots.length > 0
        ? html`
            <div class="group">
              <h3>LoRA strengths</h3>
              ${this.loraSlots.map((slot, idx) =>
                this.renderSliderRow(
                  slot.label,
                  slot.value,
                  LORA_RANGE[0],
                  LORA_RANGE[1],
                  0.01,
                  (v) => {
                    const next = [...this.loraSlots];
                    next[idx] = { ...slot, value: v };
                    this.loraSlots = next;
                  },
                ),
              )}
            </div>
          `
        : nothing}
      ${this.ipaSlots.length > 0
        ? html`
            <div class="group">
              <h3>IPAdapter weights</h3>
              ${this.ipaSlots.map((slot, idx) =>
                this.renderSliderRow(
                  slot.label,
                  slot.value,
                  IPA_RANGE[0],
                  IPA_RANGE[1],
                  0.01,
                  (v) => {
                    const next = [...this.ipaSlots];
                    next[idx] = { ...slot, value: v };
                    this.ipaSlots = next;
                  },
                ),
              )}
            </div>
          `
        : nothing}

      <button
        class="generate"
        ?disabled=${generating || !this.selectedWorkflowId}
        @click=${() => void this.submit()}
      >
        ${generating ? "generating…" : "Generate"}
      </button>
      <div class="status ${this.genState.kind === "error" ? "error" : ""}">
        ${this.renderStatus()}
      </div>
    `;
  }

  private renderSliderRow(
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    onChange: (v: number) => void,
  ) {
    return html`
      <label>
        <div class="meta">
          <span>${label}</span>
          <span class="value">${this.formatNumber(value)}</span>
        </div>
        <input
          type="range"
          .min=${String(min)}
          .max=${String(max)}
          .step=${String(step)}
          .value=${String(value)}
          @input=${(e: Event) => onChange(Number((e.target as HTMLInputElement).value))}
        />
      </label>
    `;
  }

  private formatNumber(n: number): string {
    if (Number.isInteger(n)) return String(n);
    // Two decimals max — keeps the slider value chip stable in width.
    return n.toFixed(2).replace(/\.?0+$/, "");
  }

  private renderStatus() {
    switch (this.genState.kind) {
      case "idle":
        return this.workflowsError ? `error: ${this.workflowsError}` : nothing;
      case "submitting":
        return "submitting…";
      case "running":
        if (this.genState.progress) {
          const { value, max } = this.genState.progress;
          return `running… ${value}/${max}`;
        }
        return "running…";
      case "fetching":
        return "fetching result…";
      case "error":
        return this.genState.message;
    }
  }

  private renderGallery() {
    if (this.gallery.length === 0) {
      return html`<div class="status">Generate to fill the gallery.</div>`;
    }
    const dl = this.downloading;
    const dlLabel = dl
      ? `Downloading ${dl.done}/${dl.total}…`
      : `Download (${this.gallery.length})`;
    return html`
      <div class="gallery-header">
        <h3>Gallery</h3>
        <button
          class="download"
          ?disabled=${dl !== null}
          @click=${() => void this.downloadAll()}
          title="Download every image in the gallery"
        >
          ${dlLabel}
        </button>
      </div>
      <div class="gallery">
        ${this.gallery.map(
          (g) =>
            html`<div class="tile" title=${g.ref.filename} @click=${() => this.openViewer(g)}>
              ${g.dataUrl
                ? html`<img src=${g.dataUrl} alt=${g.ref.filename} />`
                : html`<span class="placeholder">${g.ref.filename}</span>`}
            </div>`,
        )}
      </div>
      ${this.renderViewer()}
    `;
  }

  private renderViewer() {
    if (!this.viewer) return nothing;
    const { ref, dataUrl, loading, error, metadata } = this.viewer;
    const replayable = metadata?.prompt != null;
    return html`
      <div class="viewer-backdrop" @click=${this.closeViewer}>
        ${dataUrl
          ? html`<img
              src=${dataUrl}
              alt=${ref.filename}
              @click=${(e: Event) => e.stopPropagation()}
            />`
          : html`<div class="viewer-loading">${loading ? "loading…" : (error ?? "no image")}</div>`}
        <div class="viewer-controls" @click=${(e: Event) => e.stopPropagation()}>
          <button
            ?disabled=${!replayable}
            title=${replayable
              ? "Populate the form from this image's embedded metadata"
              : "No embedded metadata"}
            @click=${() => this.replayFromViewer()}
          >
            Replay
          </button>
          <button @click=${this.closeViewer}>Close</button>
        </div>
        <div class="viewer-meta">${ref.filename}${this.renderMetaPanel(metadata)}</div>
      </div>
    `;
  }

  // renderMetaPanel renders a collapsible details with the extracted
  // params summary + the raw prompt JSON. Returns nothing when no
  // metadata was embedded so the lightbox stays clean for legacy
  // images.
  private renderMetaPanel(metadata: ParsedMeta | null) {
    if (!metadata?.prompt) return nothing;
    const p = extractParamsFromGraph(metadata.prompt);
    const rawJson = JSON.stringify(metadata.prompt, null, 2);
    return html`
      <details class="viewer-metadata" @click=${(e: Event) => e.stopPropagation()}>
        <summary>metadata</summary>
        <div class="meta-summary">
          ${p.positivePrompt != null
            ? html`<div>
                <span class="k">prompt</span><span class="v">${p.positivePrompt}</span>
              </div>`
            : nothing}
          ${p.negativePrompt != null
            ? html`<div>
                <span class="k">negative</span><span class="v">${p.negativePrompt}</span>
              </div>`
            : nothing}
          ${p.seed != null
            ? html`<div><span class="k">seed</span><span class="v">${p.seed}</span></div>`
            : nothing}
          ${p.steps != null && p.cfg != null && p.sampler && p.scheduler
            ? html`<div>
                <span class="k">sampler</span>
                <span class="v">${p.sampler}/${p.scheduler} · ${p.steps} steps · cfg ${p.cfg}</span>
              </div>`
            : nothing}
          ${p.loras.length > 0
            ? html`<div>
                <span class="k">loras</span>
                <span class="v">${p.loras.map((l) => `${l.label}=${l.value}`).join(", ")}</span>
              </div>`
            : nothing}
          ${p.ipas.length > 0
            ? html`<div>
                <span class="k">ipadapter</span>
                <span class="v">${p.ipas.map((l) => `${l.label}=${l.value}`).join(", ")}</span>
              </div>`
            : nothing}
        </div>
        <div class="meta-raw">
          <button
            class="copy-raw"
            @click=${async () => {
              try {
                await navigator.clipboard.writeText(rawJson);
              } catch (err) {
                console.warn("studio: clipboard write failed", err);
              }
            }}
          >
            Copy JSON
          </button>
          <pre>${rawJson}</pre>
        </div>
      </details>
    `;
  }
}

// renderStudio is the entry point app-render.ts calls. Returns a Lit
// template that mounts the custom element and hands it the gateway
// client + session key.
export function renderStudio(props: { client: GatewayBrowserClient | null; sessionKey: string }) {
  return html`<studio-view .client=${props.client} .sessionKey=${props.sessionKey}></studio-view>`;
}

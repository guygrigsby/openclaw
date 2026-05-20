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

// Embedded PNG metadata. ComfyUI uses JSON `prompt`/`workflow` tEXt
// chunks (graph form). A1111/Forge/Civitai use a single `parameters`
// tEXt chunk with text format: positive prompt, "Negative prompt:" +
// negative, then "Steps: …, Sampler: …" k:v settings on the next
// line. We support both. `params` is populated when either path
// yields enough to drive Replay; `prompt` is the ComfyUI graph form
// only (null on A1111 sources).
type ParsedMeta = {
  prompt: WorkflowGraph | null;
  workflow: WorkflowGraph | null;
  params: ExtractedParams | null;
  raw: Record<string, string>;
  // chunkTypes is every PNG chunk type seen during parsing (IDAT,
  // IHDR, tEXt, iTXt, etc.). Used by the diagnostic message when
  // we can't extract usable metadata, so the user knows whether the
  // PNG had text chunks at all or whether a dialect was rejected.
  chunkTypes: string[];
};
type GenState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "running"; runId: string; progress?: { value: number; max: number } }
  | { kind: "fetching"; runId: string }
  | { kind: "error"; message: string };

// StylePreset mirrors the server-side StylePreset shape from
// internal/server/styles.go. Suffixes append to the user's prompt
// at submit time; denoise is the suggested KSampler.denoise default
// for img2img workflows when this preset is active. Lora describes
// an optional LoRA file requirement (filename + civitai install
// metadata); the UI cross-references against the active ComfyUI's
// installed LoRA set to surface install hints.
type CivitaiRef = {
  modelId?: number;
  versionId?: number;
  sha256?: string;
  downloadUrl?: string;
  page?: string;
};
type LoraRequirement = {
  filename: string;
  strength?: number;
  baseModel?: string;
  civitai?: CivitaiRef;
};
type StylePreset = {
  id: string;
  label: string;
  description?: string;
  promptSuffix: string;
  negativeSuffix?: string;
  denoise: number;
  baseModel?: string;
  lora?: LoraRequirement;
  source?: string;
};

// ManagerStatus is the response shape for images.manager.status.
type ManagerStatus = { present: boolean; endpoint?: string };

// extractLoraFilenames parses ComfyUI's /object_info shape and pulls
// out the LoraLoader's lora_name enum into a Set of filenames. The
// shape is `LoraLoader.input.required.lora_name = [[<names>], <metadata>]`
// in modern ComfyUI; older versions emit a flat array. Both are
// tolerated; unrecognized shapes return an empty set rather than
// throw, so the install-button visibility logic falls open.
function extractLoraFilenames(info: Record<string, unknown>): Set<string> {
  const out = new Set<string>();
  const loader = info["LoraLoader"];
  if (!loader || typeof loader !== "object") return out;
  const input = (loader as Record<string, unknown>)["input"];
  if (!input || typeof input !== "object") return out;
  const required = (input as Record<string, unknown>)["required"];
  if (!required || typeof required !== "object") return out;
  const loraName = (required as Record<string, unknown>)["lora_name"];
  if (!Array.isArray(loraName) || loraName.length === 0) return out;
  // Modern shape: [["a.safetensors", "b.safetensors"], { ...metadata }]
  if (Array.isArray(loraName[0])) {
    for (const n of loraName[0]) {
      if (typeof n === "string" && n) out.add(n);
    }
    return out;
  }
  // Flat shape fallback.
  for (const n of loraName) {
    if (typeof n === "string" && n) out.add(n);
  }
  return out;
}

// SourceImage is the staged upload for img2img runs: filename is
// what ComfyUI stored it as (used in nodeOverrides for the LoadImage
// node), dataUrl is the local-side preview shown next to the file
// picker. Cleared between runs only when the user explicitly removes
// the file.
type SourceImage = {
  filename: string;
  dataUrl: string;
};

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

// Sentinel workflow id for the "🎲 Random Workflow" dropdown entry.
// When this is selected, each iteration of the Generate batch picks
// a fresh builtin workflow at random. Not returned by
// images.workflows.list — purely a UI-side option.
const RANDOM_WORKFLOW_ID = "__random__";

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
  // Track the best score for each prompt slot so a high-priority
  // title (e.g. "Positive (user)") wins over a co-existing
  // lower-priority one (e.g. bare "Positive" — never expected in
  // the same graph but harmless to be defensive).
  let positiveScore = 0;
  let negativeScore = 0;
  // Sort keys for stable slot order (workflows reuse small int ids).
  const ids = Object.keys(graph).sort((a, b) => a.length - b.length || a.localeCompare(b));
  for (const nid of ids) {
    const node = graph[nid];
    const cls = node?.class_type ?? "";
    const title = node?._meta?.title ?? "";
    const inputs = node?.inputs ?? {};
    // Match the vanilla CLIPTextEncode plus common variants
    // (smZ CLIPTextEncode is Civitai's flavor). Anything ending in
    // "CLIPTextEncode" is treated as a prompt-encoding node.
    // Match the vanilla CLIPTextEncode plus common variants
    // (smZ CLIPTextEncode is Civitai's flavor). Anything ending in
    // "CLIPTextEncode" is treated as a prompt-encoding node.
    if (cls === "CLIPTextEncode" || cls.endsWith("CLIPTextEncode")) {
      const text = typeof inputs["text"] === "string" ? (inputs["text"] as string) : null;
      const cls2 = classifyUserPromptTitle(title);
      if (text != null) {
        // Civitai-hosted resources (urn:air:…) embedded in the text
        // won't resolve on a local ComfyUI install — strip them so
        // the user sees a clean prompt and the run won't fail on
        // missing-embedding errors.
        const cleaned = stripCivitaiURNRefs(text);
        // Higher-priority titles win when multiple positive/negative
        // CLIPTextEncode nodes exist in one workflow (e.g. vyx has
        // "Positive (user)" + "Positive (permanent: …)" + sometimes
        // "Positive (avatar …)" — only the (user) one is the user
        // input slot).
        if (cls2.kind === "positive" && cls2.score > positiveScore) {
          out.positivePrompt = cleaned;
          positiveScore = cls2.score;
        } else if (cls2.kind === "negative" && cls2.score > negativeScore) {
          out.negativePrompt = cleaned;
          negativeScore = cls2.score;
        }
      }
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

// stripCivitaiURNRefs removes Civitai-hosted resource references
// from a prompt string. Civitai stores embeddings as URNs like
//   embedding:urn:air:sd1:embedding:civitai:222256@250708
// or sometimes just the bare URN form. These resolve on Civitai's
// servers but not on a local ComfyUI install — leaving them in
// the prompt either silently no-ops or throws "embedding not
// found" depending on the encoder. Strip them along with any
// trailing comma/whitespace, then collapse the resulting double
// commas. Idempotent.
function stripCivitaiURNRefs(text: string): string {
  if (!text) return text;
  // 1. embedding:urn:air:... — the wrapped form
  // 2. bare urn:air:... — sometimes appears alone
  let out = text.replace(/embedding:urn:air:[^,\s]+/g, "").replace(/urn:air:[^,\s]+/g, "");
  // Collapse any double-comma artifacts left behind, plus tidy
  // leading/trailing commas + extra whitespace.
  out = out
    .replace(/(?:,\s*){2,}/g, ", ")
    .replace(/^[,\s]+|[,\s]+$/g, "")
    .replace(/\s{2,}/g, " ");
  return out;
}

// classifyUserPromptTitle looks at a CLIPTextEncode node's _meta.title
// and decides whether it's the user's positive/negative prompt slot
// (vs a permanent-style anchor or an avatar-tags slot). Returns a
// {kind, score} pair so a single-pass extractor can pick the
// highest-priority match when a workflow has more than one positive
// or negative CLIPTextEncode node.
//
// Title conventions across the bundled workflows:
//   - "Positive (user)" / "Negative (user)" — the explicit
//     vyx/pony/dixar_polish convention. Highest priority.
//   - "CLIP Text Encode (Prompt)" / "(Negative Prompt)" — illustrious
//     workflows; these are themselves the user inputs (no separate
//     "(user)" slot). Mid priority.
//   - bare "Positive" / "Negative" — older dixar / *_hyper8 style.
//     Lowest priority but still a user slot.
//   - "Positive (permanent: …)" / "Negative (permanent: …)" — style
//     anchors, NOT user input. Score 0 (skip).
//   - "Positive (avatar appearance, …)" — avatar-tags slot; not the
//     user prompt. Score 0 (skip).
function classifyUserPromptTitle(title: string): {
  kind: "positive" | "negative" | "skip";
  score: number;
} {
  const t = title.toLowerCase().trim();
  // Permanent / avatar slots are never the user's prompt.
  if (t.includes("permanent") || t.includes("avatar")) {
    return { kind: "skip", score: 0 };
  }
  if (t.includes("(user)")) {
    if (t.includes("positive")) return { kind: "positive", score: 100 };
    if (t.includes("negative")) return { kind: "negative", score: 100 };
  }
  // Illustrious: "CLIP Text Encode (Negative Prompt)" must hit
  // negative before "prompt" alone hits positive.
  if (t.includes("negative prompt")) return { kind: "negative", score: 75 };
  if (t.includes("prompt")) return { kind: "positive", score: 75 };
  // Bare "Positive" / "Negative" — match exact-word to avoid
  // accidentally catching e.g. "Positive (something else)".
  if (t === "positive") return { kind: "positive", score: 50 };
  if (t === "negative") return { kind: "negative", score: 50 };
  return { kind: "skip", score: 0 };
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

// parsePngTextChunksFromBytes walks PNG chunks in a raw byte buffer
// and returns every text chunk it finds, plus the chunk types seen
// (for diagnostics when the caller needs to explain "no readable
// metadata" to the user).
//
// Supports three PNG text-chunk dialects:
//   - tEXt: Latin-1 keyword + Latin-1 text. ComfyUI's default.
//   - iTXt: UTF-8 keyword + optional zlib-compressed UTF-8 text.
//     What Civitai's image optimizer (sharp/libpng) tends to write.
//   - zTXt: Latin-1 keyword + always-zlib-compressed Latin-1 text.
//     Older shape; rarely seen on AI-generated PNGs.
//
// Async because compressed payloads use the streaming
// DecompressionStream API. Uncompressed-only callers get fast paths.
async function parsePngTextChunksFromBytes(
  bytes: Uint8Array,
): Promise<{ chunks: Record<string, string>; types: string[] }> {
  const out: Record<string, string> = {};
  const types: string[] = [];
  if (
    bytes.length < 8 ||
    bytes[0] !== 0x89 ||
    bytes[1] !== 0x50 ||
    bytes[2] !== 0x4e ||
    bytes[3] !== 0x47
  ) {
    return { chunks: out, types };
  }
  let offset = 8;
  const latin1 = new TextDecoder("latin1");
  const utf8 = new TextDecoder("utf-8");
  while (offset + 8 <= bytes.length) {
    const length =
      (bytes[offset] << 24) |
      (bytes[offset + 1] << 16) |
      (bytes[offset + 2] << 8) |
      bytes[offset + 3];
    const type = latin1.decode(bytes.subarray(offset + 4, offset + 8));
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > bytes.length) break;
    types.push(type);
    if (type === "tEXt") {
      const chunk = bytes.subarray(dataStart, dataEnd);
      const nul = chunk.indexOf(0);
      if (nul > 0) {
        out[latin1.decode(chunk.subarray(0, nul))] = latin1.decode(chunk.subarray(nul + 1));
      }
    } else if (type === "iTXt") {
      // iTXt layout: keyword \0 compFlag(1) compMethod(1) langTag \0 transKw \0 text
      const chunk = bytes.subarray(dataStart, dataEnd);
      const nul1 = chunk.indexOf(0);
      if (nul1 > 0 && nul1 + 2 < chunk.length) {
        const keyword = latin1.decode(chunk.subarray(0, nul1));
        const compFlag = chunk[nul1 + 1];
        const compMethod = chunk[nul1 + 2];
        let p = nul1 + 3;
        const nul2 = chunk.indexOf(0, p);
        if (nul2 >= 0) {
          p = nul2 + 1; // skip language tag
          const nul3 = chunk.indexOf(0, p);
          if (nul3 >= 0) {
            p = nul3 + 1; // skip translated keyword
            const textBytes = chunk.subarray(p);
            try {
              if (compFlag === 1 && compMethod === 0) {
                // zlib-wrapped deflate. Modern browsers expose
                // DecompressionStream; node test envs may not.
                out[keyword] = utf8.decode(await inflate(textBytes));
              } else if (compFlag === 0) {
                out[keyword] = utf8.decode(textBytes);
              }
            } catch {
              // skip — bad payload, leave keyword absent
            }
          }
        }
      }
    } else if (type === "zTXt") {
      // zTXt: keyword \0 compMethod(1) compressedText
      const chunk = bytes.subarray(dataStart, dataEnd);
      const nul = chunk.indexOf(0);
      if (nul > 0 && nul + 1 < chunk.length) {
        const keyword = latin1.decode(chunk.subarray(0, nul));
        const compMethod = chunk[nul + 1];
        const textBytes = chunk.subarray(nul + 2);
        if (compMethod === 0) {
          try {
            out[keyword] = latin1.decode(await inflate(textBytes));
          } catch {
            // skip
          }
        }
      }
    }
    if (type === "IEND") break;
    offset = dataEnd + 4; // skip CRC
  }
  return { chunks: out, types };
}

// inflate decompresses zlib-wrapped data using the browser's
// DecompressionStream. Throws on environments without the API
// (caught by the chunk parser, which then skips the chunk).
async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream("deflate");
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(ds);
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

// detectImageFormat sniffs the first bytes for a known image-format
// signature. Used by the unified entry point to dispatch to the
// PNG-chunk parser vs the JPEG-EXIF parser, and to surface "this
// is actually a WebP" diagnostics when neither produces metadata.
function detectImageFormat(bytes: Uint8Array): "PNG" | "JPEG" | "WebP" | "GIF" | "unknown" {
  if (bytes.length < 4) return "unknown";
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47)
    return "PNG";
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "JPEG";
  if (
    bytes.length >= 12 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    return "WebP";
  }
  if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46) return "GIF";
  return "unknown";
}

// parseJpegExif walks JPEG markers, finds APP1 with the Exif
// signature, and parses the embedded TIFF/IFD to extract
// UserComment (the EXIF tag where SD-tooling stores generation
// metadata). Returns the same shape parsePngTextChunksFromBytes
// returns: { chunks, types } where chunks is keyword→text and
// types is the list of segments seen (markers like "FFE0", "Exif",
// "SOS", etc.) used by the diagnostic.
function parseJpegExif(bytes: Uint8Array): { chunks: Record<string, string>; types: string[] } {
  const out: Record<string, string> = {};
  const types: string[] = [];
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) {
    return { chunks: out, types };
  }
  let offset = 2; // skip SOI
  while (offset + 1 < bytes.length) {
    if (bytes[offset] !== 0xff) break;
    // Skip fill bytes (0xFF padding before a marker)
    while (offset < bytes.length && bytes[offset] === 0xff) offset++;
    if (offset >= bytes.length) break;
    const marker = bytes[offset];
    offset++;
    types.push(`FF${marker.toString(16).toUpperCase().padStart(2, "0")}`);
    // Standalone markers (no length): RST0-7 (D0-D7), SOI (D8),
    // EOI (D9), TEM (01).
    if (marker === 0xd9) break; // EOI
    if (marker === 0x00 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    // Length-prefixed segment.
    if (offset + 2 > bytes.length) break;
    const length = (bytes[offset] << 8) | bytes[offset + 1];
    if (length < 2 || offset + length > bytes.length) break;
    const segStart = offset + 2;
    const segEnd = offset + length;
    const seg = bytes.subarray(segStart, segEnd);
    // APP1 (E1) with "Exif\0\0" prefix → TIFF+IFD with UserComment.
    if (
      marker === 0xe1 &&
      seg.length >= 6 &&
      seg[0] === 0x45 &&
      seg[1] === 0x78 &&
      seg[2] === 0x69 &&
      seg[3] === 0x66 &&
      seg[4] === 0x00 &&
      seg[5] === 0x00
    ) {
      types.push("Exif");
      const userComment = parseTiffUserComment(seg.subarray(6));
      if (userComment != null) {
        out["UserComment"] = userComment;
      }
    }
    offset = segEnd;
    // SOS (DA) marks the start of compressed image data — no more
    // metadata segments after this.
    if (marker === 0xda) break;
  }
  return { chunks: out, types };
}

// parseTiffUserComment reads an EXIF TIFF block (the bytes after
// the "Exif\0\0" header in APP1) and returns the decoded
// UserComment string, or null if absent/unreadable.
//
// TIFF layout: 8-byte header (II/MM byte order, magic 0x002A,
// offset to IFD0), then IFD0 entries (12 bytes each: tag, type,
// count, value/offset). ExifIFDPointer (tag 0x8769) gives the
// offset to the ExifIFD where UserComment (tag 0x9286) actually
// lives in well-formed files.
function parseTiffUserComment(tiff: Uint8Array): string | null {
  if (tiff.length < 8) return null;
  const isLE = tiff[0] === 0x49 && tiff[1] === 0x49;
  const isBE = tiff[0] === 0x4d && tiff[1] === 0x4d;
  if (!isLE && !isBE) return null;
  const u16 = (o: number) => (isLE ? tiff[o] | (tiff[o + 1] << 8) : (tiff[o] << 8) | tiff[o + 1]);
  const u32 = (o: number) =>
    isLE
      ? (tiff[o] | (tiff[o + 1] << 8) | (tiff[o + 2] << 16) | (tiff[o + 3] << 24)) >>> 0
      : ((tiff[o] << 24) | (tiff[o + 1] << 16) | (tiff[o + 2] << 8) | tiff[o + 3]) >>> 0;
  if (u16(2) !== 0x002a) return null;
  const ifd0 = u32(4);

  // Walk an IFD looking for UserComment (0x9286) and
  // ExifIFDPointer (0x8769). Returns whichever it found.
  const walk = (start: number): { userComment?: Uint8Array; exifIFD?: number } => {
    if (start + 2 > tiff.length) return {};
    const n = u16(start);
    let userComment: Uint8Array | undefined;
    let exifIFD: number | undefined;
    for (let i = 0; i < n; i++) {
      const e = start + 2 + i * 12;
      if (e + 12 > tiff.length) break;
      const tag = u16(e);
      const count = u32(e + 4);
      const valOff = e + 8;
      if (tag === 0x8769) {
        exifIFD = u32(valOff);
      } else if (tag === 0x9286) {
        // UserComment: bytes (TIFF type 7 / UNDEFINED). 4 bytes
        // inline if total ≤ 4, otherwise the value is an offset.
        if (count <= 4) {
          userComment = tiff.subarray(valOff, valOff + count);
        } else {
          const dataOffset = u32(valOff);
          userComment = tiff.subarray(dataOffset, dataOffset + count);
        }
      }
    }
    return { userComment, exifIFD };
  };

  // Civitai puts UserComment in IFD0 sometimes and ExifIFD other
  // times depending on the encoder. Check both.
  let r = walk(ifd0);
  let userComment = r.userComment;
  if (r.exifIFD) {
    const r2 = walk(r.exifIFD);
    if (r2.userComment) userComment = r2.userComment;
  }
  if (!userComment) return null;

  // First 8 bytes are the EXIF character-code prefix:
  //   "UNICODE\0", "ASCII\0\0\0", "JIS\0\0\0\0\0", or
  //   "\0\0\0\0\0\0\0\0" (undefined).
  if (userComment.length < 8) return null;
  const headerBytes = userComment.subarray(0, 8);
  const header = String.fromCharCode(...Array.from(headerBytes));
  const body = userComment.subarray(8);
  // Try multiple decodings — Civitai is known to use "UNICODE\0"
  // as the prefix while actually writing UTF-8 (technically out of
  // spec). Pick the result that has fewer NUL chars / is longer
  // visually.
  const candidates: string[] = [];
  if (header.startsWith("UNICODE")) {
    try {
      candidates.push(new TextDecoder("utf-16le").decode(body));
    } catch {
      // ignore
    }
    try {
      candidates.push(new TextDecoder("utf-16be").decode(body));
    } catch {
      // ignore
    }
  }
  // Always try UTF-8 — handles ASCII\0\0\0, undefined-prefix, and
  // Civitai's mislabeled UTF-8.
  try {
    candidates.push(new TextDecoder("utf-8").decode(body));
  } catch {
    // ignore
  }
  // Pick the candidate with the most printable characters
  // (cheap proxy for "this decoded correctly").
  let best = "";
  let bestScore = -1;
  for (const c of candidates) {
    const score = countPrintable(c);
    if (score > bestScore) {
      best = c;
      bestScore = score;
    }
  }
  return best || null;
}

// countPrintable returns the count of ASCII-printable characters
// (0x20–0x7E plus newline/tab). Used to score candidate decodings
// of EXIF UserComment when the EXIF char-code prefix lies about
// endianness (Civitai writes "UNICODE\0" but the actual bytes can
// be either UTF-16LE, UTF-16BE, or even UTF-8). The correct
// decoding produces JSON which is ASCII-only — scoring pure ASCII
// disambiguates cleanly from a wrong-endianness decode that fills
// the string with CJK or extended-Latin garbage.
function countPrintable(s: string): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    if (code >= 0x20 && code <= 0x7e) n++;
    else if (code === 0x09 || code === 0x0a || code === 0x0d) n++;
  }
  return n;
}

// parseImageMetadataFromBytes is the unified entry point. Sniffs
// the format and dispatches to the right parser. Returns the same
// shape every callee uses so downstream code stays format-agnostic.
async function parseImageMetadataFromBytes(
  bytes: Uint8Array,
): Promise<{ chunks: Record<string, string>; types: string[] }> {
  const format = detectImageFormat(bytes);
  if (format === "PNG") return parsePngTextChunksFromBytes(bytes);
  if (format === "JPEG") return parseJpegExif(bytes);
  // WebP / GIF / unknown: nothing to extract today. Return empty
  // chunks but include the format detection in the types so the
  // diagnostic message can surface what we saw.
  return { chunks: {}, types: format !== "unknown" ? [format] : [] };
}

// parseDataUrl decodes a base64 data URL into image metadata
// chunks. Thin async wrapper around parseImageMetadataFromBytes
// (sniffs PNG vs JPEG vs other) for the lightbox path which already
// has data URLs from images.fetch.
async function parseDataUrl(
  dataUrl: string,
): Promise<{ chunks: Record<string, string>; types: string[] }> {
  const comma = dataUrl.indexOf(",");
  if (comma < 0) return { chunks: {}, types: [] };
  try {
    const bin = atob(dataUrl.slice(comma + 1));
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return await parseImageMetadataFromBytes(bytes);
  } catch {
    return { chunks: {}, types: [] };
  }
}

// parseA1111Parameters reads the A1111/Forge/Civitai `parameters`
// tEXt chunk format:
//
//   <positive prompt, possibly with <lora:name:weight> tokens>
//   Negative prompt: <negative>
//   Steps: 30, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7,
//     Seed: 1234, Model: duchaitenPonyReal_v20, ...
//
// Returns ExtractedParams or null when the text doesn't look like
// A1111 format (missing both "Negative prompt:" and "Steps:" keys).
// LoRA-token weights are returned as `loras` slots keyed by the
// raw lora name; the caller maps them to the current workflow's
// node ids when applying.
function parseA1111Parameters(text: string): ExtractedParams | null {
  if (!text) return null;
  if (!/Negative prompt:|Steps:/i.test(text)) return null;

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

  // Split into the three sections: positive | negative | settings.
  // Find the start of the settings line — first occurrence of
  // "Steps:" at a line boundary. The text before "Negative prompt:"
  // (if present) is the positive; between that and Steps is the
  // negative; everything from Steps onward is k:v settings.
  const stepsIdx = text.search(/(^|\n)Steps:/);
  let positiveAndNeg = text;
  let settingsPart = "";
  if (stepsIdx >= 0) {
    positiveAndNeg = text.slice(0, stepsIdx).trimEnd();
    settingsPart = text.slice(stepsIdx).replace(/^\n/, "");
  }
  const negIdx = positiveAndNeg.indexOf("Negative prompt:");
  if (negIdx >= 0) {
    out.positivePrompt = positiveAndNeg.slice(0, negIdx).trim();
    out.negativePrompt = positiveAndNeg.slice(negIdx + "Negative prompt:".length).trim();
  } else {
    out.positivePrompt = positiveAndNeg.trim();
  }

  // Tokenize the settings line. Keys are PascalCase or
  // "Title Case" with spaces, values are either a quoted string or
  // a non-comma run. Examples:
  //   Steps: 30
  //   Sampler: DPM++ 2M SDE
  //   CFG scale: 7
  //   Schedule type: Karras
  //   Lora hashes: "abc: hash, def: hash"
  if (settingsPart) {
    const re = /([A-Za-z][A-Za-z+ ]+?):\s*("(?:\\"|[^"])*"|[^,]+?)(?=,\s*[A-Z]|,?\s*$)/g;
    const kv: Record<string, string> = {};
    let m: RegExpExecArray | null;
    while ((m = re.exec(settingsPart)) !== null) {
      const key = m[1].trim();
      let val = m[2].trim();
      // Strip surrounding quotes if present.
      if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1, -1);
      kv[key] = val;
    }
    const num = (s: string | undefined): number | null => {
      if (s == null) return null;
      const n = Number(s);
      return Number.isFinite(n) ? n : null;
    };
    out.steps = num(kv["Steps"]);
    out.cfg = num(kv["CFG scale"]) ?? num(kv["CFG"]);
    out.seed = num(kv["Seed"]);
    out.sampler = kv["Sampler"] ?? null;
    // Forge uses "Schedule type"; older A1111 used "Sampler" baked
    // with "Karras" suffix. Surface both.
    out.scheduler = kv["Schedule type"] ?? kv["Scheduler"] ?? null;
  }

  // Pull <lora:name:weight> tokens out of the positive prompt and
  // expose them as loras slots. The applier later maps these names
  // to the current workflow's LoraLoader node ids by basename match.
  if (out.positivePrompt) {
    const loraRe = /<lora:([^:>]+):([\d.\-]+)>/g;
    let lm: RegExpExecArray | null;
    while ((lm = loraRe.exec(out.positivePrompt)) !== null) {
      const name = lm[1];
      const weight = Number(lm[2]);
      if (Number.isFinite(weight)) {
        out.loras.push({ nodeId: name, label: name, value: weight });
      }
    }
    // Strip the LoRA tokens from the visible positive prompt — we
    // applied them via the slider stack instead.
    out.positivePrompt = out.positivePrompt
      .replace(loraRe, "")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  // Civitai-hosted resource references (urn:air:…) won't resolve on
  // a local ComfyUI install. Strip them from both prompts so the
  // user sees a clean form and ComfyUI doesn't error on missing
  // embeddings at submit time.
  if (out.positivePrompt) out.positivePrompt = stripCivitaiURNRefs(out.positivePrompt);
  if (out.negativePrompt) out.negativePrompt = stripCivitaiURNRefs(out.negativePrompt);

  return out;
}

// metaFromRaw turns a tEXt-keyword map into a typed ParsedMeta.
// Supports two tEXt-chunk dialects:
//   - ComfyUI: `prompt` (and optionally `workflow`) are JSON
//     graphs. Failed JSON parses become null.
//   - A1111 / Forge / Civitai: `parameters` is a single text chunk
//     in the positive + negative + Steps:… format. parseA1111Parameters
//     handles it and returns an ExtractedParams populated directly.
function metaFromRaw(raw: Record<string, string>, chunkTypes: string[] = []): ParsedMeta {
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

  let prompt = tryParse("prompt");
  const workflow = tryParse("workflow");
  let a1111Source = raw["parameters"] || raw["Description"] || raw["Comment"] || "";

  // JPEG EXIF UserComment (Civitai's preferred metadata slot for
  // JPEG outputs) can hold any of three payload shapes — sometimes
  // mixed in one document:
  //   - A ComfyUI graph (top-level keys are node ids → {class_type,
  //     inputs, _meta})
  //   - A1111 plain-text "<positive>\nNegative prompt: …\nSteps: …"
  //   - A Civitai mixed JSON: {<graph nodes>, "extra": {"airs":
  //     [...], "extraMetadata": "{…JSON-encoded a1111 fields…}"}}
  //
  // Try them in order: graph first (richest); fall back to
  // extraMetadata; final fallback is plain text.
  const uc = raw["UserComment"];
  if (uc && !prompt) {
    // Layer 1 — try parsing as well-formed JSON. Strip leading/
    // trailing nulls or junk that some encoders leave behind.
    const cleaned = stripJsonGarbage(uc);
    if (cleaned) {
      try {
        const obj = JSON.parse(cleaned);
        if (obj && typeof obj === "object") {
          const looksLikeGraph = Object.values(obj).some(
            (v) =>
              v != null && typeof v === "object" && "class_type" in (v as Record<string, unknown>),
          );
          if (looksLikeGraph) prompt = obj as WorkflowGraph;
          const extra = (obj as Record<string, unknown>)["extra"];
          if (
            extra &&
            typeof extra === "object" &&
            (extra as Record<string, unknown>)["extraMetadata"]
          ) {
            const em = safeJsonParse((extra as Record<string, unknown>)["extraMetadata"] as string);
            const synth = synthesizeA1111FromCivitaiMeta(em);
            if (synth && !a1111Source) a1111Source = synth;
          }
        }
      } catch {
        // fall through to Layer 2
      }
    }

    // Layer 2 — JSON.parse may have failed (mixed encoding, trailing
    // image bytes, broken escapes). Regex out the fields we need
    // directly from the raw text. Catches Civitai's
    // {"…","extraMetadata":"{\"prompt\":\"…\"…}"} shape even when
    // the surrounding JSON is too garbled to parse cleanly.
    if (!prompt && !a1111Source) {
      const fallback = extractParamsByRegex(uc);
      if (fallback) a1111Source = fallback;
    }

    // Layer 3 — last resort, treat the whole UserComment as A1111
    // plain text. Cheap; parseA1111Parameters returns null if it
    // doesn't see "Negative prompt:" or "Steps:".
    if (!prompt && !a1111Source) a1111Source = uc;
  }

  const params = prompt ? extractParamsFromGraph(prompt) : parseA1111Parameters(a1111Source);
  return { prompt, workflow, params, raw, chunkTypes };
}

// stripJsonGarbage trims leading/trailing characters that aren't
// part of the JSON document (NULs, non-printable bytes, occasional
// trailing image data that bled into the UserComment buffer when
// the EXIF count was wrong).
function stripJsonGarbage(s: string): string {
  // Strip leading/trailing NULs and BOM.
  let t = s.replace(/^[ ﻿\s]+/, "").replace(/[ \s]+$/, "");
  // If it starts with `{`, find the matching close brace by depth.
  if (t.startsWith("{")) {
    let depth = 0;
    let inString = false;
    let escape = false;
    for (let i = 0; i < t.length; i++) {
      const c = t[i];
      if (escape) {
        escape = false;
        continue;
      }
      if (c === "\\" && inString) {
        escape = true;
        continue;
      }
      if (c === '"') {
        inString = !inString;
        continue;
      }
      if (inString) continue;
      if (c === "{") depth++;
      else if (c === "}") {
        depth--;
        if (depth === 0) return t.slice(0, i + 1);
      }
    }
  }
  return t;
}

function safeJsonParse(s: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(s);
    return v && typeof v === "object" ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

// synthesizeA1111FromCivitaiMeta turns a Civitai extraMetadata
// object ({prompt, negativePrompt, steps, cfgScale, sampler, seed})
// into an A1111-format text the existing parser already knows how
// to read.
function synthesizeA1111FromCivitaiMeta(em: Record<string, unknown> | null): string | null {
  if (!em) return null;
  const lines: string[] = [];
  if (typeof em.prompt === "string") lines.push(em.prompt);
  if (typeof em.negativePrompt === "string") lines.push(`Negative prompt: ${em.negativePrompt}`);
  const settings: string[] = [];
  if (em.steps != null) settings.push(`Steps: ${em.steps}`);
  if (em.sampler != null) settings.push(`Sampler: ${em.sampler}`);
  if (em.cfgScale != null) settings.push(`CFG scale: ${em.cfgScale}`);
  else if (em.cfg != null) settings.push(`CFG scale: ${em.cfg}`);
  if (em.seed != null) settings.push(`Seed: ${em.seed}`);
  if (typeof em.workflowId === "string") settings.push(`Workflow: ${em.workflowId}`);
  if (settings.length > 0) lines.push(settings.join(", "));
  return lines.length > 0 ? lines.join("\n") : null;
}

// extractParamsByRegex is the fallback that runs when the
// surrounding JSON is too damaged to parse but the field substrings
// are still readable in raw bytes. Pulls Civitai's `extraMetadata`
// fields (prompt, negativePrompt, steps, cfgScale, sampler, seed)
// directly via regex, then assembles an A1111-shaped string.
//
// Doesn't try to be a real JSON parser — purely substring-match
// for `"key":"value"` and `"key":number` patterns. The extracted
// strings still have JSON escapes (\\u0022, \", etc.) which we
// unescape via JSON.parse on the wrapped fragment.
function extractParamsByRegex(text: string): string | null {
  const decode = (raw: string) => {
    try {
      // Wrap in quotes so JSON.parse handles all standard escapes.
      return JSON.parse(`"${raw}"`) as string;
    } catch {
      return raw;
    }
  };
  const strField = (key: string): string | null => {
    // \"key\" : \" … (non-greedy, allowing escaped quotes)
    const re = new RegExp(`"${key}"\\s*:\\s*"((?:\\\\.|[^"\\\\])*)"`, "i");
    const m = text.match(re);
    return m ? decode(m[1]) : null;
  };
  const numField = (key: string): number | null => {
    const re = new RegExp(`"${key}"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)`, "i");
    const m = text.match(re);
    return m ? Number(m[1]) : null;
  };
  const prompt = strField("prompt");
  const neg = strField("negativePrompt");
  if (!prompt && !neg) return null;
  const lines: string[] = [];
  if (prompt) lines.push(prompt);
  if (neg) lines.push(`Negative prompt: ${neg}`);
  const settings: string[] = [];
  const steps = numField("steps");
  if (steps != null) settings.push(`Steps: ${steps}`);
  const sampler = strField("sampler");
  if (sampler) settings.push(`Sampler: ${sampler}`);
  const cfg = numField("cfgScale") ?? numField("cfg");
  if (cfg != null) settings.push(`CFG scale: ${cfg}`);
  const seed = numField("seed");
  if (seed != null) settings.push(`Seed: ${seed}`);
  if (settings.length > 0) lines.push(settings.join(", "));
  return lines.join("\n");
}

// parseEmbeddedMetadata pulls the JSON-shaped tEXt chunks ComfyUI
// writes (`prompt`, `workflow`) and returns them as parsed graphs.
// Failed JSON parses are treated as missing — callers see null and
// render "no metadata."
async function parseEmbeddedMetadata(dataUrl: string): Promise<ParsedMeta> {
  const { chunks, types } = await parseDataUrl(dataUrl);
  return metaFromRaw(chunks, types);
}

// parseEmbeddedMetadataFromFile reads a dropped File as bytes and
// parses its text chunks. Async because file reading + iTXt
// decompression both are. Returns null on a non-PNG or read failure
// so the caller can render a friendly "no metadata" message.
async function parseEmbeddedMetadataFromFile(file: File): Promise<ParsedMeta | null> {
  try {
    const buf = await file.arrayBuffer();
    const { chunks, types } = await parseImageMetadataFromBytes(new Uint8Array(buf));
    return metaFromRaw(chunks, types);
  } catch {
    return null;
  }
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
  // Pre-populated SFW anchors — Pony bases (cyberrealistic, etc.)
  // pull hard toward NSFW without an explicit clothing/nudity
  // negative. Default these in the Studio UI rather than baking into
  // the shipped workflow JSON so users who want NSFW can clear the
  // field on a per-run basis without forking a workflow. Replay
  // overwrites this when an image with embedded metadata is dropped,
  // matching expectation (the source image's negative wins).
  @state() private negativePrompt = "";
  @state() private seed = 42;
  @state() private randomizeSeed = true;

  // Common KSampler knobs (apply to every KSampler in the workflow)
  @state() private steps = 30;
  @state() private cfg = 7.5;
  @state() private sampler = "dpmpp_2m";
  @state() private scheduler = "karras";

  // Denoise: the img2img stylization-amount knob. 0.3 preserves the
  // source closely; 0.7+ regenerates substantially. Only matters
  // when sourceImage is set (text-to-image runs at denoise=1 by
  // default, baked into the workflow's KSampler). Pre-seeded from
  // the style preset's recommendation when the user picks a style;
  // the slider reflects the active value and the user can override.
  @state() private denoise = 0.55;

  // Batch
  @state() private numImages = 1;

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

  // Drag-drop state. dragOver shows the dropzone overlay while a drag
  // is happening; dropMessage surfaces the outcome (success or "no
  // metadata" / wrong file type) for a few seconds after the drop.
  @state() private dragOver = false;
  @state() private dropMessage: string | null = null;

  // Style presets (Simpsons / Clone Wars / etc.) loaded from
  // images.styles.list. selectedStyleId="" means "no style applied"
  // (raw prompt, no suffix appended).
  @state() private styles: StylePreset[] = [];
  @state() private selectedStyleId = "";

  // Installed LoRA filenames sourced from images.objectInfo. Used to
  // tell whether a style preset's lora.filename is already on disk;
  // missing entries trigger the install button. Empty before the
  // first fetch — UI treats that as "unknown, don't surface install
  // yet" rather than "definitely missing".
  @state() private installedLoras: Set<string> | null = null;
  @state() private managerStatus: ManagerStatus | null = null;
  @state() private installInFlight = false;
  @state() private installMessage: string | null = null;

  // Img2img source image, set after a successful images.upload.
  // Surfaces a thumbnail in the controls column and feeds the
  // LoadImage nodeOverride at submit time. uploadInFlight blocks the
  // generate button while the upload is happening so users can't
  // submit a workflow that references a file ComfyUI hasn't received.
  @state() private sourceImage: SourceImage | null = null;
  @state() private uploadInFlight = false;
  @state() private uploadError: string | null = null;

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
      // Default to bat-country (vyx with a leaner permanent
      // negative + goth/cozy style anchors). Falls back to vyx,
      // then any builtin, then the first entry — so the UI always
      // has a selection on first paint regardless of registry.
      const preferred = ["bat-country", "vyx"];
      const pick = preferred.map((id) => this.workflows.find((w) => w.id === id)).find(Boolean);
      const initial =
        pick ?? this.workflows.find((w) => w.source === "builtin") ?? this.workflows[0];
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

  // loadStyles fetches the prompt-style preset registry. Failures
  // leave styles empty (the picker just doesn't appear) — the rest of
  // the UI still works without style anchors.
  private async loadStyles() {
    if (!this.client) return;
    try {
      const res = await this.client.request<{ styles: StylePreset[] }>("images.styles.list", {});
      this.styles = res?.styles ?? [];
    } catch (err) {
      // Non-fatal; styles are an additive feature.
      console.warn("studio: loadStyles failed", err);
      this.styles = [];
    }
  }

  // loadManagerStatus probes the active ComfyUI for the Manager
  // extension. Used to gate the install-button affordance: when the
  // manager is present, click installs via images.manager.install;
  // when absent, the UI falls back to opening the civitai page in a
  // new tab. Failures are treated as "manager absent" — surfacing a
  // probe error to the user adds noise without value.
  private async loadManagerStatus() {
    if (!this.client) return;
    try {
      const res = await this.client.request<ManagerStatus>("images.manager.status", {});
      this.managerStatus = res ?? { present: false };
    } catch (err) {
      console.warn("studio: loadManagerStatus failed", err);
      this.managerStatus = { present: false };
    }
  }

  // loadInstalledLoras fetches images.objectInfo and parses the
  // LoraLoader enum into a Set of filenames. Cached on the
  // installedLoras state field so the picker can decide install
  // visibility without a per-style refetch. Heavy (1-3MB on a
  // fully-loaded ComfyUI) so we only call it once on connect and
  // again after a successful install.
  private async loadInstalledLoras() {
    if (!this.client) return;
    try {
      const res = await this.client.request<{ objectInfo: Record<string, unknown> }>(
        "images.objectInfo",
        {},
      );
      const info = res?.objectInfo ?? {};
      this.installedLoras = extractLoraFilenames(info);
    } catch (err) {
      console.warn("studio: loadInstalledLoras failed", err);
      // Stay null on failure so the UI shows neither "installed" nor
      // "missing" — both are misleading without ground truth.
      this.installedLoras = null;
    }
  }

  // installLora hands a style.lora requirement off to ComfyUI-Manager.
  // The manager queues the download; we don't poll for completion
  // (that would require a long-running RPC contract we don't have
  // yet). Users see "queued" and refresh the page when the LoRA
  // shows up. If the manager isn't present, falls back to opening
  // the civitai page so the user can download manually.
  private async installLora(lora: LoraRequirement) {
    if (!lora.civitai?.downloadUrl) {
      this.installMessage = "no download URL on this preset; install manually";
      return;
    }
    if (!this.managerStatus?.present) {
      // No manager — open civitai page in new tab as the fallback.
      const target = lora.civitai.page || lora.civitai.downloadUrl;
      window.open(target, "_blank", "noopener,noreferrer");
      this.installMessage = "opened CivitAI page (manager not installed)";
      return;
    }
    if (!this.client) return;
    this.installInFlight = true;
    this.installMessage = null;
    try {
      const res = await this.client.request<{ ok: boolean; message?: string }>(
        "images.manager.install",
        {
          type: "loras",
          url: lora.civitai.downloadUrl,
          filename: lora.filename,
          savePath: "loras",
        },
      );
      if (res?.ok) {
        this.installMessage = res.message
          ? `install queued: ${res.message}`
          : "install queued — refresh in a minute to use the LoRA";
      } else {
        this.installMessage = "manager rejected install request";
      }
      // Refresh the installed-LoRA cache so the badge flips quickly
      // if the manager streamed the install synchronously.
      void this.loadInstalledLoras();
    } catch (err) {
      this.installMessage = `install failed: ${err}`;
    } finally {
      this.installInFlight = false;
    }
  }

  // uploadSourceImage reads `file` as bytes, base64-encodes, and POSTs
  // to images.upload. On success, sourceImage is populated with the
  // resolved ComfyUI filename + a local data: URL preview. Errors are
  // surfaced inline; the run isn't blocked because the user might
  // legitimately want to clear the upload and submit text-only.
  private async uploadSourceImage(file: File) {
    if (!this.client) return;
    if (!file.type.startsWith("image/")) {
      this.uploadError = "Only image files supported";
      return;
    }
    this.uploadInFlight = true;
    this.uploadError = null;
    try {
      const buf = await file.arrayBuffer();
      // Browser-side base64: chunked btoa so a multi-MB image doesn't
      // blow the call stack via spread-into-fromCharCode.
      const bytes = new Uint8Array(buf);
      let binary = "";
      const CHUNK = 0x8000;
      for (let i = 0; i < bytes.length; i += CHUNK) {
        binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
      }
      const base64 = btoa(binary);

      // Generate a stable-ish filename so re-uploads of the same file
      // don't pile up in ComfyUI's input dir. ComfyUI auto-suffixes
      // on collision when overwrite=false; we set overwrite=true so
      // a re-upload of the same logical file replaces in place.
      const name = file.name || `studio-upload-${Date.now()}.png`;
      const res = await this.client.request<{
        filename: string;
        subfolder: string;
        type: string;
      }>("images.upload", {
        filename: name,
        base64,
        contentType: file.type || "application/octet-stream",
        overwrite: true,
      });
      if (!res?.filename) {
        this.uploadError = "upload returned no filename";
        return;
      }
      // Build a local preview from the bytes we already have, no need
      // for a round-trip back through images.fetch.
      const previewBlob = new Blob([buf], { type: file.type || "image/png" });
      const dataUrl = URL.createObjectURL(previewBlob);
      this.sourceImage = { filename: res.filename, dataUrl };
    } catch (err) {
      this.uploadError = `upload failed: ${err}`;
    } finally {
      this.uploadInFlight = false;
    }
  }

  // useAsSource re-uploads the currently-viewed gallery image as the
  // next img2img source so the user can iterate restyles without
  // re-picking a file. Reads bytes from the viewer's data URL (which
  // images.fetch already populated), POSTs to images.upload, then
  // sets sourceImage to the resolved ComfyUI filename. The viewer
  // closes on success so the focus returns to the controls column.
  private async useAsSource() {
    if (!this.client || !this.viewer || !this.viewer.dataUrl) return;
    const ref = this.viewer.ref;
    this.uploadInFlight = true;
    this.uploadError = null;
    try {
      // Decode the data URL (base64) into raw bytes. dataUrl shape:
      // "data:<mime>;base64,<payload>". Anything else (rare) we treat
      // as an unsupported source.
      const m = /^data:([^;]+);base64,(.*)$/.exec(this.viewer.dataUrl);
      if (!m) {
        this.uploadError = "viewer data URL not base64; cannot reuse";
        return;
      }
      const contentType = m[1] || "image/png";
      const binary = atob(m[2]);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

      // Re-encode for transport. atob → bytes → btoa is wasteful but
      // keeps the upload path identical to the file-picker flow,
      // which the gateway side has tests for.
      let chunked = "";
      const CHUNK = 0x8000;
      for (let i = 0; i < bytes.length; i += CHUNK) {
        chunked += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
      }
      const base64 = btoa(chunked);

      // Tag the upload so re-using the same gallery image multiple
      // times doesn't collide. ComfyUI auto-suffixes with overwrite=
      // false, but a stable name + overwrite=true keeps the input
      // dir tidy across iterations.
      const name = `studio-reuse-${ref.filename}`;
      const res = await this.client.request<{
        filename: string;
        subfolder: string;
        type: string;
      }>("images.upload", {
        filename: name,
        base64,
        contentType,
        overwrite: true,
      });
      if (!res?.filename) {
        this.uploadError = "upload returned no filename";
        return;
      }
      const previewBlob = new Blob([bytes], { type: contentType });
      const dataUrl = URL.createObjectURL(previewBlob);
      this.sourceImage = { filename: res.filename, dataUrl };
      // Auto-switch the workflow picker to the img2img builtin if
      // the user is currently on a text-to-image workflow. They
      // clicked "Use as source" — they want img2img. Without this
      // the upload sits unused because the active workflow has no
      // LoadImage node to receive it.
      if (this.findLoadImageNodeId() === null) {
        const img2img = this.workflows.find((w) => w.id === "img2img-pony");
        if (img2img) {
          this.selectedWorkflowId = img2img.id;
          await this.loadWorkflowGraph(img2img.id);
        }
      }
      // Close the viewer so focus returns to the controls — the user
      // will want to tweak prompt/denoise for the next run, and
      // staying in the lightbox would block that.
      this.closeViewer();
    } catch (err) {
      this.uploadError = `upload failed: ${err}`;
    } finally {
      this.uploadInFlight = false;
    }
  }

  // clearSourceImage drops the staged upload + revokes the object URL
  // so the browser doesn't leak the in-memory blob across sessions.
  private clearSourceImage() {
    if (this.sourceImage?.dataUrl?.startsWith("blob:")) {
      try {
        URL.revokeObjectURL(this.sourceImage.dataUrl);
      } catch {
        // best-effort
      }
    }
    this.sourceImage = null;
    this.uploadError = null;
  }

  // findLoadImageNodeId scans the active graph for the first node
  // whose class_type is LoadImage. The img2img workflow uses one;
  // IPAdapter-based workflows may also surface here, which is fine —
  // the same upload picker drives both source-image override paths.
  private findLoadImageNodeId(): string | null {
    if (!this.currentGraph) return null;
    for (const [nid, node] of Object.entries(this.currentGraph)) {
      if (node?.class_type === "LoadImage") {
        return nid;
      }
    }
    return null;
  }

  private async loadWorkflowGraph(id: string) {
    if (!this.client) return;
    // Random sentinel: no graph to fetch. Clear introspection so
    // the LoRA/IPAdapter panels collapse and the form falls back
    // to KSampler-only common knobs (which apply to whatever
    // workflow each iteration picks).
    if (id === RANDOM_WORKFLOW_ID) {
      this.currentGraph = null;
      this.loraSlots = [];
      this.ipaSlots = [];
      return;
    }
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
    // map captures the full intent. When sourceImage is set (img2img
    // mode), also pin denoise to the user's slider value — t2i
    // KSamplers run at denoise=1 in the workflow JSON, so we leave
    // them alone in that mode (slider is hidden anyway).
    for (const [nid, node] of Object.entries(this.currentGraph)) {
      const cls = node?.class_type ?? "";
      if (cls === "KSampler" || cls === "KSamplerAdvanced") {
        const seedKey = cls === "KSamplerAdvanced" ? "noise_seed" : "seed";
        const knobs: Record<string, unknown> = {
          steps: this.steps,
          cfg: this.cfg,
          sampler_name: this.sampler,
          scheduler: this.scheduler,
          [seedKey]: seedForRun,
        };
        if (this.sourceImage) {
          knobs.denoise = this.denoise;
        }
        overrides[nid] = knobs;
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
    // Img2img source image: route the uploaded filename to the
    // workflow's first LoadImage node. The patcher silently drops
    // overrides for nodes that don't exist, so this is safe even
    // when the active workflow is t2i (no LoadImage at all).
    if (this.sourceImage) {
      const loadImageNid = this.findLoadImageNodeId();
      if (loadImageNid) {
        overrides[loadImageNid] = {
          ...(overrides[loadImageNid] ?? {}),
          image: this.sourceImage.filename,
        };
      }
    }
    return overrides;
  }

  private async submit() {
    if (!this.client) {
      this.genState = { kind: "error", message: "gateway client not connected" };
      return;
    }
    const userPrompt = this.prompt.trim();
    if (!userPrompt) {
      this.genState = { kind: "error", message: "prompt is required" };
      return;
    }
    if (this.uploadInFlight) {
      this.genState = { kind: "error", message: "wait for upload to finish" };
      return;
    }
    // Style suffix application: append the preset's promptSuffix to
    // the user's prompt and negativeSuffix to the negative. We don't
    // mutate this.prompt — keep the user's edit surface clean and
    // recompute the augmented version per-submit so toggling styles
    // doesn't accumulate suffixes.
    const style = this.styles.find((s) => s.id === this.selectedStyleId);
    const prompt = style ? userPrompt + style.promptSuffix : userPrompt;
    const userNegative = this.negativePrompt.trim();
    const negative = style?.negativeSuffix
      ? userNegative
        ? userNegative + style.negativeSuffix
        : style.negativeSuffix.replace(/^,\s*/, "")
      : userNegative;
    this.genState = { kind: "submitting" };
    this.attachListener();

    // Pre-pick the workflow for each iteration. Normal mode reuses
    // the user's selection; Random sentinel rolls a fresh builtin
    // per image so a count=10 batch produces 10 different
    // styles/checkpoints.
    const count = Math.max(1, Math.floor(this.numImages));
    const isRandom = this.selectedWorkflowId === RANDOM_WORKFLOW_ID;
    const builtinIds = this.workflows
      .filter((w) => w.source === "builtin" && w.id !== RANDOM_WORKFLOW_ID)
      .map((w) => w.id);
    const pickWorkflow = (): string => {
      if (!isRandom) return this.selectedWorkflowId;
      if (builtinIds.length === 0) return this.selectedWorkflowId;
      return builtinIds[Math.floor(Math.random() * builtinIds.length)];
    };

    // Submit the batch. Each call gets its own seed (random or
    // sequential); the server creates a runId per call and we listen
    // for events on all of them simultaneously.
    for (let i = 0; i < count; i++) {
      const s = this.randomizeSeed
        ? Math.floor(Math.random() * 0x7fffffff)
        : Math.floor(this.seed) + i;
      const wfId = pickWorkflow();
      // Random mode skips nodeOverrides for LoRA/IPAdapter — those
      // are workflow-specific and don't transfer cleanly across
      // templates. KSampler knobs (steps/cfg/sampler/scheduler)
      // still go through because they're universal.
      const overrides = isRandom ? this.buildKSamplerOverrides(s) : this.buildNodeOverrides(s);
      try {
        const res = await this.client.request<{ runId: string }>("images.generate", {
          sessionKey: this.sessionKey,
          prompt,
          negativePrompt: negative || undefined,
          seed: s,
          workflowId: wfId || undefined,
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

  // buildKSamplerOverrides is a slimmer version of
  // buildNodeOverrides for Random mode: applies the common KSampler
  // knobs (steps/cfg/sampler/scheduler/seed) to every KSampler in
  // whichever workflow gets picked, but skips LoRA/IPAdapter slots
  // since those don't map across templates. nodeOverrides for nodes
  // the chosen workflow doesn't have are silently dropped server-
  // side, so this is safe to apply blindly.
  private buildKSamplerOverrides(seedForRun: number): Record<string, Record<string, unknown>> {
    // Without a graph (random mode doesn't introspect), we don't
    // know which node ids are KSamplers. The server-side patcher
    // applies overrides only to known nodes and silently skips
    // unknowns, so we send the common-knob bundle keyed by every
    // small integer id (3, 5, 9, 11, 15, 18 are the KSampler ids
    // across the bundled workflows). Cheap and correct.
    const knobs = {
      steps: this.steps,
      cfg: this.cfg,
      sampler_name: this.sampler,
      scheduler: this.scheduler,
      seed: seedForRun,
    };
    const overrides: Record<string, Record<string, unknown>> = {};
    for (const id of ["3", "5", "9", "11", "15", "18"]) {
      overrides[id] = { ...knobs };
    }
    return overrides;
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
      // Parse PNG text chunks once on arrival. parseEmbeddedMetadata
      // tolerates non-PNGs and absent chunks (returns nulls), so the
      // lightbox always has a stable shape to render against. Async
      // because iTXt may be zlib-compressed.
      const metadata = url ? await parseEmbeddedMetadata(url) : null;
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

  // applyMetadata pulls every knob the Studio surface exposes out of
  // an embedded prompt graph and pushes the values into the form.
  // Used by both the lightbox Replay button and the drag-drop entry
  // point — they share the same destination state, so they share the
  // same applier. The user reviews and clicks Generate themselves —
  // never auto-fire a run because the human almost always wants to
  // tweak something before re-rolling.
  private applyMetadata(metadata: ParsedMeta) {
    const p = metadata.params;
    if (!p) return;

    // ComfyUI source: graph available, signature-match against
    // shipped templates and switch the dropdown if a match exists.
    // A1111 source: no graph, keep the current workflow selection
    // (the user picked it; we apply prompts/settings into it).
    if (metadata.prompt) {
      void this.findAndSelectWorkflow(graphSignature(metadata.prompt));
    }

    if (p.positivePrompt != null) this.prompt = p.positivePrompt;
    if (p.negativePrompt != null) this.negativePrompt = p.negativePrompt;
    if (p.seed != null) {
      this.seed = p.seed;
      this.randomizeSeed = false; // exact-seed replay is the point
    }
    if (p.steps != null) this.steps = p.steps;
    if (p.cfg != null) this.cfg = p.cfg;
    if (p.sampler) this.sampler = this.normalizeSampler(p.sampler);
    if (p.scheduler) this.scheduler = this.normalizeScheduler(p.scheduler);

    // LoRA application has two paths:
    //   - ComfyUI graph: p.loras' nodeId IS the workflow's node id;
    //     apply by direct nodeId match.
    //   - A1111 parameters: p.loras' nodeId is the lora NAME (we
    //     stuffed the name there since a parameters chunk has no
    //     graph). Match against current loraSlots' label/lora_name
    //     basename so the slider picks up the new weight.
    if (p.loras.length > 0) {
      const fromGraph = metadata.prompt != null;
      if (fromGraph) {
        const byId = new Map(p.loras.map((l) => [l.nodeId, l.value]));
        this.loraSlots = this.loraSlots.map((s) =>
          byId.has(s.nodeId) ? { ...s, value: byId.get(s.nodeId) as number } : s,
        );
      } else {
        const byName = new Map(p.loras.map((l) => [l.label.toLowerCase(), l.value]));
        this.loraSlots = this.loraSlots.map((s) => {
          // Match the slot's label against any incoming LoRA name
          // (case-insensitive substring or basename overlap). LoRA
          // labels in the bundled workflows include both vanity
          // titles ("ExpressiveH (Pony)") and filenames; either can
          // line up with the A1111 token name.
          for (const [name, weight] of byName) {
            if (
              s.label.toLowerCase().includes(name) ||
              name.includes(s.label.toLowerCase().split(" ")[0])
            ) {
              return { ...s, value: weight };
            }
          }
          return s;
        });
      }
    }
    if (p.ipas.length > 0 && metadata.prompt != null) {
      const byId = new Map(p.ipas.map((l) => [l.nodeId, l.value]));
      this.ipaSlots = this.ipaSlots.map((s) =>
        byId.has(s.nodeId) ? { ...s, value: byId.get(s.nodeId) as number } : s,
      );
    }
  }

  // normalizeSampler maps A1111-style sampler names to ComfyUI
  // sampler ids when they differ. A1111 uses display names like
  // "DPM++ 2M SDE", ComfyUI uses "dpmpp_2m_sde". Untouched when
  // already in ComfyUI form.
  private normalizeSampler(s: string): string {
    const t = s.trim().toLowerCase();
    const map: Record<string, string> = {
      euler: "euler",
      "euler a": "euler_ancestral",
      "euler ancestral": "euler_ancestral",
      "dpm++ 2m": "dpmpp_2m",
      "dpm++ 2m sde": "dpmpp_2m_sde",
      "dpm++ 3m sde": "dpmpp_3m_sde",
      "dpm 2": "dpm_2",
      deis: "deis",
      "uni pc": "uni_pc",
      lcm: "lcm",
    };
    // Strip trailing scheduler-ish words sometimes packed in (older
    // A1111 formats had things like "DPM++ 2M Karras" as the
    // sampler).
    const stripped = t
      .replace(/\b(karras|exponential|sgm uniform|simple|ddim uniform)$/i, "")
      .trim();
    return map[stripped] ?? map[t] ?? s;
  }

  // normalizeScheduler does the same for A1111's "Schedule type"
  // values which are TitleCase.
  private normalizeScheduler(s: string): string {
    const t = s.trim().toLowerCase();
    const map: Record<string, string> = {
      karras: "karras",
      normal: "normal",
      exponential: "exponential",
      "sgm uniform": "sgm_uniform",
      simple: "simple",
      "ddim uniform": "ddim_uniform",
      automatic: "karras", // Forge default
    };
    return map[t] ?? s;
  }

  private replayFromViewer() {
    if (!this.viewer?.metadata) return;
    this.applyMetadata(this.viewer.metadata);
    this.closeViewer();
  }

  // handleDroppedFile reads a dropped file (PNG expected) and applies
  // its embedded metadata to the form. Surfaces drop status in
  // dropMessage so the user knows whether the file had usable
  // metadata or whether it was, say, a JPG without ComfyUI chunks.
  private async handleDroppedFile(file: File) {
    this.dragOver = false;
    if (!file.type.startsWith("image/")) {
      this.dropMessage = `dropped file is ${file.type || "unknown type"}, expected image/png`;
      return;
    }
    const meta = await parseEmbeddedMetadataFromFile(file);
    if (!meta?.params) {
      this.dropMessage = `${file.name}: ${this.formatNoMetadataMessage(meta)}`;
      return;
    }
    this.applyMetadata(meta);
    this.dropMessage = `populated from ${file.name}`;
    // Clear the success message after a beat so the controls panel
    // doesn't carry a stale "loaded from foo.png" forever.
    window.setTimeout(() => {
      if (this.dropMessage?.startsWith("populated from")) {
        this.dropMessage = null;
      }
    }, 4000);
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

  // deleteCurrent removes the currently-viewed image from disk + the
  // talon image index via images.delete, then drops it from the
  // gallery state and closes the lightbox. No confirmation: the
  // image is auto-saved + recoverable from ComfyUI history (a re-run
  // is one click away), so a confirm dialog is more friction than
  // protection.
  private async deleteCurrent() {
    if (!this.client || !this.viewer) return;
    const ref = this.viewer.ref;
    try {
      await this.client.request<{ ok: boolean; removed: number }>("images.delete", {
        filename: ref.filename,
        subfolder: ref.subfolder ?? "",
      });
      this.gallery = this.gallery.filter((g) => g.ref.filename !== ref.filename);
      this.closeViewer();
    } catch (err) {
      // Stay in the lightbox on failure so the user sees the error
      // message rather than silently confused-empty.
      this.viewer = {
        ...this.viewer,
        error: `delete failed: ${err}`,
      };
    }
  }

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
    void this.loadStyles();
    void this.loadManagerStatus();
    void this.loadInstalledLoras();
    window.addEventListener("keydown", this.onKeyDown);
    // Drag-drop listeners go on window — not the .wrap div — because
    // the browser's default behavior is to navigate to the dropped
    // file. If we only listen inside the shadow DOM, dropping
    // anywhere outside that subtree (or releasing during a brief
    // miss) hands the file to the browser and it opens the image
    // in the tab. Window-level listeners with preventDefault keep
    // the file ours.
    window.addEventListener("dragenter", this.onWindowDragEnter);
    window.addEventListener("dragover", this.onWindowDragOver);
    window.addEventListener("dragleave", this.onWindowDragLeave);
    window.addEventListener("drop", this.onWindowDrop);
  }

  override disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("dragenter", this.onWindowDragEnter);
    window.removeEventListener("dragover", this.onWindowDragOver);
    window.removeEventListener("dragleave", this.onWindowDragLeave);
    window.removeEventListener("drop", this.onWindowDrop);
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
  static override styles = css`
    :host {
      /* Fill the .content area like every other tab. Padding is
         handled by .content's own 16/20/32; we only own the
         interior. */
      display: block;
      width: 100%;
      color: var(--text, #d6dce8);
    }
    /* Card chrome matching the global .card rule from
       components.css (--card bg, --border border,
       --radius-lg corner). Replicated here because Lit shadow DOM
       doesn't inherit the global class — uses the same CSS
       variables so light/dark theme switches flow through. */
    .wrap {
      display: grid;
      /* 1fr / 2fr → controls take ~1/3, gallery ~2/3. minmax floor
         keeps the controls usable on mid-narrow viewports before
         the media-query collapse kicks in. */
      grid-template-columns: minmax(320px, 1fr) 2fr;
      gap: 1.25rem;
      align-items: start;
      position: relative; /* anchor for the dropzone overlay */
      background: var(--card, #11141c);
      border: 1px solid var(--border, #1a1f2c);
      border-radius: var(--radius-lg, 12px);
      padding: 1.25rem;
    }
    /* Adaptive: under ~900px the two-column grid collapses to a
       single column so controls stack above the gallery instead of
       getting squashed below their 320px min. Same breakpoint
       layout.mobile.css uses elsewhere. */
    @media (max-width: 900px) {
      .wrap {
        grid-template-columns: 1fr;
      }
    }
    /* Dropzone: covers the entire viewport (position: fixed) while a
       file is being dragged in. Window-level drag/drop listeners
       catch drops anywhere on the page, so the overlay should
       reflect that — not just the Studio panel. Inert
       (pointer-events: none) so it doesn't interfere with the
       browser's native drag/drop event flow. */
    .dropzone {
      position: fixed;
      inset: 0;
      background: rgba(95, 168, 255, 0.08);
      border: 2px dashed var(--accent, #5fa8ff);
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
      z-index: 10000;
    }
    .dropzone-card {
      background: var(--bg-elevated, #14171f);
      border: 1px solid var(--border, #1a1f2c);
      border-radius: 0.5rem;
      padding: 1rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      align-items: center;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    }
    .dropzone-card strong {
      color: var(--text-strong, #f4f7fb);
      font-size: 1rem;
    }
    .dropzone-card span {
      color: var(--muted, #7a8398);
      font-size: 0.8rem;
    }
    .status.drop-msg {
      color: var(--accent, #5fa8ff);
    }
    .controls {
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
    }
    /* img2img source-image picker: thumbnail must stay small so a
       full-res phone photo doesn't push the rest of the controls
       column off the visible viewport. The flex-row layout keeps
       file input + thumbnail + filename + clear button on one line.
       max-width on the thumb fences anything over 64px wide. */
    .source-image-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .source-thumb {
      max-width: 64px;
      max-height: 64px;
      width: auto;
      height: auto;
      object-fit: cover;
      border-radius: 4px;
      border: 1px solid var(--border, #2a2f3a);
    }
    /* Denoise slider sits in its own group so the explanatory hint
       is visually attached to the slider, not floating with the
       label below it. */
    .denoise-row {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    /* LoRA install hint stacks the requirement text + install button
       + status message vertically so a long civitai filename doesn't
       wrap awkwardly across the row. */
    .lora-install {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      margin-top: 0.25rem;
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
    /* Slider row: range + small number input share one line BELOW
       the label. Label gets its own full-width row above (the
       parent <label> is already display:flex column). Range flexes
       to fill; number is fixed-width and tabular-nums so the column
       doesn't jitter while dragging. */
    .slider-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .slider-row input[type="range"] {
      flex: 1 1 auto;
      min-width: 0;
    }
    .value-num {
      flex: 0 0 4.5rem;
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-size: 0.8rem;
      padding: 0.2rem 0.45rem;
      background: var(--input, #1a1f2c);
      color: var(--text-strong, #f4f7fb);
      border: 1px solid var(--border, #1a1f2c);
      border-radius: 0.25rem;
    }
    .value-num:focus {
      outline: 0;
      border-color: var(--ring, #5fa8ff);
      box-shadow:
        0 0 0 2px var(--bg, #07080c),
        0 0 0 3px var(--ring, #5fa8ff);
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
    /* Strip native number-input spinner arrows on every variant
       (the inline .value-num next to sliders + the Seed and Number
       of Images fields). The slider already covers fine
       adjustment; the type/clamp behavior of the number input is
       what we want, the arrows are visual noise. */
    input[type="number"] {
      appearance: textfield;
      -moz-appearance: textfield;
    }
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
      -webkit-appearance: none;
      appearance: none;
      margin: 0;
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
    .viewer-controls button.danger {
      border-color: rgba(239, 68, 68, 0.7);
      color: #fca5a5;
    }
    .viewer-controls button.danger:hover {
      background: rgba(239, 68, 68, 0.85);
      color: white;
      border-color: rgba(239, 68, 68, 0.85);
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
      gap: 0.6rem;
    }
    /* Each metadata row: key sits above value (form-style label
       layout) so long prompts/negatives don't squash the value into
       a narrow right column. */
    .viewer-metadata .meta-summary > div {
      display: flex;
      flex-direction: column;
      gap: 0.15rem;
    }
    .viewer-metadata .meta-summary .k {
      color: rgba(255, 255, 255, 0.6);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.7rem;
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

  // --- drag/drop -------------------------------------------------------
  //
  // Listeners are wired on `window` (not the .wrap div) so a drop
  // anywhere on the page is caught and prevented from triggering the
  // browser's default "open file in tab" behavior. Two source kinds
  // are supported:
  //
  //   1. Local file drops (Finder, Downloads, etc.) — arrive in
  //      e.dataTransfer.files. Direct ArrayBuffer read.
  //   2. Cross-window image drags (dragging from another tab/window)
  //      — arrive as URLs in e.dataTransfer.getData("text/uri-list")
  //      or "text/html". Need a fetch round-trip; subject to CORS.
  //
  // dragOverDepth is a counter for nested dragenter/dragleave events
  // (the browser fires leave when crossing inner element boundaries).
  // We only hide the overlay when the depth hits zero.

  private dragOverDepth = 0;

  private dataTransferHasUsefulPayload(dt: DataTransfer | null): boolean {
    if (!dt) return false;
    if (Array.from(dt.items ?? []).some((it) => it.kind === "file")) return true;
    // types is a DOMStringList in some browsers — `Array.from` wraps it
    // either way.
    const types = Array.from(dt.types ?? []);
    return (
      types.includes("Files") || types.includes("text/uri-list") || types.includes("text/html")
    );
  }

  private onWindowDragEnter = (e: DragEvent) => {
    if (!this.dataTransferHasUsefulPayload(e.dataTransfer)) return;
    e.preventDefault();
    this.dragOverDepth++;
    this.dragOver = true;
  };

  private onWindowDragOver = (e: DragEvent) => {
    if (!this.dataTransferHasUsefulPayload(e.dataTransfer)) return;
    // preventDefault on dragover is what tells the browser "I'll
    // handle this drop"; without it the drop event never fires and
    // the browser navigates to the file/URL.
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
  };

  private onWindowDragLeave = (e: DragEvent) => {
    if (!this.dataTransferHasUsefulPayload(e.dataTransfer)) return;
    this.dragOverDepth = Math.max(0, this.dragOverDepth - 1);
    if (this.dragOverDepth === 0) this.dragOver = false;
  };

  private onWindowDrop = async (e: DragEvent) => {
    if (!this.dataTransferHasUsefulPayload(e.dataTransfer)) return;
    e.preventDefault();
    this.dragOverDepth = 0;
    this.dragOver = false;
    const dt = e.dataTransfer;
    if (!dt) return;
    // 1. Local file drop (Finder/Downloads) — direct ArrayBuffer read.
    const files = Array.from(dt.files ?? []);
    if (files.length > 0) {
      await this.handleDroppedFile(files[0]);
      return;
    }
    // 2. Cross-window drag — extract a URL and fetch.
    const url = this.extractUrlFromDataTransfer(dt);
    if (url) {
      await this.handleDroppedUrl(url);
      return;
    }
    this.dropMessage = "drop a PNG file or a ComfyUI image";
  };

  private extractUrlFromDataTransfer(dt: DataTransfer): string | null {
    // text/uri-list is the cleanest source — newline-separated URIs
    // with `#`-prefixed comments.
    const uri = dt.getData("text/uri-list");
    if (uri) {
      const lines = uri
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter((l) => l && !l.startsWith("#"));
      if (lines.length > 0) return lines[0];
    }
    // text/plain sometimes carries the URL alone (Firefox does this
    // when dragging the page-image's address bar entry).
    const plain = dt.getData("text/plain")?.trim();
    if (plain && /^(data|https?|blob):/i.test(plain)) return plain;
    // text/html: dragging an <img> from another window tends to give
    // us the source HTML for that element. Pull the first src out.
    const html = dt.getData("text/html");
    if (html) {
      const m = html.match(/<img[^>]+src=["']([^"']+)["']/i);
      if (m) return m[1];
    }
    return null;
  }

  private async handleDroppedUrl(url: string) {
    this.dropMessage = `loading from ${this.summarizeUrl(url)}…`;
    try {
      if (url.startsWith("data:")) {
        // Same-doc data URL — reuse the base64 path.
        const meta = await parseEmbeddedMetadata(url);
        if (!meta.params) {
          this.dropMessage = this.formatNoMetadataMessage(meta);
          return;
        }
        this.applyMetadata(meta);
        this.dropMessage = "populated from dropped image";
        this.scheduleDropMessageClear();
        return;
      }
      // External URL: fetch as binary. CORS may block; report cleanly.
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      const { chunks, types } = await parseImageMetadataFromBytes(new Uint8Array(buf));
      const meta = metaFromRaw(chunks, types);
      if (!meta.params) {
        this.dropMessage = this.formatNoMetadataMessage(meta);
        return;
      }
      this.applyMetadata(meta);
      this.dropMessage = "populated from dropped image";
      this.scheduleDropMessageClear();
    } catch (err) {
      // Most common: CORS denial when dragging from another origin.
      // The browser doesn't tell us which, so the message hedges and
      // suggests the workaround that always works (save → drop file).
      this.dropMessage = `couldn't read dropped URL (CORS or fetch failed) — save the image, then drop the file. ${err}`;
    }
  }

  // formatNoMetadataMessage builds the diagnostic shown when a drop
  // produced no usable params. Surfaces both the text-chunk keys
  // (so we know whether "parameters" was present-but-unparseable
  // vs absent) and every PNG chunk type (so we can tell whether the
  // file was a stripped PNG vs a non-PNG vs had compressed text we
  // can't decode).
  private formatNoMetadataMessage(meta: ParsedMeta | null): string {
    if (!meta) return "no readable prompt metadata (couldn't parse file)";
    const textKeys = Object.keys(meta.raw ?? {});
    const types = meta.chunkTypes ?? [];
    const interesting = types.filter((t) => t === "tEXt" || t === "iTXt" || t === "zTXt");
    if (textKeys.length > 0) {
      // Show a snippet so we can see what was actually decoded.
      // Most useful key first — UserComment (JPEG EXIF) wins over
      // generic stuff like "Description".
      const order = ["UserComment", "parameters", "Description", "Comment", "prompt", "workflow"];
      const sampled = order.find((k) => textKeys.includes(k)) ?? textKeys[0];
      const sample = (meta.raw[sampled] ?? "").slice(0, 200).replace(/\s+/g, " ");
      // Also dump the full content to the console for copy-paste
      // back if the snippet isn't enough to diagnose.
      console.warn(
        "studio: unrecognized metadata chunk",
        sampled,
        "length=",
        (meta.raw[sampled] ?? "").length,
        "\nfull:\n",
        meta.raw[sampled],
      );
      return `no readable prompt metadata (chunk "${sampled}" decoded but unrecognized; first 200 chars: ${sample}…)`;
    }
    if (interesting.length > 0) {
      return `no readable prompt metadata (text chunks ${interesting.join(", ")} were present but couldn't be decoded)`;
    }
    if (types.length > 0) {
      return `no readable prompt metadata (saw PNG chunks ${types.slice(0, 6).join(", ")}${types.length > 6 ? ", …" : ""}; no text chunks at all — Civitai may have stripped them)`;
    }
    return "no readable prompt metadata (file isn't a PNG or has no chunks)";
  }

  private summarizeUrl(url: string): string {
    if (url.startsWith("data:")) return "data:…";
    try {
      const u = new URL(url);
      return u.host + u.pathname.split("/").slice(-1)[0];
    } catch {
      return url.slice(0, 64);
    }
  }

  private scheduleDropMessageClear() {
    window.setTimeout(() => {
      if (this.dropMessage?.startsWith("populated")) this.dropMessage = null;
    }, 4000);
  }

  override render() {
    if (!this.client) {
      return html`<div class="wrap"><div>gateway not connected</div></div>`;
    }
    return html`
      <div class="wrap">
        <div class="controls">${this.renderControls()}</div>
        <div>${this.renderGallery()}</div>
        ${this.dragOver
          ? html`<div class="dropzone">
              <div class="dropzone-card">
                <strong>Drop image to load metadata</strong>
                <span>Embedded ComfyUI prompt + workflow → form</span>
              </div>
            </div>`
          : nothing}
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
          <option
            .value=${RANDOM_WORKFLOW_ID}
            ?selected=${this.selectedWorkflowId === RANDOM_WORKFLOW_ID}
          >
            🎲 Random — pick a fresh workflow per image
          </option>
          ${this.workflows.map(
            (w) =>
              html`<option .value=${w.id} ?selected=${w.id === this.selectedWorkflowId}>
                ${w.label}
              </option>`,
          )}
        </select>
      </label>

      ${this.safeRender("source-image", () => this.renderSourceImage())}
      ${this.safeRender("denoise", () => this.renderDenoiseSlider())}
      ${this.safeRender("style", () => this.renderStylePicker())}

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
            @change=${(e: Event) => {
              const n = Number((e.target as HTMLInputElement).value);
              if (Number.isFinite(n)) this.seed = n;
            }}
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
          @change=${(e: Event) => {
            // @change fires only on blur/Enter, after the user
            // commits the typed value. Using @input would clobber
            // mid-typing values like "1" → "10" with intermediate
            // partial parses.
            const n = Number((e.target as HTMLInputElement).value);
            this.numImages = Number.isFinite(n) && n > 0 ? Math.floor(n) : 1;
          }}
        />
      </label>

      ${this.selectedWorkflowId === RANDOM_WORKFLOW_ID
        ? html`
            <div class="group">
              <h3>Random mode</h3>
              <div class="status">
                Each image draws a fresh builtin workflow. LoRA + IPAdapter sliders are
                workflow-specific and don't carry across — they're hidden in this mode. Prompt,
                negative, seed, steps, CFG, sampler, scheduler still apply.
              </div>
            </div>
          `
        : nothing}
      ${this.selectedWorkflowId !== RANDOM_WORKFLOW_ID && this.loraSlots.length > 0
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
      ${this.selectedWorkflowId !== RANDOM_WORKFLOW_ID && this.ipaSlots.length > 0
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
      ${this.dropMessage ? html`<div class="status drop-msg">${this.dropMessage}</div>` : nothing}
    `;
  }

  // safeRender wraps a render-helper call in a try/catch so a thrown
  // exception in one section can't blank the whole controls column.
  // Surfaces the error inline (instead of bubbling up and corrupting
  // the parent template) and logs to the console so the actual stack
  // trace is recoverable. Used for the new img2img-related sections
  // since they're the most recently churned and most likely to break.
  private safeRender(label: string, fn: () => unknown): unknown {
    try {
      return fn();
    } catch (err) {
      console.error(`studio: ${label} render threw`, err);
      return html`<div class="status error">[${label}] ${String(err)}</div>`;
    }
  }

  // renderSourceImage shows the file picker, in-flight indicator,
  // upload error (if any), and a thumbnail + clear button when an
  // image is staged. Surfaces only when the active workflow has a
  // LoadImage node — otherwise the upload would be a no-op since
  // nothing in the graph would consume it.
  private renderSourceImage() {
    const hasLoadImage = this.findLoadImageNodeId() !== null;
    if (!hasLoadImage && !this.sourceImage) return nothing;
    return html`
      <label class="source-image">
        <span>Source image (img2img)</span>
        <div class="source-image-row">
          <input
            type="file"
            accept="image/*"
            ?disabled=${this.uploadInFlight}
            @change=${(e: Event) => {
              const f = (e.target as HTMLInputElement).files?.[0];
              if (f) void this.uploadSourceImage(f);
            }}
          />
          ${this.uploadInFlight ? html`<span class="muted">uploading…</span>` : nothing}
          ${this.sourceImage
            ? html`
                <img class="source-thumb" src=${this.sourceImage.dataUrl} alt="source" />
                <span class="muted">${this.sourceImage.filename}</span>
                <button type="button" @click=${() => this.clearSourceImage()}>Clear</button>
              `
            : nothing}
        </div>
        ${this.uploadError ? html`<div class="status error">${this.uploadError}</div>` : nothing}
      </label>
    `;
  }

  // renderDenoiseSlider shows the img2img stylization-amount knob.
  // Only surfaces when sourceImage is set (text-to-image runs at
  // denoise=1 by default and the slider would have no effect there).
  // Snaps to the picked style's recommended value on style change;
  // user moves win after.
  private renderDenoiseSlider() {
    if (!this.sourceImage) return nothing;
    return html`
      <div class="denoise-row">
        ${this.renderSliderRow("Denoise (stylization)", this.denoise, 0, 1, 0.05, (v) => {
          this.denoise = v;
        })}
        <small class="muted">
          0.3 preserves the source closely; 0.7+ regenerates substantially.
        </small>
      </div>
    `;
  }

  // renderStylePicker is a flat dropdown of style presets. "(none)"
  // is the empty-id sentinel so the user can opt out without
  // toggling state. Description shows below the row when a preset
  // is selected so the user knows what suffix is about to be
  // appended. When the active preset declares a LoRA requirement,
  // a sub-row surfaces install state: "Installed" / "Install via
  // Manager" button / "Open in CivitAI" fallback.
  private renderStylePicker() {
    if (this.styles.length === 0) return nothing;
    const active = this.styles.find((s) => s.id === this.selectedStyleId);
    return html`
      <label>
        <span>Style</span>
        <select
          .value=${this.selectedStyleId}
          @change=${(e: Event) => {
            this.selectedStyleId = (e.target as HTMLSelectElement).value;
            this.installMessage = null;
            // Snap the denoise slider to the preset's recommendation
            // when the user picks a style. They can still override
            // afterward — the slider is the source of truth at
            // submit time.
            const picked = this.styles.find((s) => s.id === this.selectedStyleId);
            if (picked) {
              this.denoise = picked.denoise;
            }
          }}
        >
          <option value="" ?selected=${!this.selectedStyleId}>(none)</option>
          ${this.styles.map(
            (s) =>
              html`<option .value=${s.id} ?selected=${s.id === this.selectedStyleId}>
                ${s.label}
              </option>`,
          )}
        </select>
        ${active?.description ? html`<small class="muted">${active.description}</small>` : nothing}
        ${this.renderLoraInstallState(active)}
      </label>
    `;
  }

  // renderLoraInstallState shows the install-button affordance for
  // the active style's LoRA requirement. Three branches:
  //   - LoRA present in installedLoras → green "Installed" badge.
  //   - LoRA missing AND manager detected → "Install via Manager" button.
  //   - LoRA missing AND manager absent → "Open in CivitAI" link.
  // installedLoras=null (probe hasn't run) suppresses everything so
  // we don't flash a stale state on first paint.
  private renderLoraInstallState(active: StylePreset | undefined) {
    const lora = active?.lora;
    if (!lora) return nothing;
    if (this.installedLoras === null) {
      return html`<small class="muted">checking ${lora.filename}…</small>`;
    }
    const installed = this.installedLoras.has(lora.filename);
    if (installed) {
      return html`<small class="muted">✓ ${lora.filename} installed</small>`;
    }
    const managerOK = this.managerStatus?.present === true;
    return html`
      <div class="lora-install">
        <small class="muted">requires ${lora.filename} (not installed)</small>
        ${managerOK
          ? html`<button
              type="button"
              ?disabled=${this.installInFlight}
              @click=${() => void this.installLora(lora)}
            >
              ${this.installInFlight ? "queuing…" : "Install via Manager"}
            </button>`
          : lora.civitai?.page || lora.civitai?.downloadUrl
            ? html`<button type="button" @click=${() => void this.installLora(lora)}>
                Open in CivitAI
              </button>`
            : html`<small class="muted">no install URL on this preset</small>`}
        ${this.installMessage ? html`<small class="muted">${this.installMessage}</small>` : nothing}
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
    // Both the range slider and the number input write through the
    // same apply() so they stay in lockstep. apply clamps to
    // [min, max] before forwarding so the underlying state never
    // drifts past the slider's visual range.
    //
    // Two fire modes:
    //   - The range slider uses @input — every drag step pushes a
    //     fresh value, and range inputs always emit a valid number
    //     so partial-typing doesn't apply.
    //   - The number input uses @change instead of @input. @change
    //     fires only on blur or Enter, after the user finishes
    //     typing. Using @input here means typing "0.5" fires three
    //     keystroke events ("0", "0.", "0.5"), and the intermediate
    //     "0." parses as 0 — clobbering state mid-typing. Worse,
    //     when typing inside an existing value like "0.72", a
    //     misplaced "." or partial edit can produce "0.7.5" → NaN
    //     or "27" → clamped to 2 on a [-2, 2] slider. @change
    //     waits for the committed value, so the user gets to type
    //     the whole number undisturbed.
    const apply = (raw: string) => {
      const n = Number(raw);
      if (!Number.isFinite(n)) return;
      onChange(Math.min(max, Math.max(min, n)));
    };
    return html`
      <label>
        <span>${label}</span>
        <div class="slider-row">
          <input
            type="range"
            .min=${String(min)}
            .max=${String(max)}
            .step=${String(step)}
            .value=${String(value)}
            @input=${(e: Event) => apply((e.target as HTMLInputElement).value)}
          />
          <input
            class="value-num"
            type="number"
            .min=${String(min)}
            .max=${String(max)}
            .step=${String(step)}
            .value=${this.formatNumber(value)}
            @change=${(e: Event) => apply((e.target as HTMLInputElement).value)}
          />
        </div>
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
    // Replayable when EITHER a ComfyUI graph or A1111 params chunk
    // gave us extracted params. Drag-drops from civitai-A1111 sources
    // hit the params path; drag-drops from talon's auto-saved PNGs
    // hit the graph path.
    const replayable = metadata?.params != null;
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
          <button
            ?disabled=${!dataUrl || this.uploadInFlight}
            title="Re-upload this image as the img2img source for the next run"
            @click=${() => void this.useAsSource()}
          >
            ${this.uploadInFlight ? "uploading…" : "Use as source"}
          </button>
          <button
            class="danger"
            title="Delete this image from disk + gallery"
            @click=${() => void this.deleteCurrent()}
          >
            Delete
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
    if (!metadata?.params) return nothing;
    const p = metadata.params;
    // Raw payload depends on source: ComfyUI graphs render as JSON,
    // A1111 params chunks render as the raw text exactly as embedded.
    const rawDisplay = metadata.prompt
      ? JSON.stringify(metadata.prompt, null, 2)
      : (metadata.raw["parameters"] ?? "");
    const sourceLabel = metadata.prompt ? "comfyui" : "a1111";
    return html`
      <details class="viewer-metadata" @click=${(e: Event) => e.stopPropagation()}>
        <summary>metadata · ${sourceLabel}</summary>
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
          ${p.steps != null || p.cfg != null || p.sampler || p.scheduler
            ? html`<div>
                <span class="k">sampler</span>
                <span class="v"
                  >${[
                    p.sampler ?? "?",
                    p.scheduler ? `/${p.scheduler}` : "",
                    p.steps != null ? ` · ${p.steps} steps` : "",
                    p.cfg != null ? ` · cfg ${p.cfg}` : "",
                  ].join("")}</span
                >
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
                await navigator.clipboard.writeText(rawDisplay);
              } catch (err) {
                console.warn("studio: clipboard write failed", err);
              }
            }}
          >
            Copy ${metadata.prompt ? "JSON" : "Parameters"}
          </button>
          <pre>${rawDisplay}</pre>
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

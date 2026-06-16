import { AutoRouter, error } from "itty-router";
import { Llm, Redis, Variables } from "@fermyon/spin-sdk";

// ---------------------------------------------------------------------------
// Few-shot inference over Aiven for Valkey
//
// Flow for POST /infer:
//   1. Embed the incoming prompt with all-minilm-l6-v2 (384-dim vector).
//   2. KNN-search Valkey's HNSW index for the 3 most similar PAST SUCCESSFUL
//      inferences (cosine distance).
//   3. Inject those as few-shot examples into the llama2-chat prompt.
//   4. Return the completion.
//
// POST /feedback lets a caller mark a (prompt, completion) pair as successful,
// which embeds it and writes it into the example bank so future queries can
// retrieve it. This is what makes the demo improve over time.
// ---------------------------------------------------------------------------

const EMBED_MODEL = "all-minilm-l6-v2";
// Chat completion is served by any OpenAI-compatible /v1/chat/completions
// endpoint, configured at deploy time via Spin variables (see spin.toml). The
// same request/response shape works across providers — point inference_api_url
// at whichever you want:
//   Ollama:    http://localhost:11434/v1/chat/completions   (no key)
//   OpenAI:    https://api.openai.com/v1/chat/completions    (sk-... key)
//   Anthropic: https://api.anthropic.com/v1/chat/completions (Anthropic key,
//              model e.g. "claude-opus-4-8" — OpenAI-compatible layer)
// The host must be listed in spin.toml's allowed_outbound_hosts.
const INDEX_NAME = "idx:examples";
const KEY_PREFIX = "example:";
const VECTOR_DIM = 384; // all-minilm-l6-v2 output dimension
const TOP_K = 3;

// Semantic response cache. A second HNSW index, separate from the example bank
// so cached raw answers never leak into few-shot retrieval. A request is a
// cache hit when a past entry's input is similar enough AND it was generated
// from the same retrieved few-shot examples (see examplesFingerprint).
const CACHE_INDEX = "idx:cache";
const CACHE_PREFIX = "cache:";

// Semantic router. A third HNSW index holding a handful of labeled exemplar
// utterances per intent (see scripts/routes.json). On each /infer we embed the
// input ONCE (the same vector few-shot retrieval and the cache already need),
// KNN-search this index for the nearest exemplar, and take its `route` label.
// That label selects a tailored system prompt so the LLM gets intent-specific
// instructions before it ever runs. If the nearest exemplar isn't similar
// enough (below route_threshold), we fall back to the GENERAL route.
const ROUTE_INDEX = "idx:routes";
const ROUTE_PREFIX = "route:";
const GENERAL_ROUTE = "general";

// Per-route system prompts. The router picks one of these by intent; the
// default GENERAL prompt is used both for the fallback route and for any route
// label that lacks a dedicated entry, so adding new routes.json labels never
// breaks the lookup. Keep each instruction short — it's prepended to the
// few-shot conversation built in buildMessages().
const ROUTE_PROMPTS: Record<string, string> = {
  general:
    "You are a helpful assistant. The conversation below contains past " +
    "requests and the responses that worked well — match their tone and " +
    "format. Reply with only the response itself, no preamble.",
  "schedule-meeting":
    "You draft concise meeting requests. Propose a specific duration and time " +
    "window, ask the recipient to confirm, and offer to send a calendar invite.",
  "reminder-deadline":
    "You write short, friendly deadline reminders. State what is due and when, " +
    "and offer help if the recipient is blocked. Keep it warm, not nagging.",
  "follow-up":
    "You write post-meeting follow-ups. Thank the recipient, recap what was " +
    "agreed, name the next step, and invite corrections.",
  "out-of-office":
    "You write out-of-office auto-replies. State the away period, point urgent " +
    "matters to a colleague, and promise a reply on return. No greeting line.",
  agenda:
    "You produce tight meeting agendas: a one-line title, a short numbered list " +
    "of items, and the attendees. No prose around the list.",
  "decline-invite":
    "You politely decline meeting invitations. Thank the sender, cite a " +
    "scheduling conflict, and offer to find another time or share notes async.",
  "meeting-minutes":
    "You write meeting minutes: title, attendees, key points and decisions, and " +
    "action items with owners and due dates. Be factual and compact.",
  "request-approval":
    "You write approval requests. Clearly state what needs review, attach or " +
    "reference the details, and give a deadline that keeps work on schedule.",
  "thank-you":
    "You write brief, sincere thank-you notes. Name the specific help given and " +
    "why it mattered. Warm and genuine, never effusive.",
  "status-update":
    "You write quick status updates: whether things are on track, milestones " +
    "hit, next steps, and a promise to flag risks early. One short paragraph.",
  reschedule:
    "You write reschedule requests. Apologize briefly, propose a concrete " +
    "alternative time, and thank the recipient for their flexibility.",
  "submit-report":
    "You write reminders to submit a report. State the report and due date, say " +
    "where to file it, and ask the recipient to flag blockers.",
  "confirm-appointment":
    "You write appointment confirmations. Restate the topic and time, invite " +
    "changes, and close on a forward-looking note.",
  announcement:
    "You write brief team announcements. Lead with what is changing, say what it " +
    "means for the reader and when it takes effect, and invite questions.",
  "request-documents":
    "You write requests for documents. Say which documents you need, for what, " +
    "and by when, and thank the recipient for their help.",
};

// Resolve a route label to its system prompt, defaulting to GENERAL.
function routePrompt(route: string): string {
  return ROUTE_PROMPTS[route] ?? ROUTE_PROMPTS[GENERAL_ROUTE];
}

// Pull the Valkey connection string from a Spin variable (see spin.toml).
function valkeyUrl(): string {
  const url = Variables.get("valkey_url");
  if (!url) {
    console.error("[config] valkey_url variable is not set");
    throw new Error("valkey_url variable is not set");
  }
  return url;
}

// Inference endpoint config, all from Spin variables so the same build can
// target Ollama, OpenAI, or Anthropic without code changes.
function inferenceApiUrl(): string {
  const url = Variables.get("inference_api_url");
  if (!url) {
    console.error("[config] inference_api_url variable is not set");
    throw new Error("inference_api_url variable is not set");
  }
  return url;
}
function chatModel(): string {
  const model = Variables.get("chat_model");
  if (!model) {
    console.error("[config] chat_model variable is not set");
    throw new Error("chat_model variable is not set");
  }
  return model;
}
// Optional: a bearer token for hosted providers (OpenAI, Anthropic). Local
// Ollama needs none, so this is allowed to be empty.
function inferenceApiKey(): string {
  return Variables.get("inference_api_key") ?? "";
}

// Minimum cosine similarity for a cached response to count as a hit. High by
// default so only near-identical questions reuse an answer.
function cacheThreshold(): number {
  const v = parseFloat(Variables.get("cache_threshold") ?? "");
  return Number.isFinite(v) ? v : 0.95;
}
// Cache entry lifetime in seconds (0 disables expiry). Default 1 hour.
function cacheTtlSeconds(): number {
  const v = parseInt(Variables.get("cache_ttl_seconds") ?? "", 10);
  return Number.isFinite(v) && v >= 0 ? v : 3600;
}

// Minimum cosine similarity for the router to trust the nearest exemplar. Below
// this the input doesn't clearly belong to any known intent, so we fall back to
// the GENERAL route. Default is moderate so clear intents route and vague ones
// don't get force-fit into a specialist prompt.
function routeThreshold(): number {
  const v = parseFloat(Variables.get("route_threshold") ?? "");
  return Number.isFinite(v) ? v : 0.6;
}

// Embed a single piece of text and return its float vector.
function embed(text: string): number[] {
  const result = Llm.generateEmbeddings(EMBED_MODEL, [text]);
  return result.embeddings[0];
}

// Serialize a JS float array into the little-endian float32 byte buffer that
// the valkey-search KNN query (and HSET on the vector field) expects.
function floatsToBytes(floats: number[]): Uint8Array {
  const buf = new ArrayBuffer(floats.length * 4);
  const view = new DataView(buf);
  floats.forEach((f, i) => view.setFloat32(i * 4, f, true /* little-endian */));
  return new Uint8Array(buf);
}

// Spin's Redis.execute takes tagged RedisParameter values, not bare strings,
// and returns tagged RedisResult values. These helpers keep the call sites
// readable.
const enc = new TextEncoder();
const dec = new TextDecoder();

type RedisParam = { tag: "binary"; val: Uint8Array } | { tag: "int64"; val: bigint };

// A string argument (Redis is binary-safe; command verbs and text args ride as
// binary payloads).
function arg(s: string): RedisParam {
  return { tag: "binary", val: enc.encode(s) };
}
// A raw bytes argument (e.g. the float32 vector blob).
function bytes(b: Uint8Array): RedisParam {
  return { tag: "binary", val: b };
}

// Decode any RedisResult element to a string for parsing.
function resultToString(r: { tag: string; val?: unknown }): string {
  switch (r.tag) {
    case "binary":
      return dec.decode(r.val as Uint8Array);
    case "status":
      return r.val as string;
    case "int64":
      return String(r.val as bigint);
    default:
      return "";
  }
}

// FT.CREATE errors if the index already exists. Swallow exactly that failure so
// index creation is idempotent; rethrow anything else. Spin throws a tagged
// error object whose human-readable message lives in `.payload.val` (and
// sometimes `.val`), not in String(e), so check all.
function ignoreAlreadyExists(e: unknown): void {
  const err = e as { val?: unknown; payload?: { val?: unknown } };
  const msg = `${err?.payload?.val ?? ""} ${err?.val ?? ""} ${String(e)}`;
  if (!msg.includes("already exists")) throw e;
}

// Ensure the HNSW vector index for the few-shot example bank exists.
function ensureIndex(conn: ReturnType<typeof Redis.open>): void {
  try {
    conn.execute("FT.CREATE", [
      arg(INDEX_NAME),
      arg("ON"), arg("HASH"),
      arg("PREFIX"), arg("1"), arg(KEY_PREFIX),
      arg("SCHEMA"),
      arg("prompt"), arg("TEXT"),
      arg("completion"), arg("TEXT"),
      arg("embedding"), arg("VECTOR"), arg("HNSW"), arg("6"),
      arg("TYPE"), arg("FLOAT32"),
      arg("DIM"), arg(String(VECTOR_DIM)),
      arg("DISTANCE_METRIC"), arg("COSINE"),
    ]);
  } catch (e) {
    ignoreAlreadyExists(e);
  }
}

// Ensure the HNSW vector index for the semantic response cache exists. Same
// vector setup as the example bank, plus a `fingerprint` TAG so a KNN search
// can be scoped to entries built from the same few-shot examples.
function ensureCacheIndex(conn: ReturnType<typeof Redis.open>): void {
  try {
    conn.execute("FT.CREATE", [
      arg(CACHE_INDEX),
      arg("ON"), arg("HASH"),
      arg("PREFIX"), arg("1"), arg(CACHE_PREFIX),
      arg("SCHEMA"),
      arg("input"), arg("TEXT"),
      arg("completion"), arg("TEXT"),
      arg("fingerprint"), arg("TAG"),
      arg("embedding"), arg("VECTOR"), arg("HNSW"), arg("6"),
      arg("TYPE"), arg("FLOAT32"),
      arg("DIM"), arg(String(VECTOR_DIM)),
      arg("DISTANCE_METRIC"), arg("COSINE"),
    ]);
  } catch (e) {
    ignoreAlreadyExists(e);
  }
}

// Ensure the HNSW vector index for the semantic router exists. Each entry is a
// labeled exemplar utterance: the `route` field holds the intent label we hand
// back, the vector lets us KNN-search for the nearest exemplar to an input.
function ensureRouteIndex(conn: ReturnType<typeof Redis.open>): void {
  try {
    conn.execute("FT.CREATE", [
      arg(ROUTE_INDEX),
      arg("ON"), arg("HASH"),
      arg("PREFIX"), arg("1"), arg(ROUTE_PREFIX),
      arg("SCHEMA"),
      arg("utterance"), arg("TEXT"),
      arg("route"), arg("TEXT"),
      arg("embedding"), arg("VECTOR"), arg("HNSW"), arg("6"),
      arg("TYPE"), arg("FLOAT32"),
      arg("DIM"), arg(String(VECTOR_DIM)),
      arg("DISTANCE_METRIC"), arg("COSINE"),
    ]);
  } catch (e) {
    ignoreAlreadyExists(e);
  }
}

interface Example {
  prompt: string;
  completion: string;
  score: number; // cosine distance; lower = more similar
}

function toExample(map: Record<string, string>): Example {
  return {
    prompt: map["prompt"] ?? "",
    completion: map["completion"] ?? "",
    score: parseFloat(map["score"] ?? "1"),
  };
}

// Run a KNN search for the TOP_K nearest stored examples to `queryVec`.
function searchExamples(
  conn: ReturnType<typeof Redis.open>,
  queryVec: number[],
): Example[] {
  const raw = conn.execute("FT.SEARCH", [
    arg(INDEX_NAME),
    arg(`*=>[KNN ${TOP_K} @embedding $vec AS score]`),
    arg("PARAMS"), arg("2"), arg("vec"), bytes(floatsToBytes(queryVec)),
    arg("RETURN"), arg("3"), arg("prompt"), arg("completion"), arg("score"),
    arg("DIALECT"), arg("2"),
  ]);

  // Spin flattens RESP arrays, so the reply is a single RedisResult[]:
  //   [ total, key1, field, val, field, val, ..., key2, field, val, ... ]
  // We don't know how many fields each doc returns up front, so we walk the
  // list, treating any "example:" key as a record boundary.
  const flat = raw.map(resultToString);
  const examples: Example[] = [];
  let current: Record<string, string> | null = null;

  for (let i = 1; i < flat.length; i++) {
    const tok = flat[i];
    if (tok.startsWith(KEY_PREFIX)) {
      if (current) examples.push(toExample(current));
      current = {};
      continue;
    }
    if (current) {
      current[tok] = flat[i + 1] ?? "";
      i++; // consumed the value
    }
  }
  if (current) examples.push(toExample(current));
  return examples;
}

interface RouteMatch {
  route: string;      // the chosen intent label
  similarity: number; // cosine similarity to the nearest exemplar (0..1)
  matched: boolean;   // false when we fell back to GENERAL below threshold
}

// Classify an input by finding the nearest labeled exemplar in idx:routes.
// Reuses the query vector already computed for few-shot retrieval, so routing
// adds one KNN search and zero extra embedding calls. Returns the GENERAL route
// (matched:false) when the index is empty or the best match is too weak.
function pickRoute(
  conn: ReturnType<typeof Redis.open>,
  queryVec: number[],
): RouteMatch {
  let raw;
  try {
    raw = conn.execute("FT.SEARCH", [
      arg(ROUTE_INDEX),
      arg(`*=>[KNN 1 @embedding $vec AS score]`),
      arg("PARAMS"), arg("2"), arg("vec"), bytes(floatsToBytes(queryVec)),
      arg("RETURN"), arg("2"), arg("route"), arg("score"),
      arg("DIALECT"), arg("2"),
    ]);
  } catch {
    // Empty index or query miss — no routing info, use GENERAL.
    return { route: GENERAL_ROUTE, similarity: 0, matched: false };
  }

  const flat = raw.map(resultToString);
  let route: string | null = null;
  let score = 1;
  for (let i = 1; i < flat.length; i++) {
    if (flat[i] === "route") route = flat[i + 1] ?? "";
    if (flat[i] === "score") score = parseFloat(flat[i + 1] ?? "1");
  }

  const similarity = Number((1 - score).toFixed(4)); // cosine distance -> sim
  if (!route || similarity < routeThreshold()) {
    return { route: GENERAL_ROUTE, similarity, matched: false };
  }
  return { route, similarity, matched: true };
}

// Store one labeled exemplar utterance into the router index. Keyed by route +
// utterance content so re-seeding updates in place rather than duplicating.
function storeRoute(
  conn: ReturnType<typeof Redis.open>,
  route: string,
  utterance: string,
): string {
  const vec = embed(utterance);
  const id = `${ROUTE_PREFIX}${route}:${hash(utterance)}`;
  conn.execute("HSET", [
    arg(id),
    arg("utterance"), arg(utterance),
    arg("route"), arg(route),
    arg("embedding"), bytes(floatsToBytes(vec)),
  ]);
  return id;
}

// Count how many examples are currently in the bank. valkey-search rejects a
// bare "*" query, so we read num_docs from FT.INFO instead. The reply is a
// flat [field, value, field, value, ...] list; find num_docs and take the next.
function countDocs(conn: ReturnType<typeof Redis.open>, index: string): number {
  const raw = conn.execute("FT.INFO", [arg(index)]).map(resultToString);
  const i = raw.indexOf("num_docs");
  return i >= 0 ? parseInt(raw[i + 1], 10) || 0 : 0;
}

// Store a successful (prompt, completion) pair as a retrievable example.
function storeExample(
  conn: ReturnType<typeof Redis.open>,
  prompt: string,
  completion: string,
): string {
  const vec = embed(prompt);
  // A stable-ish id from the prompt content so re-submitting updates in place.
  const id = `${KEY_PREFIX}${hash(prompt)}`;
  conn.execute("HSET", [
    arg(id),
    arg("prompt"), arg(prompt),
    arg("completion"), arg(completion),
    arg("embedding"), bytes(floatsToBytes(vec)),
  ]);
  return id;
}

// Tiny non-crypto string hash (FNV-1a) for deterministic example keys.
function hash(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16);
}

// A fingerprint of the few-shot examples a completion was generated from. The
// cache only reuses an entry if the SAME examples would be retrieved again, so
// adding/removing examples from the bank doesn't serve stale answers. Order is
// stable (KNN returns nearest-first), so we hash the example keys in order.
function examplesFingerprint(examples: Example[]): string {
  return hash(examples.map((e) => e.prompt).join(" ")) || "none";
}

interface CacheHit {
  completion: string;
  similarity: number;
}

// Look for a cached completion semantically close to `input` AND built from the
// same few-shot examples (fingerprint). Returns the hit only if similarity
// meets the threshold. valkey-search needs a TAG prefilter syntax of the form
// `@field:{value}` combined with the KNN clause.
function lookupCache(
  conn: ReturnType<typeof Redis.open>,
  queryVec: number[],
  fingerprint: string,
): CacheHit | null {
  let raw;
  try {
    raw = conn.execute("FT.SEARCH", [
      arg(CACHE_INDEX),
      arg(`@fingerprint:{${fingerprint}}=>[KNN 1 @embedding $vec AS score]`),
      arg("PARAMS"), arg("2"), arg("vec"), bytes(floatsToBytes(queryVec)),
      arg("RETURN"), arg("2"), arg("completion"), arg("score"),
      arg("DIALECT"), arg("2"),
    ]);
  } catch {
    // Empty index or query miss — treat as no hit.
    return null;
  }

  const flat = raw.map(resultToString);
  let completion: string | null = null;
  let score = 1;
  for (let i = 1; i < flat.length; i++) {
    if (flat[i] === "completion") completion = flat[i + 1] ?? "";
    if (flat[i] === "score") score = parseFloat(flat[i + 1] ?? "1");
  }
  if (completion == null) return null;

  const similarity = 1 - score; // cosine distance -> similarity
  if (similarity < cacheThreshold()) return null;
  return { completion, similarity: Number(similarity.toFixed(4)) };
}

// Write a completion into the semantic cache, keyed by input content +
// fingerprint so distinct example sets get distinct entries. Applies the TTL.
function storeCache(
  conn: ReturnType<typeof Redis.open>,
  input: string,
  completion: string,
  queryVec: number[],
  fingerprint: string,
): void {
  const id = `${CACHE_PREFIX}${fingerprint}:${hash(input)}`;
  conn.execute("HSET", [
    arg(id),
    arg("input"), arg(input),
    arg("completion"), arg(completion),
    arg("fingerprint"), arg(fingerprint),
    arg("embedding"), bytes(floatsToBytes(queryVec)),
  ]);
  const ttl = cacheTtlSeconds();
  if (ttl > 0) conn.execute("EXPIRE", [arg(id), { tag: "int64", val: BigInt(ttl) }]);
}

// Delete every cache entry (keys carry the CACHE_PREFIX). Returns the count
// removed. Leaves the example bank untouched.
function clearCache(conn: ReturnType<typeof Redis.open>): number {
  let removed = 0;
  let cursor = "0";
  do {
    const reply = conn.execute("SCAN", [
      arg(cursor),
      arg("MATCH"), arg(`${CACHE_PREFIX}*`),
      arg("COUNT"), { tag: "int64", val: 100n },
    ]);
    cursor = resultToString(reply[0]);
    for (let i = 1; i < reply.length; i++) {
      const key = resultToString(reply[i]);
      if (key) {
        conn.execute("DEL", [arg(key)]);
        removed++;
      }
    }
  } while (cursor !== "0");
  return removed;
}

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

// Build the chat messages with retrieved examples as few-shot context. Rather
// than stuffing the examples into one big user turn (which reasoning models
// like gpt-oss treat as text to analyze, diluting them), we replay each example
// as a real prior exchange: a user turn with the past request, then an assistant
// turn with the response that worked. The model then pattern-matches on the
// conversation shape and imitates the exemplars instead of reasoning over them.
function buildMessages(
  input: string,
  examples: Example[],
  systemPrompt: string = ROUTE_PROMPTS[GENERAL_ROUTE],
): ChatMessage[] {
  const messages: ChatMessage[] = [
    { role: "system", content: systemPrompt },
  ];

  for (const ex of examples) {
    messages.push({ role: "user", content: ex.prompt });
    messages.push({ role: "assistant", content: ex.completion });
  }

  messages.push({ role: "user", content: input });
  return messages;
}

// Call an OpenAI-compatible /v1/chat/completions endpoint for a completion.
// Ollama, OpenAI, and Anthropic all speak this shape, so swapping providers is
// purely a matter of changing the Spin variables (URL, model, key). Returns the
// generated text plus token counts mapped to the same shape the app expects.
async function chatCompletion(
  messages: ChatMessage[],
): Promise<{ text: string; usage: { promptTokenCount: number; generatedTokenCount: number } }> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  const key = inferenceApiKey();
  if (key) headers["authorization"] = `Bearer ${key}`;

  const res = await fetch(inferenceApiUrl(), {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: chatModel(),
      messages,
      // Reasoning models (e.g. gemma/qwen on Ollama) spend tokens in a
      // thinking phase before emitting the answer, so keep this generous or
      // the completion comes back empty with finish_reason="length".
      max_tokens: 1024,
      temperature: 0.7,
      stream: false,
    }),
  });
  if (!res.ok) {
    throw new Error(`inference ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as {
    choices?: { message?: { content?: string } }[];
    usage?: { prompt_tokens?: number; completion_tokens?: number };
  };
  // Read ONLY message.content. Reasoning models (gpt-oss) also return a
  // `reasoning` field with their internal thinking — we deliberately drop it so
  // it never appears in the response.
  return {
    text: data.choices?.[0]?.message?.content ?? "",
    usage: {
      promptTokenCount: data.usage?.prompt_tokens ?? 0,
      generatedTokenCount: data.usage?.completion_tokens ?? 0,
    },
  };
}

// Open a streaming chat completion. Returns the upstream Response so the caller
// can read its SSE body incrementally. Same OpenAI-compatible request as
// chatCompletion(), but with stream:true and usage requested in the final chunk.
async function chatCompletionStream(messages: ChatMessage[]): Promise<Response> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  const key = inferenceApiKey();
  if (key) headers["authorization"] = `Bearer ${key}`;

  const res = await fetch(inferenceApiUrl(), {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: chatModel(),
      messages,
      max_tokens: 1024,
      temperature: 0.7,
      stream: true,
      stream_options: { include_usage: true },
    }),
  });
  if (!res.ok) {
    throw new Error(`inference ${res.status}: ${await res.text()}`);
  }
  return res;
}

// Pull the user-facing text out of one OpenAI-style streaming chunk. We read
// ONLY delta.content — reasoning models (gpt-oss) also stream a delta.reasoning
// trace, but that's the model thinking out loud and must not appear in the
// response, so we ignore it.
function deltaText(chunk: {
  choices?: { delta?: { content?: string } }[];
}): string {
  return chunk.choices?.[0]?.delta?.content ?? "";
}

// Pull a human-readable message out of whatever was thrown. Spin host errors
// are tagged objects whose text lives in `.payload.val` (and sometimes `.val`),
// not in String(e) — which gives "[object Object]". JS Errors use `.message`.
function errorMessage(e: unknown): string {
  const err = e as { message?: string; val?: unknown; payload?: { val?: unknown } };
  return String(err?.payload?.val ?? err?.val ?? err?.message ?? e);
}

// Global error hook: log every uncaught route error to stdout (so it lands in
// `docker logs` / the platform log collector) before formatting the response.
// Without this, itty-router serializes the error into the HTTP body only and
// nothing reaches the logs.
const router = AutoRouter({
  catch: (e: unknown, req: Request) => {
    console.error(`[error] ${req.method} ${new URL(req.url).pathname}:`, errorMessage(e));
    return error(500, errorMessage(e));
  },
});

router.get("/health", () => json({ status: "ok" }));

// GET /stats  -> { exampleCount }  (size of the few-shot example bank)
router.get("/stats", () => {
  const conn = Redis.open(valkeyUrl());
  ensureIndex(conn);
  return json({
    exampleCount: countDocs(conn, INDEX_NAME),
    cacheCount: countDocs(conn, CACHE_INDEX),
    routeCount: countDocs(conn, ROUTE_INDEX),
  });
});

// POST /routes  { "route": "...", "utterance": "..." }
// Add one labeled exemplar to the router index. Used by scripts/seed-routes.mjs
// to load routes.json; can also teach a new exemplar at runtime.
router.post("/routes", async (req) => {
  const body = (await req.json()) as { route?: string; utterance?: string };
  const route = body?.route?.trim();
  const utterance = body?.utterance?.trim();
  if (!route || !utterance)
    return json({ error: "need both 'route' and 'utterance'" }, 400);

  const conn = Redis.open(valkeyUrl());
  ensureRouteIndex(conn);
  return json({ stored: storeRoute(conn, route, utterance) });
});

// GET /routes  -> { routes: ["schedule-meeting", ...] }
// The known route labels (those with a dedicated system prompt). Used by the UI
// to populate the route picker on the promote flow.
router.get("/routes", () => json({ routes: Object.keys(ROUTE_PROMPTS) }));

// POST /cache/clear  -> flush all semantic-cache entries (example bank intact)
router.post("/cache/clear", () => {
  const conn = Redis.open(valkeyUrl());
  return json({ cleared: clearCache(conn) });
});

// Serve a tiny single-page UI for poking at /infer and /feedback by hand.
router.get("/", () => new Response(INDEX_HTML, {
  headers: { "content-type": "text/html; charset=utf-8" },
}));

// POST /retrieve  { "input": "..." }
// The fast half of /infer: embed + KNN search only, no LLM call. The UI hits
// this first to show the matched few-shot examples immediately, then calls
// /infer for the (much slower) completion.
router.post("/retrieve", async (req) => {
  const body = (await req.json()) as { input?: string };
  const input = body?.input?.trim();
  if (!input) return json({ error: "missing 'input'" }, 400);

  const conn = Redis.open(valkeyUrl());
  ensureIndex(conn);

  const examples = searchExamples(conn, embed(input));
  return json({
    input,
    fewShotExamples: examples.map((e) => ({
      prompt: e.prompt,
      similarity: Number((1 - e.score).toFixed(4)),
    })),
  });
});

// POST /infer/stream  { "input": "..." }  -> Server-Sent Events
// One request that streams the whole flow as it happens:
//   event: examples  data: { fewShotExamples: [...] }   (sent immediately)
//   event: delta     data: { text: "..." }               (one per token chunk)
//   event: done      data: { usage: {...} }              (final)
//   event: error     data: { error: "..." }
// The retrieval (embed + KNN) runs first and its result is flushed before the
// slow LLM stream begins, so the UI shows examples instantly then fills in text.
router.post("/infer/stream", async (req) => {
  const body = (await req.json()) as { input?: string; noCache?: boolean };
  const input = body?.input?.trim();
  if (!input) return json({ error: "missing 'input'" }, 400);
  const noCache = !!body?.noCache;

  const encoder = new TextEncoder();
  const sse = (event: string, data: unknown) =>
    encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const conn = Redis.open(valkeyUrl());
        ensureIndex(conn);
        ensureCacheIndex(conn);
        ensureRouteIndex(conn);
        const queryVec = embed(input);
        // Route first: the chosen intent selects the system prompt below.
        const match = pickRoute(conn, queryVec);
        const examples = searchExamples(conn, queryVec);
        const fingerprint = examplesFingerprint(examples);

        // Flush the route + retrieved examples right away — the fast part.
        controller.enqueue(sse("examples", {
          input,
          route: match.route,
          routeSimilarity: match.similarity,
          routeMatched: match.matched,
          fewShotExamples: examples.map((e) => ({
            prompt: e.prompt,
            similarity: Number((1 - e.score).toFixed(4)),
          })),
        }));

        // Cache hit: emit the whole cached completion as one delta and finish,
        // skipping the (slow) LLM call entirely.
        if (!noCache) {
          const hit = lookupCache(conn, queryVec, fingerprint);
          if (hit) {
            controller.enqueue(sse("cached", { similarity: hit.similarity }));
            controller.enqueue(sse("delta", { text: hit.completion }));
            controller.enqueue(sse("done", {
              usage: { promptTokenCount: 0, generatedTokenCount: 0 },
              cached: true,
            }));
            return;
          }
        }

        // Miss: stream the LLM completion token-by-token, accumulating the full
        // text so we can write it to the cache once complete.
        const upstream = await chatCompletionStream(
          buildMessages(input, examples, routePrompt(match.route)),
        );
        const reader = upstream.body!.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        let usage: unknown = null;
        let full = "";

        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });

          // OpenAI-style SSE: lines beginning "data: ", terminated by "\n\n".
          let nl: number;
          while ((nl = buf.indexOf("\n")) !== -1) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (payload === "[DONE]") continue;
            try {
              const chunk = JSON.parse(payload);
              const text = deltaText(chunk);
              if (text) {
                full += text;
                controller.enqueue(sse("delta", { text }));
              }
              if (chunk.usage) usage = chunk.usage;
            } catch {
              // Ignore partial/non-JSON keepalive lines.
            }
          }
        }

        const completion = full.trim();
        if (completion) storeCache(conn, input, completion, queryVec, fingerprint);

        const u = usage as { prompt_tokens?: number; completion_tokens?: number } | null;
        controller.enqueue(sse("done", {
          usage: {
            promptTokenCount: u?.prompt_tokens ?? 0,
            generatedTokenCount: u?.completion_tokens ?? 0,
          },
          cached: false,
        }));
      } catch (e) {
        console.error("[infer/stream] failed:", (e as Error)?.stack ?? e);
        controller.enqueue(sse("error", { error: String((e as Error)?.message ?? e) }));
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
    },
  });
});

// POST /infer  { "input": "...", "noCache"?: true }
// Set noCache to skip the cache lookup (a fresh completion is still written to
// the cache afterwards).
router.post("/infer", async (req) => {
  const body = (await req.json()) as { input?: string; noCache?: boolean };
  const input = body?.input?.trim();
  if (!input) return json({ error: "missing 'input'" }, 400);

  const conn = Redis.open(valkeyUrl());
  ensureIndex(conn);
  ensureCacheIndex(conn);
  ensureRouteIndex(conn);

  const queryVec = embed(input);
  const match = pickRoute(conn, queryVec);
  const examples = searchExamples(conn, queryVec);
  const fingerprint = examplesFingerprint(examples);
  const fewShotExamples = examples.map((e) => ({
    prompt: e.prompt,
    similarity: Number((1 - e.score).toFixed(4)), // cosine sim from distance
  }));
  const routing = {
    route: match.route,
    routeSimilarity: match.similarity,
    routeMatched: match.matched,
  };

  // Cache lookup: same-ish input built from the same examples → reuse.
  if (!body?.noCache) {
    const hit = lookupCache(conn, queryVec, fingerprint);
    if (hit) {
      return json({
        input,
        completion: hit.completion,
        fewShotExamples,
        ...routing,
        usage: { promptTokenCount: 0, generatedTokenCount: 0 },
        cached: true,
        cacheSimilarity: hit.similarity,
      });
    }
  }

  try {
    const inference = await chatCompletion(
      buildMessages(input, examples, routePrompt(match.route)),
    );
    const completion = inference.text.trim();
    if (completion) storeCache(conn, input, completion, queryVec, fingerprint);

    return json({
      input,
      completion,
      fewShotExamples,
      ...routing,
      usage: inference.usage,
      cached: false,
    });
  } catch (e) {
    console.error("[infer] failed:", (e as Error)?.stack ?? e);
    return json({ error: String((e as Error)?.message ?? e) }, 502);
  }
});

// POST /feedback  { "input": "...", "completion": "..." }
// Marks a pair as successful so it joins the few-shot example bank.
router.post("/feedback", async (req) => {
  const body = (await req.json()) as { input?: string; completion?: string };
  const input = body?.input?.trim();
  const completion = body?.completion?.trim();
  const route = (body as { route?: string })?.route?.trim();
  if (!input || !completion)
    return json({ error: "need both 'input' and 'completion'" }, 400);

  const conn = Redis.open(valkeyUrl());
  ensureIndex(conn);
  const id = storeExample(conn, input, completion);

  // Optionally also teach the router: the same input becomes a labeled exemplar
  // for `route`, so the corpus and the router grow together. Skipped when no
  // route is given (the example bank grows but routing is left alone).
  let routeId: string | undefined;
  if (route) {
    ensureRouteIndex(conn);
    routeId = storeRoute(conn, route, input);
  }

  return json({ stored: id, routeStored: routeId });
});

function json(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: { "content-type": "application/json" },
  });
}

//@ts-ignore  -- Spin's JS host injects the fetch event handler
addEventListener("fetch", (event: FetchEvent) => {
  event.respondWith(router.fetch(event.request));
});

// ---------------------------------------------------------------------------
// Test UI. A single self-contained page (no build step, no external assets)
// served from GET /. Run an inference, see the retrieved few-shot examples and
// token usage, then promote a good (input, completion) pair into the example
// bank with one click to watch retrieval improve.
// ---------------------------------------------------------------------------
const INDEX_HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Few-shot inference</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    font: 15px/1.5 system-ui, -apple-system, sans-serif;
    max-width: 720px; margin: 0 auto; padding: 2rem 1rem;
  }
  h1 { font-size: 1.4rem; margin-bottom: .25rem; }
  p.sub { color: #888; margin-top: 0; }
  label { display: block; font-weight: 600; margin: 1rem 0 .35rem; }
  textarea, input {
    width: 100%; padding: .6rem .7rem; font: inherit;
    border: 1px solid #8884; border-radius: 8px; background: #8881;
  }
  textarea { min-height: 5rem; resize: vertical; }
  button {
    font: inherit; font-weight: 600; cursor: pointer;
    padding: .55rem 1.1rem; margin-top: 1rem;
    border: 0; border-radius: 8px; background: #3b82f6; color: #fff;
  }
  button:disabled { opacity: .5; cursor: progress; }
  button.secondary { background: #6b7280; }
  .card {
    border: 1px solid #8883; border-radius: 10px;
    padding: 1rem; margin-top: 1.5rem;
  }
  .completion { white-space: pre-wrap; }
  .examples { margin-top: 1rem; }
  .ex {
    font-size: .85rem; color: #888;
    border-top: 1px dashed #8884; padding-top: .5rem; margin-top: .5rem;
  }
  .meta { font-size: .8rem; color: #888; margin-top: 1rem; }
  .err { color: #ef4444; }
  .hidden { display: none; }
  .row { display: flex; align-items: center; gap: 1rem; }
  .inline { display: inline-flex; align-items: center; gap: .35rem; font-weight: 400; margin: 1rem 0 0; }
  .inline input { width: auto; }
  .badge {
    font-size: .75rem; font-weight: 600; padding: .1rem .5rem;
    border-radius: 999px; background: #16a34a; color: #fff; vertical-align: middle;
  }
  button.link {
    background: none; color: #3b82f6; padding: 0; margin: 0;
    font-size: .8rem; text-decoration: underline; font-weight: 400;
  }
  .route { margin-bottom: .75rem; font-size: .85rem; }
  .route .chip {
    font-family: ui-monospace, monospace; font-weight: 600;
    padding: .1rem .5rem; border-radius: 6px;
    background: #8b5cf6; color: #fff;
  }
  .route .chip.fallback { background: #6b7280; }
  .route .sim { color: #888; margin-left: .4rem; }
</style>
</head>
<body>
  <h1>Few-shot inference</h1>
  <p class="sub">Embed &rarr; KNN-search Valkey for similar past wins &rarr; inject as examples &rarr; complete.</p>
  <p class="sub">
    Example bank: <strong id="count">…</strong> stored &middot;
    Cache: <strong id="cacheCount">…</strong> entries
    <button id="clearCache" class="link">clear</button> &middot;
    Routes: <strong id="routeCount">…</strong>
  </p>

  <label for="input">Your request</label>
  <textarea id="input" placeholder="e.g. Write a friendly out-of-office reply"></textarea>
  <div class="row">
    <button id="infer">Run inference</button>
    <label class="inline"><input type="checkbox" id="noCache" /> skip cache</label>
  </div>

  <div id="result" class="card hidden">
    <div id="route" class="route hidden"></div>
    <strong>Completion</strong> <span id="cacheBadge" class="badge hidden"></span>
    <div id="completion" class="completion"></div>

    <div id="examples" class="examples"></div>
    <div id="meta" class="meta"></div>

    <div class="row">
      <button id="promote" class="secondary">👍 Good — add to example bank</button>
      <label class="inline">teach route
        <select id="promoteRoute"><option value="">— don't teach —</option></select>
      </label>
    </div>
    <span id="promoteMsg" class="meta"></span>
  </div>

  <div id="error" class="card err hidden"></div>

<script>
const $ = (id) => document.getElementById(id);
let lastInput = "", lastCompletion = "", lastRoute = "";

// Populate the "teach route" picker from the known route labels.
async function loadRoutes() {
  try {
    const res = await fetch("/routes");
    const data = await res.json();
    for (const r of data.routes || []) {
      const opt = document.createElement("option");
      opt.value = r; opt.textContent = r;
      $("promoteRoute").appendChild(opt);
    }
  } catch { /* picker just stays at "don't teach" */ }
}
loadRoutes();

async function refreshCount() {
  try {
    const res = await fetch("/stats");
    const data = await res.json();
    $("count").textContent = data.exampleCount;
    $("cacheCount").textContent = data.cacheCount ?? 0;
    $("routeCount").textContent = data.routeCount ?? 0;
  } catch {
    $("count").textContent = "?";
    $("cacheCount").textContent = "?";
    $("routeCount").textContent = "?";
  }
}
refreshCount();

async function postJSON(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let data; try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!res.ok) throw new Error(data.error || text || res.status);
  return data;
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").classList.remove("hidden");
}

function renderRoute(data) {
  const el = $("route");
  if (!data.route) { el.classList.add("hidden"); return; }
  const cls = data.routeMatched ? "chip" : "chip fallback";
  const tail = data.routeMatched
    ? '<span class="sim">sim ' + data.routeSimilarity + "</span>"
    : '<span class="sim">no confident match (sim ' + data.routeSimilarity +
      ") — fell back</span>";
  el.innerHTML = "Routed to <span class=\\"" + cls + "\\">" +
    data.route.replace(/</g, "&lt;") + "</span>" + tail;
  el.classList.remove("hidden");
}

function renderExamples(exs) {
  $("examples").innerHTML = exs && exs.length
    ? "<strong>Retrieved examples (" + exs.length + ")</strong>" +
      exs.map((e) =>
        '<div class="ex">sim ' + e.similarity + " — " +
        e.prompt.replace(/</g, "&lt;") + "</div>").join("")
    : '<div class="ex">No prior examples retrieved yet — add some below.</div>';
}

// Dispatch one parsed SSE event to the UI.
function onSSE(event, data) {
  if (event === "examples") {
    renderRoute(data);
    renderExamples(data.fewShotExamples);
    $("completion").textContent = "";
    lastInput = data.input;
    // Pre-select the picker to the matched route so promoting also reinforces
    // it by default; a fallback leaves it on "don't teach" for the user to set.
    lastRoute = data.routeMatched ? data.route : "";
    $("promoteRoute").value = lastRoute;
  } else if (event === "cached") {
    // Cache hit: flag it; the cached completion arrives as the next delta.
    const badge = $("cacheBadge");
    badge.textContent = "cached ✓ sim " + data.similarity;
    badge.classList.remove("hidden");
  } else if (event === "delta") {
    // First token replaces the "Generating…" placeholder.
    lastCompletion += data.text;
    $("completion").textContent = lastCompletion;
  } else if (event === "done") {
    const u = data.usage || {};
    $("meta").textContent = data.cached
      ? "served from cache (0 tokens)"
      : "tokens: " + (u.promptTokenCount ?? "?") + " prompt / " +
        (u.generatedTokenCount ?? "?") + " generated";
    if (!lastCompletion) $("completion").textContent = "(empty)";
  } else if (event === "error") {
    showError("Inference failed: " + data.error);
  }
}

$("infer").onclick = async () => {
  const input = $("input").value.trim();
  if (!input) return;
  $("error").classList.add("hidden");
  $("infer").disabled = true;
  $("infer").textContent = "Running…";
  $("promote").disabled = true;
  $("promoteMsg").textContent = "";
  $("cacheBadge").classList.add("hidden");
  $("route").classList.add("hidden");
  lastInput = input;
  lastCompletion = "";
  $("completion").textContent = "Generating completion…";
  $("meta").textContent = "";
  $("result").classList.remove("hidden");

  try {
    const res = await fetch("/infer/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input, noCache: $("noCache").checked }),
    });
    if (!res.ok || !res.body) {
      throw new Error("HTTP " + res.status);
    }

    // Parse the SSE stream by hand: events are blank-line-separated blocks of
    // "event:" / "data:" lines. (EventSource only supports GET, so we can't
    // use it for a POST body.)
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buf.indexOf("\\n\\n")) !== -1) {
        const block = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        let ev = "message", dataStr = "";
        for (const line of block.split("\\n")) {
          if (line.startsWith("event:")) ev = line.slice(6).trim();
          else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
        }
        if (dataStr) {
          try { onSSE(ev, JSON.parse(dataStr)); } catch {}
        }
      }
    }
    $("promote").disabled = false;
    refreshCount(); // a miss just added a cache entry
  } catch (e) {
    showError("Inference failed: " + e.message);
  } finally {
    $("infer").disabled = false;
    $("infer").textContent = "Run inference";
  }
};

$("clearCache").onclick = async () => {
  try {
    const data = await postJSON("/cache/clear", {});
    $("cacheCount").textContent = "0";
    $("promoteMsg").textContent = "Cleared " + data.cleared + " cache entries";
  } catch (e) {
    showError("Clear cache failed: " + e.message);
  }
};

$("promote").onclick = async () => {
  $("promote").disabled = true;
  try {
    const route = $("promoteRoute").value;
    const data = await postJSON("/feedback", {
      input: lastInput,
      completion: lastCompletion,
      route, // empty string -> example bank only, router untouched
    });
    $("promoteMsg").textContent = data.routeStored
      ? "Stored as " + data.stored + " · taught route '" + route + "'"
      : "Stored as " + data.stored;
    refreshCount();
  } catch (e) {
    $("promoteMsg").textContent = "Failed: " + e.message;
  } finally {
    $("promote").disabled = false;
  }
};
</script>
</body>
</html>`;

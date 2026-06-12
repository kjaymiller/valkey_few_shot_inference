import { AutoRouter } from "itty-router";
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

// Pull the Valkey connection string from a Spin variable (see spin.toml).
function valkeyUrl(): string {
  const url = Variables.get("valkey_url");
  if (!url) throw new Error("valkey_url variable is not set");
  return url;
}

// Inference endpoint config, all from Spin variables so the same build can
// target Ollama, OpenAI, or Anthropic without code changes.
function inferenceApiUrl(): string {
  const url = Variables.get("inference_api_url");
  if (!url) throw new Error("inference_api_url variable is not set");
  return url;
}
function chatModel(): string {
  const model = Variables.get("chat_model");
  if (!model) throw new Error("chat_model variable is not set");
  return model;
}
// Optional: a bearer token for hosted providers (OpenAI, Anthropic). Local
// Ollama needs none, so this is allowed to be empty.
function inferenceApiKey(): string {
  return Variables.get("inference_api_key") ?? "";
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

// Ensure the HNSW vector index exists. FT.CREATE errors if it already exists,
// so we treat an "Index already exists" failure as success (idempotent).
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
    // Spin throws a tagged error object whose human-readable message lives in
    // `.payload.val` (and sometimes `.val`), not in String(e). Check all.
    const err = e as { val?: unknown; payload?: { val?: unknown } };
    const msg = `${err?.payload?.val ?? ""} ${err?.val ?? ""} ${String(e)}`;
    if (!msg.includes("already exists")) throw e;
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

// Count how many examples are currently in the bank. valkey-search rejects a
// bare "*" query, so we read num_docs from FT.INFO instead. The reply is a
// flat [field, value, field, value, ...] list; find num_docs and take the next.
function countExamples(conn: ReturnType<typeof Redis.open>): number {
  const raw = conn.execute("FT.INFO", [arg(INDEX_NAME)]).map(resultToString);
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
function buildMessages(input: string, examples: Example[]): ChatMessage[] {
  const messages: ChatMessage[] = [
    {
      role: "system",
      content:
        "You are a helpful assistant. The conversation below contains past " +
        "requests and the responses that worked well — match their tone and " +
        "format. Reply with only the response itself, no preamble.",
    },
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

const router = AutoRouter();

router.get("/health", () => json({ status: "ok" }));

// GET /stats  -> { exampleCount }  (size of the few-shot example bank)
router.get("/stats", () => {
  const conn = Redis.open(valkeyUrl());
  ensureIndex(conn);
  return json({ exampleCount: countExamples(conn) });
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
  const body = (await req.json()) as { input?: string };
  const input = body?.input?.trim();
  if (!input) return json({ error: "missing 'input'" }, 400);

  const encoder = new TextEncoder();
  const sse = (event: string, data: unknown) =>
    encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const conn = Redis.open(valkeyUrl());
        ensureIndex(conn);
        const examples = searchExamples(conn, embed(input));

        // Flush the retrieved examples right away — this is the fast part.
        controller.enqueue(sse("examples", {
          input,
          fewShotExamples: examples.map((e) => ({
            prompt: e.prompt,
            similarity: Number((1 - e.score).toFixed(4)),
          })),
        }));

        // Now stream the LLM completion token-by-token.
        const upstream = await chatCompletionStream(buildMessages(input, examples));
        const reader = upstream.body!.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        let usage: unknown = null;

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
              if (text) controller.enqueue(sse("delta", { text }));
              if (chunk.usage) usage = chunk.usage;
            } catch {
              // Ignore partial/non-JSON keepalive lines.
            }
          }
        }

        const u = usage as { prompt_tokens?: number; completion_tokens?: number } | null;
        controller.enqueue(sse("done", {
          usage: {
            promptTokenCount: u?.prompt_tokens ?? 0,
            generatedTokenCount: u?.completion_tokens ?? 0,
          },
        }));
      } catch (e) {
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

// POST /infer  { "input": "..." }
router.post("/infer", async (req) => {
  const body = (await req.json()) as { input?: string };
  const input = body?.input?.trim();
  if (!input) return json({ error: "missing 'input'" }, 400);

  const conn = Redis.open(valkeyUrl());
  ensureIndex(conn);

  const queryVec = embed(input);
  const examples = searchExamples(conn, queryVec);

  const inference = await chatCompletion(buildMessages(input, examples));

  return json({
    input,
    completion: inference.text.trim(),
    fewShotExamples: examples.map((e) => ({
      prompt: e.prompt,
      similarity: Number((1 - e.score).toFixed(4)), // cosine sim from distance
    })),
    usage: inference.usage,
  });
});

// POST /feedback  { "input": "...", "completion": "..." }
// Marks a pair as successful so it joins the few-shot example bank.
router.post("/feedback", async (req) => {
  const body = (await req.json()) as { input?: string; completion?: string };
  const input = body?.input?.trim();
  const completion = body?.completion?.trim();
  if (!input || !completion)
    return json({ error: "need both 'input' and 'completion'" }, 400);

  const conn = Redis.open(valkeyUrl());
  ensureIndex(conn);
  const id = storeExample(conn, input, completion);

  return json({ stored: id });
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
</style>
</head>
<body>
  <h1>Few-shot inference</h1>
  <p class="sub">Embed &rarr; KNN-search Valkey for similar past wins &rarr; inject as examples &rarr; complete.</p>
  <p class="sub">Example bank: <strong id="count">…</strong> stored</p>

  <label for="input">Your request</label>
  <textarea id="input" placeholder="e.g. Write a friendly out-of-office reply"></textarea>
  <button id="infer">Run inference</button>

  <div id="result" class="card hidden">
    <strong>Completion</strong>
    <div id="completion" class="completion"></div>

    <div id="examples" class="examples"></div>
    <div id="meta" class="meta"></div>

    <button id="promote" class="secondary">👍 Good — add to example bank</button>
    <span id="promoteMsg" class="meta"></span>
  </div>

  <div id="error" class="card err hidden"></div>

<script>
const $ = (id) => document.getElementById(id);
let lastInput = "", lastCompletion = "";

async function refreshCount() {
  try {
    const res = await fetch("/stats");
    const data = await res.json();
    $("count").textContent = data.exampleCount;
  } catch {
    $("count").textContent = "?";
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
    renderExamples(data.fewShotExamples);
    $("completion").textContent = "";
    lastInput = data.input;
  } else if (event === "delta") {
    // First token replaces the "Generating…" placeholder.
    lastCompletion += data.text;
    $("completion").textContent = lastCompletion;
  } else if (event === "done") {
    const u = data.usage || {};
    $("meta").textContent =
      "tokens: " + (u.promptTokenCount ?? "?") + " prompt / " +
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
  lastInput = input;
  lastCompletion = "";
  $("completion").textContent = "Generating completion…";
  $("meta").textContent = "";
  $("result").classList.remove("hidden");

  try {
    const res = await fetch("/infer/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input }),
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
  } catch (e) {
    showError("Inference failed: " + e.message);
  } finally {
    $("infer").disabled = false;
    $("infer").textContent = "Run inference";
  }
};

$("promote").onclick = async () => {
  $("promote").disabled = true;
  try {
    const data = await postJSON("/feedback", {
      input: lastInput,
      completion: lastCompletion,
    });
    $("promoteMsg").textContent = "Stored as " + data.stored;
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

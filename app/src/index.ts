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
const CHAT_MODEL = "llama2-chat";
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
    const msg = String(e);
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

// Build the llama2-chat prompt with retrieved examples as few-shot context.
function buildPrompt(input: string, examples: Example[]): string {
  const shots = examples
    .map(
      (ex, i) =>
        `Example ${i + 1}:\nInput: ${ex.prompt}\nOutput: ${ex.completion}`,
    )
    .join("\n\n");

  const preamble = examples.length
    ? `Here are some past successful responses to similar inputs. Use them as a guide for tone and format.\n\n${shots}\n\n`
    : "";

  return `<<SYS>>You are a helpful assistant. ${preamble}<</SYS>>\n\nInput: ${input}\nOutput:`;
}

const router = AutoRouter();

router.get("/health", () => json({ status: "ok" }));

// POST /infer  { "input": "..." }
router.post("/infer", async (req) => {
  const body = (await req.json()) as { input?: string };
  const input = body?.input?.trim();
  if (!input) return json({ error: "missing 'input'" }, 400);

  const conn = Redis.open(valkeyUrl());
  ensureIndex(conn);

  const queryVec = embed(input);
  const examples = searchExamples(conn, queryVec);

  const prompt = buildPrompt(input, examples);
  const inference = Llm.infer(CHAT_MODEL, prompt, {
    maxTokens: 200,
    temperature: 0.7,
  });

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

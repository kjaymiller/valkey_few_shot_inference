# Spin JS/TS Component Gotchas

Hard-won notes from building a `@fermyon/spin-sdk` 3.x TypeScript component
(HTTP trigger, serverless AI embeddings, Redis/Valkey, local Ollama for chat).
Versions used: Spin 3.6.3, `@fermyon/spin-sdk` 3.x, Node 22, esbuild + `j2w`.

Share these across projects — most are not in the official docs and cost real
debugging time to find.

---

## 1. Build pipeline (esbuild → j2w → wasm)

The build is two steps: bundle TS to ESM with esbuild, then componentize to
Wasm with `j2w`.

```jsonc
// package.json scripts
"build:js":  "esbuild src/index.ts --bundle --outfile=dist/index.js --format=esm --platform=browser --target=es2022 --external:fermyon:* --external:spin:* --external:wasi:*",
"build:wasm":"mkdir -p target && j2w -i dist/index.js -o target/few-shot.wasm --trigger-type spin3-http",
"build":     "npm run build:js && npm run build:wasm"
```

- **esbuild externals are mandatory.** The SDK imports Wasm host interfaces
  (`fermyon:*`, `spin:*`, `wasi:*`). Without
  `--external:fermyon:* --external:spin:* --external:wasi:*`, esbuild fails to
  resolve them and the build dies.
- **`j2w --trigger-type` is `spin3-http`, NOT `http`.** This is the world name
  in the SDK's `wit/world.wit`. Using `http` fails.
- **`j2w` won't create the output dir.** `mkdir -p target` first.

### ⚠️ The j2w checksum-cache trap (cost the most time)

`j2w` caches by checksum. If it thinks the input `dist/index.js` hasn't
changed it prints:

```
No changes detected in source file. Skipping componentization.
```

…and **silently leaves the OLD wasm in place.** This bit us repeatedly: source
was edited, `build:js` rewrote `dist/index.js`, but `j2w` skipped, so
`spin up` kept serving a stale binary and our fixes "didn't work."

**Always force a clean wasm rebuild when iterating:**

```bash
rm -f target/few-shot.wasm && npm run build
```

Verify your change actually made it into the binary before restarting Spin:

```bash
strings target/few-shot.wasm | grep -c 'some-unique-string-from-your-change'
```

> Note: esbuild may minify identifiers (`false` → `!1`), so `grep` the wasm for
> a string literal or a stable substring, not a minifiable token.

---

## 2. Spin reloads everything at startup — restart after EVERY rebuild

`spin up` loads the wasm **and** AI models **and** config variables at boot.
There is no hot reload. After any rebuild or model/config change you must kill
and restart the process. Stale leftover `spin` processes (especially from
backgrounded runs) will squat on the port and keep serving the old build:

```bash
pkill -f 'spin (up|trigger)'
spin up --variable valkey_url="redis://127.0.0.1:6379" --listen 127.0.0.1:3000
```

Confirm which process owns the port and what it's actually running:

```bash
lsof -nP -iTCP:3000 -sTCP:LISTEN
ps -o lstart=,command= -p <PID>
```

---

## 3. Config variables: `--variable name=...` vs env var name

`spin.toml`:

```toml
[component.few-shot.variables]
valkey_url = "{{ valkey_url }}"

[variables]
valkey_url = { required = true, secret = true }
```

Read in code with `Variables.get("valkey_url")` → returns `string | null`.

Two ways to supply it, and a classic foot-gun:

- CLI flag: `spin up --variable valkey_url="rediss://..."`
- Env var: Spin reads `SPIN_VARIABLE_VALKEY_URL` (UPPERCASE, prefixed).
  A bare `VALKEY_URL` is **not** read by Spin.

**Shell-expansion foot-gun with secret injectors (`some-injector exec -- ...`):**

Any tool that injects secrets into a child process's environment hits this. The
outer shell expands `$VAR` *before* the injector runs, so it's empty:

```bash
# WRONG — outer shell expands $VALKEY_URL (empty) BEFORE the injector sets it:
<injector> exec -- spin up --variable valkey_url="$VALKEY_URL"

# RIGHT — defer expansion to the child process the injector spawned:
<injector> exec -- sh -c 'spin up --variable valkey_url="$VALKEY_URL"'
# or
<injector> exec -- sh -c 'SPIN_VARIABLE_VALKEY_URL="$VALKEY_URL" spin up'
```

If the variable arrives empty/missing you'll see your own code's
`valkey_url variable is not set`; if it arrives malformed you'll see Spin's
`invalid-address`. Always sanity-check what a secret actually resolves to before
blaming Spin — pipe it to `wc -c` and eyeball the first characters. (A secret
sourced from a command's output can silently capture an error/warning message
instead of the value if that command wrote to stdout on failure.)

---

## 4. Errors thrown by Spin host calls are TAGGED OBJECTS, not strings

Redis, LLM, and other host calls throw a structured object, **not** an `Error`
with a string message. The human-readable text is nested — typically at
`e.payload.val`, sometimes `e.val`. `String(e)` yields `"[object Object]"`,
which silently breaks naive message matching.

```ts
// The 500 body looks like:
// {"error":"[object Object]","payload":{"tag":"other","val":"Index: idx:examples ... already exists."}}

try {
  conn.execute("FT.CREATE", [ /* ... */ ]);
} catch (e) {
  // Check payload.val AND val AND the string form — be defensive.
  const err = e as { val?: unknown; payload?: { val?: unknown } };
  const msg = `${err?.payload?.val ?? ""} ${err?.val ?? ""} ${String(e)}`;
  if (!msg.includes("already exists")) throw e; // idempotent FT.CREATE
}
```

This is the #1 source of "my error handling looks right but doesn't fire."

---

## 5. Redis/Valkey API: tagged params in, flattened tagged results out

`Redis.open(url).execute(cmd, args)` is correct, but:

- **Args are TAGGED params, not bare strings.** Use
  `{ tag: "binary", val: Uint8Array }` or `{ tag: "int64", val: bigint }`.
  Redis is binary-safe, so command verbs and text args both ride as binary:

  ```ts
  const enc = new TextEncoder();
  const arg   = (s: string)     => ({ tag: "binary", val: enc.encode(s) });
  const bytes = (b: Uint8Array) => ({ tag: "binary", val: b });
  ```

- **Results come back as tagged `RedisResult[]`** —
  `{ tag: "binary" | "status" | "int64" | "nil", val }`. Decode binary with
  `TextDecoder`.

- **Spin FLATTENS nested RESP arrays.** An `FT.SEARCH` reply that is normally
  nested per-document arrives as **one flat list**:
  `[ total, key1, field, val, field, val, key2, field, val, ... ]`.
  You must walk it yourself, treating each key-prefix token as a record
  boundary — there are no per-doc sub-arrays to index into.

- **Vectors:** serialize `number[]` to little-endian float32 bytes for both
  `HSET ... embedding <blob>` and the `KNN ... $vec` PARAM:

  ```ts
  function floatsToBytes(floats: number[]): Uint8Array {
    const buf = new ArrayBuffer(floats.length * 4);
    const view = new DataView(buf);
    floats.forEach((f, i) => view.setFloat32(i * 4, f, true /* LE */));
    return new Uint8Array(buf);
  }
  ```

---

## 6. Serverless AI models must be present on disk under `.spin/ai-models/`

`spin.toml` declares the models:

```toml
ai_models = ["all-minilm-l6-v2", "llama2-chat"]
```

…but declaring is not installing. At runtime the host looks for files under
`<component-dir>/.spin/ai-models/<model-name>` and errors otherwise:

```
The directory expected to house the embeddings models
'.spin/ai-models/all-minilm-l6-v2' does not exist.
no model directory found in registry for model 'llama2-chat'
```

- **Sentence-transformer / embedding models = a DIRECTORY** containing
  `config.json`, `model.safetensors`, `tokenizer.json`. You can populate it
  from a HuggingFace cache, but **copy the real files** — don't symlink into
  `~/.cache/huggingface/...` (those are themselves blob symlinks and can be
  GC'd / not followed):

  ```bash
  SNAP=$(find ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
  mkdir -p .spin/ai-models/all-minilm-l6-v2
  for f in config.json model.safetensors tokenizer.json; do
    cp -L "$SNAP/$f" ".spin/ai-models/all-minilm-l6-v2/$f"   # -L dereferences
  done
  ```

- **LLM chat models (`llama2-chat` etc.) = a single GGUF file** at
  `.spin/ai-models/<model-name>`. Multi-GB; must be downloaded.

- `Llm.generateEmbeddings("all-minilm-l6-v2", [text])` → 384-dim vector.
- `Llm.infer("llama2-chat", prompt, { maxTokens, temperature })`.

---

## 7. Using local Ollama instead of Spin's serverless chat model

When you don't have / don't want a multi-GB GGUF locally, point chat at a local
Ollama server over HTTP. Embeddings can stay on Spin's `Llm`.

**Spin sandboxes outbound network** — you must allowlist the host in
`spin.toml` or the `fetch` fails:

```toml
allowed_outbound_hosts = ["redis://*:*", "rediss://*:*", "http://localhost:11434"]
```

Then call Ollama's `/api/generate` with `fetch`:

```ts
const res = await fetch("http://localhost:11434/api/generate", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    model: "gemma4:31b-mlx",
    prompt,
    stream: false,
    think: false,                       // see gotcha below
    options: { temperature: 0.7, num_predict: 200 },
  }),
});
const data = await res.json();          // { response, prompt_eval_count, eval_count, ... }
```

### ⚠️ Thinking models return an empty `response`

A "thinking" model (gemma4, and many recent ones) will, by default, spend its
entire `num_predict` budget on internal reasoning and return an **empty
`response`** (`eval_count` is maxed, `done_reason` not `stop`). Symptoms:
`completion: ""` but tokens were generated.

- Set **`"think": false`** in the request body to get a direct answer.
- `/api/generate` puts the answer in `response`; `/api/chat` puts it in
  `message.content` (and reasoning, if any, in `message.thinking`). Don't read
  the wrong field.
- Quick diagnosis — hit Ollama directly and inspect all fields:
  ```bash
  curl -s http://localhost:11434/api/generate \
    -d '{"model":"gemma4:31b-mlx","prompt":"Say hi.","stream":false,"think":false}' \
    | python3 -c 'import sys,json; print(json.load(sys.stdin))'
  ```

### ⚠️ Prompt template matters across models

A llama2-specific template (`<<SYS>>...<</SYS>>` with a trailing `Output:`
label) makes non-llama models stall/produce nothing. Use a plain instruction
format that ends with the actual request, e.g.:

```
You are a helpful assistant. <few-shot examples...>
Now write the response to this request. Reply with only the response itself.

Request: <input>
```

> Note: local Ollama is a dev-only choice — a deployed Spin component can't
> reach `localhost:11434`. Swap back to Spin serverless AI (or a reachable
> hosted endpoint) for production.

---

## Quick debug checklist when "it doesn't work"

1. Is the route even defined, and with the right METHOD? A 404 from
   itty-router just means no matching route (e.g. GET on a POST-only route, or
   hitting `/`). Routing is rarely the real bug.
2. `rm -f target/*.wasm && npm run build` — defeat the j2w cache.
3. `strings target/*.wasm | grep <your-change>` — confirm it's in the binary.
4. `pkill -f 'spin (up|trigger)'` then restart — no hot reload, kill zombies.
5. Decode the real error: look at `payload.val`, not `String(e)`.
6. Is the config variable actually arriving? (`SPIN_VARIABLE_*` naming +
   shell-expansion order with secret injectors.)
7. Are the AI model files physically on disk under `.spin/ai-models/`?

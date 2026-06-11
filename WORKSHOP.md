# Workshop: Self-Improving Few-Shot Inference with Spin + Aiven for Valkey

**Duration:** ~60–75 minutes
**Level:** Intermediate (comfortable in a terminal; no prior Spin/Valkey experience needed)

In this workshop you'll build an AI app that gets *better at a task the more
good answers it sees*. Each request is embedded into a vector, the app searches
**Aiven for Valkey** (on **Akamai**) for the most similar *past successful*
answers, and feeds them to the model as **few-shot examples**. You'll provision
the database with **OpenTofu**, run the app on **Fermyon Spin**, and watch
retrieval change the model's output in real time.

---

## What you'll learn

- How vector similarity search turns a plain key-value store into a retrieval engine
- How few-shot prompting grounds an LLM in your own past successes
- How to provision managed infrastructure declaratively with OpenTofu
- How to run a portable WebAssembly AI app with Spin's serverless inference

## Architecture at a glance

```
                 ┌─────────────────── Spin component (WebAssembly) ───────────────────┐
  POST /infer    │  1. embed(input)        all-minilm-l6-v2  →  384-dim vector         │
  {"input": …} ─▶│  2. FT.SEARCH KNN 3     ───────────────▶   Aiven for Valkey (HNSW)  │
                 │  3. build prompt with the 3 retrieved examples                      │
                 │  4. Llm.infer(llama2-chat)              →  completion                │
                 └─────────────────────────────────────────────────────────────────────┘
  POST /feedback {"input","completion"}  ─▶  embed + HSET  →  grows the example bank
```

---

## Prerequisites checklist

The fastest path is [mise](https://mise.jdx.dev), which reads `mise.toml` and
installs the exact tool versions this workshop was built against:

```bash
mise install        # node 22, opentofu 1.x, fermyon spin 3.x
```

Then confirm each tool answers with a version (mise puts them on your PATH;
`mise exec -- <cmd>` also works):

```bash
tofu version        # OpenTofu ≥ 1.6   → https://opentofu.org/docs/intro/install/
spin --version      # Fermyon Spin ≥ 3 → https://developer.fermyon.com/spin/install
node --version      # Node ≥ 18        → https://nodejs.org
jq --version        # for pretty JSON  → brew install jq / apt install jq
```

> ⚠️ mise's registry `spin` is the **Spinnaker** CLI, a different tool. This
> repo's `mise.toml` pins `github:fermyon/spin` (the WebAssembly runtime), so
> `mise install` gets the right one.

You also need:

- An **Aiven account** — free trial at <https://console.aiven.io>. Note your **project name** (top-left in the console).
- An **Aiven API token** (you'll create this in Module 1).

> **No GPU? No API keys?** Correct — embeddings *and* inference run through
> Spin's built-in serverless AI models, so there are no third-party AI keys in
> this workshop. The only credential is your Aiven token.

---

## Module 0 — Get the code (5 min)

```bash
git clone <this-repo-url> few_shot_inference
cd few_shot_inference
```

Take 2 minutes to skim the layout:

| Path | What it is |
|------|------------|
| `infra/` | OpenTofu: provisions Aiven for Valkey on Akamai |
| `app/src/index.ts` | The Spin component — the whole app logic |
| `app/scripts/seed.mjs` | Seeds the "past successes" example bank |
| `app/spin.toml` | Spin manifest: HTTP route, AI models, Valkey connection |

✅ **Checkpoint:** You're in the repo root and the four paths above exist (`ls infra app`).

---

## Module 1 — Provision Valkey on Akamai (15 min)

### 1.1 Create an Aiven API token

In the Aiven console: **User → Tokens → Generate token**, or via CLI:

```bash
export AIVEN_TOKEN="$(aiven user access-token create \
  --max-age-seconds 7200 --description workshop --json | jq -r .full_token)"
```

The OpenTofu provider reads `AIVEN_TOKEN` automatically.

### 1.2 Point the config at your project

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and set **`aiven_project`** to your project name. Leave
the rest as-is for now:

```hcl
aiven_project = "my-project-name"
cloud_name    = "akamai-us-east"   # an Akamai (Linode)-backed region
valkey_plan   = "startup-4"         # smallest plan with vector search
```

> 💡 To see which Akamai regions your project can reach:
> `aiven cloud list | grep akamai`

### 1.3 Apply

```bash
tofu init      # downloads the Aiven provider
tofu plan      # review: should show 1 resource to add
tofu apply     # type 'yes' — provisioning takes ~3–5 minutes
```

While it builds, here's what `main.tf` is asking for and *why*:

- `aiven_valkey` — a managed, Redis-compatible store.
- `valkey_persistence = "rdb"` — your example bank survives restarts.
- `valkey_maxmemory_policy = "noeviction"` — never silently drop example vectors.
- `public_access` — so the workshop laptop can connect directly.

### 1.4 Capture the connection string

```bash
export VALKEY_URL="$(tofu output -raw valkey_service_uri)"
echo "${VALKEY_URL%%@*}@…"   # prints scheme+user without leaking the host/pass
```

✅ **Checkpoint:** `tofu apply` finished with `Apply complete!` and `echo
$VALKEY_URL` starts with `rediss://`. Keep this terminal — `VALKEY_URL` lives
in its environment.

> ⏳ A freshly-created service can take an extra minute to accept connections
> even after `apply` returns. If Module 3 errors with a connection refusal,
> wait 60s and retry.

---

## Module 2 — Build the Spin app (10 min)

Open a **new terminal** in the repo (you'll keep the infra terminal for its
`VALKEY_URL`), then carry the variable over or re-export it.

```bash
cd ../app      # from infra/, or cd app from repo root
npm install
spin build     # bundles src/index.ts → WebAssembly
```

`spin build` runs esbuild to bundle the TypeScript, then compiles the JS bundle
into a `.wasm` module. First build pulls the JS-to-Wasm toolchain and may take a
minute.

### What the code does (read `src/index.ts` alongside this)

1. **`embed(text)`** calls `Llm.generateEmbeddings("all-minilm-l6-v2", …)` →
   a 384-number vector representing meaning.
2. **`ensureIndex`** runs `FT.CREATE … VECTOR HNSW … DIM 384 DISTANCE_METRIC
   COSINE` once (idempotently). HNSW is the graph index that makes nearest-
   neighbor search fast.
3. **`searchExamples`** runs `FT.SEARCH "*=>[KNN 3 @embedding $vec]"` — "find
   the 3 stored vectors closest to this one."
4. **`buildPrompt`** stitches the retrieved examples into the llama2-chat prompt
   as labeled `Input/Output` pairs.
5. **`storeExample`** (used by `/feedback`) embeds and `HSET`s a new success.

✅ **Checkpoint:** `spin build` ends with `Finished building all Spin
components` and `app/target/few-shot.wasm` exists.

---

## Module 3 — Run it and seed past successes (10 min)

### 3.1 Start the app

In the `app/` terminal (with `VALKEY_URL` exported):

```bash
spin up --variable valkey_url="$VALKEY_URL"
# Serving http://127.0.0.1:3000
```

Leave it running. In **another terminal**, sanity-check:

```bash
curl -s http://127.0.0.1:3000/health    # → {"status":"ok"}
```

### 3.2 Seed the example bank

```bash
cd app
APP_URL=http://127.0.0.1:3000 npm run seed
```

This POSTs six "past successful" answers to `/feedback`. The app embeds each one
and stores it in Valkey. You'll see a `✓` and the stored key per example.

✅ **Checkpoint:** All six examples report `✓ … -> {"stored":"example:…"}`.

> 🔎 Peek at the data (optional): connect with
> `redis-cli -u "$VALKEY_URL" --no-auth-warning` and run
> `FT.INFO idx:examples` — `num_docs` should be 6.

---

## Module 4 — See few-shot retrieval in action (15 min)

### 4.1 Ask something *similar* to a seeded example

```bash
curl -s http://127.0.0.1:3000/infer \
  -H 'content-type: application/json' \
  -d '{"input":"Write an out-of-office note for a conference trip"}' | jq
```

Look at the `fewShotExamples` array in the response — it shows **which past
answers were retrieved** and their cosine similarity. The out-of-office
vacation example should rank highest, and the completion should echo its tone
and format.

### 4.2 The key experiment: prove retrieval matters

Ask something with **no similar seeded example**, then seed one and ask again.

```bash
# (a) Cold — no relevant example yet:
curl -s http://127.0.0.1:3000/infer \
  -d '{"input":"Write a haiku about deployment pipelines"}' | jq '.fewShotExamples, .completion'

# (b) Teach it a good answer:
curl -s http://127.0.0.1:3000/feedback \
  -d '{"input":"Write a haiku about CI/CD","completion":"Green checks cascade down / a merge button glows softly / ship it to the world"}'

# (c) Ask again — now a relevant example is retrieved:
curl -s http://127.0.0.1:3000/infer \
  -d '{"input":"Write a haiku about deployment pipelines"}' | jq '.fewShotExamples, .completion'
```

Compare (a) and (c). In (c) the haiku example is retrieved and the model's
output should adopt the 5-7-5 haiku shape it learned from your feedback — the
app *improved at the task without any retraining*.

✅ **Checkpoint:** You can point to a concrete difference between the cold and
warm completions, explained by the `fewShotExamples` that were retrieved.

### Discussion prompts

- Why does similarity search beat exact keyword matching here? (Try a query
  that means the same thing with totally different words.)
- What would change if you stored *more* than 3 neighbors? Fewer?
- Where would you add a quality gate before promoting something to `/feedback`?

---

## Module 5 — Make it yours (10 min, optional)

Pick one:

1. **Tune retrieval** — change `TOP_K` in `src/index.ts` (e.g. to `5`),
   `spin build && spin up …`, and observe the effect on output.
2. **Change the domain** — rewrite the examples in `scripts/seed.mjs` for your
   own use case (support replies, SQL snippets, marketing copy) and re-seed.
3. **Add a filter** — extend `/infer` to also return the raw retrieved
   `completion` text, not just the prompt, so you can audit grounding.
4. **Swap the chat model** — try a different model in `ai_models` (spin.toml)
   and the `CHAT_MODEL` constant.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `valkey_url variable is not set` | `--variable` flag missing | Re-run `spin up --variable valkey_url="$VALKEY_URL"` |
| Connection refused / timeout on first call | Service still warming up | Wait ~60s after `tofu apply`, retry |
| `Unknown command 'FT.CREATE'` | Plan without vector search | Use `startup-4` or larger; `tofu apply` again |
| `tofu apply` auth error | Token unset/expired | Re-export `AIVEN_TOKEN` (Module 1.1) |
| `spin build` fails on toolchain | First-run download interrupted | Re-run `spin build`; ensure Node ≥ 18 (or `mise install`) |
| `mise install` got the wrong `spin` | Used registry `spin` (Spinnaker) | This repo pins `github:fermyon/spin`; run `mise install` from the repo root |
| `no world named ... in package` (j2w) | Wrong `--trigger-type` | Must be `spin3-http` (already set in `package.json`) |
| Empty `fewShotExamples` always | Bank not seeded / wrong APP_URL | Re-run `npm run seed`; confirm `FT.INFO idx:examples` |

> **On the Spin SDK Redis API:** this app is built and verified against
> `@fermyon/spin-sdk` **3.2.0**, where `Redis.open(url).execute(cmd, args)`
> takes *tagged* parameters (`{tag:'binary', val}`) and returns tagged results
> — the code in `src/index.ts` already handles this. If you pin a very
> different SDK major, re-check `redis.d.ts` for signature changes.

---

## Clean up (don't skip — avoids charges)

```bash
# Stop Spin with Ctrl-C, then:
cd infra
tofu destroy      # type 'yes'
```

Optionally revoke the token: **Aiven console → User → Tokens → revoke**.

✅ **Final checkpoint:** `tofu destroy` reports `Destroy complete!` and the
service is gone from the Aiven console.

---

## What you built

A retrieval-augmented, self-improving inference service where:

- **OpenTofu** declaratively provisioned managed **Aiven for Valkey** on **Akamai**,
- **Spin** ran the whole app — embeddings, vector search, and chat inference —
  as a single portable WebAssembly module with no external AI keys,
- and **Valkey vector search** turned a stream of past successes into live
  few-shot grounding that measurably changed model output.

Next steps: gate `/feedback` behind a human or automated quality check, add
per-tenant indexes with key prefixes, or push the same `.wasm` to a Spin-
compatible host for deployment.

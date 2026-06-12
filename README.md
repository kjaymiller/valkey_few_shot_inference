# Few-Shot Inference with Spin + Aiven for Valkey

An AI-forward demo built on a [Fermyon Spin](https://www.fermyon.com/spin)
WebAssembly component and **Aiven for Valkey** vector search. It combines two
independent capabilities, both powered by the same embedding + HNSW machinery:

1. **Few-shot retrieval** — embed each incoming prompt, find the 3 most similar
   *past successful* inferences, and inject them as few-shot examples before
   calling the inference engine. The system gets better at a task the more good
   examples it accumulates.
2. **Semantic response caching** — before calling the (slow) inference engine,
   check whether a semantically similar request has already been answered and,
   if so, return the cached completion instantly. See
   [Semantic response caching](#semantic-response-caching).

These are orthogonal: few-shot improves answer *quality*; caching improves
*latency and cost* on repeat questions. Each uses its own Valkey index.

```mermaid
flowchart TD
    infer["POST /infer<br/>{input}"]
    feedback["POST /feedback<br/>{input, completion}"]
    valkey[("Aiven for Valkey<br/>valkey-search / HNSW")]
    ollama["OpenAI-compatible chat API<br/>/v1/chat/completions<br/>(Ollama, OpenAI, Anthropic, ...)"]
    completion["completion"]

    subgraph spin["Spin component (Wasm)"]
        embed["1. embed input<br/>all-minilm-l6-v2 to 384-d vector"]
        cache["2. cache lookup<br/>KNN idx:cache (scoped to examples)"]
        search["3. FT.SEARCH KNN 3<br/>idx:examples"]
        prompt["4. build prompt with retrieved examples"]
        store["store completion in idx:cache"]
        fb_embed["embed + HSET into example bank"]
    end

    infer --> embed
    embed --> cache
    cache -->|hit| completion
    cache -->|miss| search
    search -->|vector search| valkey
    valkey -->|retrieved examples| prompt
    embed -.-> prompt
    prompt -->|"5. HTTP generate"| ollama
    ollama -->|generated text| store
    store --> completion
    cache <-->|KNN / write| valkey

    feedback --> fb_embed
    fb_embed -->|write back| valkey
```

## Layout

| Path | What |
|------|------|
| `infra/` | OpenTofu config that provisions Aiven for Valkey with the `valkey-search` capability. |
| `app/`   | Spin TypeScript HTTP component (`/infer`, `/infer/stream`, `/feedback`, `/cache/clear`, `/stats`, `/health`, and a test UI at `/`). |
| `app/scripts/seed.mjs` | Seeds the example bank with sample successful inferences. |

## Prerequisites

You'll need the following installed:

- [OpenTofu](https://opentofu.org) ≥ 1.6
- [Fermyon Spin](https://developer.fermyon.com/spin/install) ≥ 3.0 (serverless AI support)
- Node.js ≥ 18 (`mise.toml` pins node 22)

These versions are pinned in `mise.toml`. Optionally, if you use
[mise](https://mise.jdx.dev), one command installs all the correct versions:

```bash
mise install        # installs node 22, opentofu 1.x, fermyon spin 3.x
```

> [!NOTE]
> mise's registry `spin` is the *Spinnaker* CLI. This repo pins
> `github:fermyon/spin` (the WebAssembly runtime) to avoid the name clash.

You also need an [Aiven account](https://console.aiven.io) + API token
(`export AIVEN_TOKEN=…`).

## 1. Provision Aiven for Valkey

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # edit aiven_project, region
export AIVEN_TOKEN="$(aiven user access-token create --max-age-seconds 3600 --json | jq -r .full_token)"

tofu init
tofu apply

# Grab the connection string for the Spin app:
export VALKEY_URL="$(tofu output -raw valkey_service_uri)"
```

`valkey_service_uri` is a full `rediss://user:pass@host:port` URL. Aiven for
Valkey ships the `valkey-search` module on all plans, which is what powers the
`FT.CREATE` / `FT.SEARCH` (HNSW + KNN) commands below.

## 2. Build & run the Spin app

```bash
cd ../app
npm install
spin build
spin up --variable valkey_url="$VALKEY_URL"
# serving on http://127.0.0.1:3000
```

> The first `/infer` call lazily creates the `idx:examples` HNSW index
> (idempotently), so no separate index-setup step is needed.

The repo also ships `mise` task shortcuts that wrap the above:

```bash
mise run infra-up                      # tofu init + apply (in infra/)
export VALKEY_URL=$(mise run valkey-url)
mise run build                         # npm install + spin build (in app/)
mise run up                            # spin up with valkey_url wired in
mise run seed                          # seed the example bank
mise run infra-down                    # tofu destroy
```

## 3. Seed example "wins" and try it

```bash
# In a second terminal:
APP_URL=http://127.0.0.1:3000 npm run seed

curl -s http://127.0.0.1:3000/infer \
  -H 'content-type: application/json' \
  -d '{"input":"Write an out-of-office note for a conference trip"}' | jq
```

The response shows the completion plus which past examples were retrieved and
their cosine similarity — so you can see the few-shot grounding in action:

```json
{
  "input": "Write an out-of-office note for a conference trip",
  "completion": "Thanks for reaching out! I'm away at a conference until …",
  "fewShotExamples": [
    { "prompt": "Write a friendly out-of-office reply for a 2-week vacation.", "similarity": 0.83 }
  ],
  "usage": { "promptTokenCount": 142, "generatedTokenCount": 88 }
}
```

## 4. Teach it new wins

Any `(input, completion)` you consider successful can be promoted into the
example bank, where it becomes retrievable for future similar prompts:

```bash
curl -s http://127.0.0.1:3000/feedback \
  -H 'content-type: application/json' \
  -d '{"input":"…","completion":"…"}'
```

## Teardown

```bash
cd infra && tofu destroy
```

## How few-shot retrieval works

- **Embeddings**: Spin's built-in `all-minilm-l6-v2` serverless model produces
  384-dimensional vectors — no external embedding API.
- **Index**: `FT.CREATE idx:examples ON HASH PREFIX example: … VECTOR HNSW …
  DIM 384 DISTANCE_METRIC COSINE`.
- **Query**: `FT.SEARCH idx:examples "*=>[KNN 3 @embedding $vec AS score]"`
  returns the nearest neighbors; the app converts cosine *distance* to a
  similarity score for display.
- Vectors are stored as little-endian `FLOAT32` byte buffers in the
  `embedding` hash field, matching what `valkey-search` expects.

## Semantic response caching

Generating a completion is the slow, expensive part of a request. The app keeps
a **semantic cache** of past completions so that a repeat — or near-repeat —
question is answered from Valkey instead of the inference engine.

On every `/infer` (and `/infer/stream`), before calling the model the app:

1. Embeds the input (the same vector it already needs for few-shot retrieval).
2. Runs a KNN search against a **separate** cache index, `idx:cache`.
3. If the nearest entry's cosine similarity is at least `cache_threshold`
   **and** it was generated from the same retrieved examples, the stored
   completion is returned immediately (0 inference tokens).
4. On a miss, the model is called as usual and the result is written into
   `idx:cache` with a TTL.

The cache index is kept separate from `idx:examples` so cached raw answers never
leak into few-shot retrieval. Entries are keyed on input content **plus** a
fingerprint of the retrieved examples, so changing the example bank can't serve
a completion that was grounded in different context.

```bash
# First call: miss -> generates and caches.
curl -s http://127.0.0.1:3000/infer \
  -H 'content-type: application/json' \
  -d '{"input":"Draft a reminder about the budget deadline"}' | jq '.cached, .usage'
# false ... real token counts

# Same question again: hit -> instant, no tokens.
curl -s http://127.0.0.1:3000/infer \
  -H 'content-type: application/json' \
  -d '{"input":"Draft a reminder about the budget deadline"}' | jq '.cached, .cacheSimilarity'
# true 1
```

Cache controls:

- **Bypass** a single request with `{"input":"…","noCache":true}` (the lookup is
  skipped, but the fresh result is still cached).
- **Clear** the whole cache with `curl -X POST http://127.0.0.1:3000/cache/clear`
  — the example bank is untouched.
- **`/stats`** reports both `exampleCount` and `cacheCount`.

### Tuning

Two Spin variables control caching (see `spin.toml`):

| Variable | Default | Effect |
|----------|---------|--------|
| `cache_threshold` | `0.95` | Minimum cosine similarity for a hit. `1.0` = exact match only; lower is more aggressive (more hits, looser matches). |
| `cache_ttl_seconds` | `3600` | How long a cached completion lives. `0` disables expiry. |

> [!NOTE]
> Because entries are scoped to the retrieved examples, a dense example bank
> makes the cache hit mainly on near-exact repeats: a small wording change can
> shift which examples are retrieved and miss the cache. Raise `cache_threshold`
> toward `1.0` for stricter reuse, or lower it for more aggressive matching.

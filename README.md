# Few-Shot Inference with Spin + Aiven for Valkey

An AI-forward demo: a [Fermyon Spin](https://www.fermyon.com/spin) WebAssembly
component embeds each incoming prompt, queries **Aiven for Valkey** (running on
**Akamai**) for the 3 most similar *past successful* inferences using vector
search, and injects them as **few-shot examples** before calling the inference
engine. The system gets better at a task the more good examples it accumulates.

```
                 ┌─────────────────────────── Spin component (Wasm) ──────────────────────────┐
  POST /infer    │  1. embed(input)  ──all-minilm-l6-v2──▶ 384-d vector                        │
  {"input": …} ──▶  2. FT.SEARCH KNN 3  ───────────────▶  Aiven for Valkey (valkey-search/HNSW)│
                 │  3. build prompt w/ retrieved examples                                      │
                 │  4. Llm.infer(llama2-chat)  ──────────▶  completion                         │
                 └────────────────────────────────────────────────────────────────────────────┘
  POST /feedback {"input","completion"}  ──▶  embed + HSET into the example bank
```

## Layout

| Path | What |
|------|------|
| `infra/` | OpenTofu config that provisions Aiven for Valkey on Akamai with the `valkey-search` capability. |
| `app/`   | Spin TypeScript HTTP component (`/infer`, `/feedback`, `/health`). |
| `app/scripts/seed.mjs` | Seeds the example bank with sample successful inferences. |

## Prerequisites

The toolchain (OpenTofu, Fermyon Spin, Node) is pinned in `mise.toml`. With
[mise](https://mise.jdx.dev) installed, one command gets exact versions:

```bash
mise install        # installs node 22, opentofu 1.x, fermyon spin 3.x
```

> Heads-up: mise's registry `spin` is the *Spinnaker* CLI. This repo pins
> `github:fermyon/spin` (the WebAssembly runtime) to avoid the name clash.

Prefer not to use mise? Install manually:

- [OpenTofu](https://opentofu.org) ≥ 1.6
- [Fermyon Spin](https://developer.fermyon.com/spin/install) ≥ 3.0 (serverless AI support)
- Node.js ≥ 18

Either way you also need an [Aiven account](https://console.aiven.io) + API
token (`export AIVEN_TOKEN=…`).

## 1. Provision Valkey on Akamai

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

## How the vector search works

- **Embeddings**: Spin's built-in `all-minilm-l6-v2` serverless model produces
  384-dimensional vectors — no external embedding API.
- **Index**: `FT.CREATE idx:examples ON HASH PREFIX example: … VECTOR HNSW …
  DIM 384 DISTANCE_METRIC COSINE`.
- **Query**: `FT.SEARCH idx:examples "*=>[KNN 3 @embedding $vec AS score]"`
  returns the nearest neighbors; the app converts cosine *distance* to a
  similarity score for display.
- Vectors are stored as little-endian `FLOAT32` byte buffers in the
  `embedding` hash field, matching what `valkey-search` expects.

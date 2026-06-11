# Aiven for Valkey — a managed, Redis-compatible data store running on
# Akamai (Linode) cloud. We enable the valkey-search capability so the Spin
# app can build an HNSW vector index and run KNN similarity queries over the
# embeddings of past successful inferences.

resource "aiven_valkey" "few_shot" {
  project      = var.aiven_project
  cloud_name   = var.cloud_name
  plan         = var.valkey_plan
  service_name = var.service_name

  valkey_user_config {
    # Persist data so the few-shot example bank survives restarts.
    valkey_persistence = "rdb"

    # Keep evicted keys deterministic: never evict our example vectors.
    valkey_maxmemory_policy = "noeviction"

    # Allow the KEYS/SCAN-free FT.* search commands and longer-lived
    # connections used by the indexer script.
    valkey_timeout = 300

    public_access {
      valkey = true
    }
  }
}

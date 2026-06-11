output "valkey_service_uri" {
  description = "Full valkeys:// connection URI (host, port, user, password)."
  value       = aiven_valkey.few_shot.service_uri
  sensitive   = true
}

output "valkey_host" {
  description = "Valkey service hostname."
  value       = aiven_valkey.few_shot.service_host
}

output "valkey_port" {
  description = "Valkey service port."
  value       = aiven_valkey.few_shot.service_port
}

output "valkey_username" {
  description = "Default Valkey username."
  value       = aiven_valkey.few_shot.service_username
}

output "valkey_password" {
  description = "Default Valkey password."
  value       = aiven_valkey.few_shot.service_password
  sensitive   = true
}

output "spin_runtime_config_hint" {
  description = "How to feed the connection into the Spin app."
  value       = "Export VALKEY_URL with: export VALKEY_URL=$(tofu output -raw valkey_service_uri)"
}

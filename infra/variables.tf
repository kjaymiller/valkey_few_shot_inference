variable "aiven_api_token" {
  description = "Aiven API token. Prefer exporting AIVEN_TOKEN instead of setting this."
  type        = string
  default     = ""
  sensitive   = true
}

variable "aiven_project" {
  description = "Name of the existing Aiven project to deploy into."
  type        = string
}

variable "cloud_name" {
  description = "Cloud region for the service. Akamai (Linode) regions are prefixed with 'do-' replaced by Aiven's naming, e.g. akamai-us-east, akamai-eu-west."
  type        = string
  default     = "akamai-us-east"
}

variable "valkey_plan" {
  description = "Aiven for Valkey service plan. startup-4 is the smallest plan that ships the valkey-search (vector) capability."
  type        = string
  default     = "startup-4"
}

variable "service_name" {
  description = "Name for the Aiven for Valkey service."
  type        = string
  default     = "few-shot-valkey"
}

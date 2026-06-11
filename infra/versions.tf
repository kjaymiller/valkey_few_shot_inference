terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aiven = {
      source  = "aiven/aiven"
      version = ">= 4.0.0, < 5.0.0"
    }
  }
}

provider "aiven" {
  # Reads the token from the AIVEN_TOKEN environment variable when api_token
  # is left unset. Generate one with: aiven user access-token create
  api_token = var.aiven_api_token != "" ? var.aiven_api_token : null
}

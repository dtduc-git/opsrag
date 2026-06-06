# Acme Notes — infrastructure module (sample data for opsrag).
#
# A trimmed, provider-neutral-ish sketch of the Acme Notes data tier: a managed
# PostgreSQL instance with a read replica and an object-storage bucket for
# attachment uploads. Values are placeholders; this is illustrative corpus
# content, not a deployable module.

terraform {
  required_version = ">= 1.5"
}

variable "environment" {
  description = "Deployment environment (e.g. prod, staging)."
  type        = string
  default     = "prod"
}

variable "db_disk_gb" {
  description = "Primary database disk size in GB."
  type        = number
  default     = 200
}

# Primary PostgreSQL instance for Acme Notes.
resource "acme_db_instance" "acme_notes_primary" {
  name             = "acme-notes-${var.environment}"
  engine           = "postgres"
  engine_version   = "16"
  tier             = "db-custom-4-16384"
  disk_size_gb     = var.db_disk_gb
  high_availability = true

  # WAL archiving must keep up with write volume — see the 2026-01-15 outage
  # postmortem under samples/postmortems/.
  backup {
    enabled                = true
    point_in_time_recovery = true
    retention_days         = 14
  }
}

# Synchronous standby used for failover (see runbook 002).
resource "acme_db_replica" "acme_notes_standby" {
  name           = "acme-notes-${var.environment}-standby"
  primary_id     = acme_db_instance.acme_notes_primary.id
  replication    = "synchronous"
}

# Object storage for note attachments.
resource "acme_storage_bucket" "acme_notes_attachments" {
  name          = "acme-notes-attachments-${var.environment}"
  versioning    = true
  force_destroy = false

  lifecycle_rule {
    action_days = 365
    action      = "delete-noncurrent"
  }
}

output "primary_connection_name" {
  value = acme_db_instance.acme_notes_primary.name
}

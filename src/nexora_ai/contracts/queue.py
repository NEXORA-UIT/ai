"""Cross-language BullMQ queue and job names for the current contract version."""

QUEUE_NAME = "nexora-rag-ingestion"
PHASE0_PING_JOB_NAME = "phase0.ping"

QUEUE_PROTOCOL = {
    "queueName": QUEUE_NAME,
    "jobNames": {"phase0Ping": PHASE0_PING_JOB_NAME},
}

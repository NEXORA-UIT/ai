"""Stable configuration names and public messages for the internal API."""

from pathlib import Path

SERVICE_API_KEY_ENV = "AI_SERVICE_API_KEY"
SERVICE_API_KEY_SCHEME_NAME = "AIServiceBearer"
SERVICE_API_KEY_MISSING_DETAIL = "Invalid or missing service API key"
SERVICE_API_KEY_UNCONFIGURED_DETAIL = "AI service API key is not configured"
UPLOAD_DIRECTORY_ENV = "AI_UPLOAD_DIRECTORY"
DEFAULT_UPLOAD_DIRECTORY = Path(".local/uploads")
MAX_STORED_FILE_NAME_LENGTH = 255
UPLOAD_COPY_CHUNK_BYTES = 1024 * 1024

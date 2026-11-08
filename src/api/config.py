"""Stable configuration names and public messages for the internal API."""

from pathlib import Path

SERVICE_API_KEY_ENV = "AI_SERVICE_API_KEY"
SERVICE_API_KEY_SCHEME_NAME = "AIServiceBearer"
SERVICE_API_KEY_MISSING_DETAIL = "Invalid or missing service API key"
SERVICE_API_KEY_UNCONFIGURED_DETAIL = "AI service API key is not configured"
DATABASE_UNAVAILABLE_DETAIL = "AI database is unavailable"
UPLOAD_FILE_TOO_LARGE_DETAIL = "Uploaded file exceeds the size limit"
API_V1_PREFIX = "/v1"
DOCUMENTS_ROUTE_PATH = "/documents"
SEARCH_ROUTE_PATH = "/search"
ANSWER_ROUTE_PATH = "/answer"
UPLOAD_DIRECTORY_ENV = "AI_UPLOAD_DIRECTORY"
DEFAULT_UPLOAD_DIRECTORY = Path(".local/uploads")

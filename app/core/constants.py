"""
Application-wide constants.
Avoid hardcoding values throughout the project.
"""

# ==========================================
# API
# ==========================================

API_PREFIX = "/api/v1"
RAG_SIMILARITY_THRESHOLD = 0.55

# ==========================================
# Supported File Types
# ==========================================

SUPPORTED_FILE_EXTENSIONS = {
    # OCR (Mistral)
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
    ".webp",
    # Office
    ".docx",
    ".pptx",
    # Tabular
    ".csv",
    ".xlsx",
    # Text
    ".txt",
    ".md",
}

SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/webp",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/plain",
    "text/markdown",
}

# ==========================================
# Retrieval
# ==========================================

DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# ==========================================
# Chunking
# ==========================================

MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 2000

# ==========================================
# Upload
# ==========================================

MAX_FILE_SIZE_MB = 20

UPLOAD_FOLDER = "data/uploads"

# ==========================================
# Status Messages
# ==========================================

UPLOAD_SUCCESS = "Document uploaded successfully."

INDEX_SUCCESS = "Document indexed successfully."

DELETE_SUCCESS = "Document deleted successfully."

NO_DOCUMENT_FOUND = "No document found."

# ==========================================
# Metadata Keys
# ==========================================

SOURCE = "source"
PAGE = "page"
CHUNK_ID = "chunk_id"

# ==========================================
# API Tags (Swagger)
# ==========================================

CHAT_TAG = "Chat"
DOCUMENT_TAG = "Documents"
SYSTEM_TAG = "System"

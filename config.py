import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set in .env file")

# Chunking settings
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# Supported file extensions
SUPPORTED_EXTS = [".pdf", ".docx", ".txt", ".mp3", ".wav", ".m4a", ".flac", ".ogg"]

import os
from dotenv import load_dotenv
from semantic_cache import SemanticCache

class Config:
    """Configuration manager for the RAG application."""

    def __init__(self):
        """Initialize configuration by loading environment variables."""
        load_dotenv()

        # Gemini API key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set in .env file")

        # PostgreSQL/pgvector configuration
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = os.getenv("POSTGRES_PORT", "5432")
        self.postgres_db = os.getenv("POSTGRES_DB", "rag_vectorstore")
        self.postgres_user = os.getenv("POSTGRES_USER", "postgres")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD")
        if not self.postgres_password:
            raise ValueError("POSTGRES_PASSWORD environment variable not set in .env file")

        # Vector store collection name
        self.collection_name = os.getenv("COLLECTION_NAME", "document_embeddings")

        # Connection pool settings
        self.postgres_pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
        self.postgres_max_overflow = int(os.getenv("POSTGRES_MAX_OVERFLOW", "10"))

        # Chunking settings
        self.chunk_size = 300
        self.chunk_overlap = 80

        # Supported file extensions
        self.supported_exts = [".pdf", ".docx", ".txt", ".mp3", ".wav", ".m4a", ".flac", ".ogg"]

        # Semantic cache configuration
        self.cache_similarity_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))

        # Initialize semantic cache (lightweight, in-memory)
        self.semantic_cache = SemanticCache(similarity_threshold=self.cache_similarity_threshold)

        # Reranking configuration
        self.reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.retrieve_k = int(os.getenv("RETRIEVE_K", "20"))  # Candidates to retrieve before reranking
        self.final_k = int(os.getenv("FINAL_K", "8"))  # Final documents after reranking

    def get_connection_string(self):
        """Get PostgreSQL connection string."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

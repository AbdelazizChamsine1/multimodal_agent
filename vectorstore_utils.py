import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Integer, DateTime
from sqlalchemy.pool import QueuePool
import os
import hashlib
from datetime import datetime
 
 
class VectorStoreManager:
    """Manages pgvector vector stores, embeddings, and database connections."""
 
    def __init__(self, config, max_concurrent_stores=3):
        """Initialize the vector store manager.
 
        Args:
            config: Config instance (required)
            max_concurrent_stores: Maximum number of vectorstores to create concurrently
        """
        self.config = config
        self._connection_string = None
        self._embeddings = None
        self._engine = None
        self._metadata = None
        self.vectorstores = {}
        self.max_concurrent_stores = max_concurrent_stores
        max_workers = os.cpu_count() or max_concurrent_stores
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
 
    def get_connection_string(self):
        """Get PostgreSQL connection string ."""
        if self._connection_string is None:
            self._connection_string = self.config.get_connection_string()
        return self._connection_string
 
    def get_embeddings(self):
        """Get or create embeddings model singleton for efficient reuse."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings
 
    def get_engine(self):
        """Get or create SQLAlchemy engine with connection pooling."""
        if self._engine is None:
            self._engine = create_engine(
                self.get_connection_string(),
                poolclass=QueuePool,
                pool_size=self.config.postgres_pool_size,
                max_overflow=self.config.postgres_max_overflow,
                pool_pre_ping=True,  # Verify connections before using
                echo=False
            )
        return self._engine
 
    def init_vectorstore(self):
        """Initialize pgvector extension and create necessary tables."""
        engine = self.get_engine()
        with engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print(f"[INFO] pgvector extension enabled on database '{self.config.postgres_db}'")
 
        # Create metadata tracking table
        self._create_metadata_table()
 
    def _get_metadata(self):
        """Get or create SQLAlchemy MetaData object (singleton)."""
        if self._metadata is None:
            self._metadata = MetaData()
        return self._metadata
 
    def _create_metadata_table(self):
        """Create a table to track file hashes and modification times."""
        engine = self.get_engine()
        metadata = self._get_metadata()
 
        # Define file tracking table
        file_tracking = Table(
            'rag_file_tracking',
            metadata,
            Column('filename', String(255), primary_key=True),
            Column('file_hash', String(64), nullable=False),
            Column('chunk_count', Integer, nullable=False),
            Column('last_processed', DateTime, nullable=False),
            extend_existing=True
        )
 
        # Create table if not exists
        metadata.create_all(engine, checkfirst=True)
 
    def _compute_file_hash(self, file_path):
        """Compute SHA-256 hash of a file.
 
        Args:
            file_path: Path to file
 
        Returns:
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
 
    def _get_stored_file_hash(self, filename):
        """Get stored hash for a file from database.
 
        Args:
            filename: Name of file
 
        Returns:
            Hash string or None if file not tracked
        """
        engine = self.get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT file_hash FROM rag_file_tracking WHERE filename = :filename"),
                {"filename": filename}
            )
            row = result.fetchone()
            return row[0] if row else None
 
    def _update_file_tracking(self, filename, file_hash, chunk_count):
        """Update or insert file tracking record.
 
        Args:
            filename: Name of file
            file_hash: SHA-256 hash of file
            chunk_count: Number of chunks created
        """
        engine = self.get_engine()
        with engine.connect() as conn:
            # Upsert (PostgreSQL syntax)
            conn.execute(
                text("""
                    INSERT INTO rag_file_tracking (filename, file_hash, chunk_count, last_processed)
                    VALUES (:filename, :file_hash, :chunk_count, :last_processed)
                    ON CONFLICT (filename)
                    DO UPDATE SET
                        file_hash = :file_hash,
                        chunk_count = :chunk_count,
                        last_processed = :last_processed
                """),
                {
                    "filename": filename,
                    "file_hash": file_hash,
                    "chunk_count": chunk_count,
                    "last_processed": datetime.now()
                }
            )
            conn.commit()
 
    def check_file_needs_update(self, file_path, filename):
        """Check if a file needs to be reprocessed based on hash.
 
        Args:
            file_path: Full path to file
            filename: Name of file
 
        Returns:
            Tuple of (needs_update: bool, reason: str)
        """
        if not os.path.exists(file_path):
            return False, "File not found"
 
        # Compute current hash
        current_hash = self._compute_file_hash(file_path)
 
        # Get stored hash
        stored_hash = self._get_stored_file_hash(filename)
 
        if stored_hash is None:
            return True, "New file"
        elif current_hash != stored_hash:
            return True, "File modified"
        else:
            return False, "File unchanged"
 
    def sanitize_collection_name(self, filename):
        """Convert filename to valid PostgreSQL table name.
 
        Args:
            filename: Original filename
 
        Returns:
            Sanitized collection name
        """
        # Remove extension and replace invalid chars with underscores
        name = os.path.splitext(filename)[0]
        name = ''.join(c if c.isalnum() else '_' for c in name)
        return name.lower()[:50]  # Limit length
 
    async def create_per_file_vectorstores_async(self, chunks_by_file, folder_path):
        """Create pgvector vectorstores with per-file metadata (asynchronous with concurrency control).
        Uses incremental processing - only creates/updates vectorstores for files with chunks.
 
        Args:
            chunks_by_file: Dict mapping filename -> list of chunks (empty list = unchanged file)
            folder_path: Path to folder containing files (for hash checking)
 
        Returns:
            Dict mapping filename -> PGVector vectorstore instance
        """
        # self.init_vectorstore()
        embeddings = self.get_embeddings()
        connection_string = self.get_connection_string()
 
        # Load existing vectorstores for unchanged files
        self._load_existing_vectorstores(chunks_by_file)
 
        # Determine which files need vectorstore creation (files with chunks)
        files_to_process = []
 
        for filename, chunks in chunks_by_file.items():
            if chunks:  # Only process files that have chunks (newly processed files)
                file_path = os.path.join(folder_path, filename)
                files_to_process.append((filename, chunks, file_path))
 
        if not files_to_process:
            print("[INFO] No new vectorstores to create - all loaded from database!")
            return self.vectorstores
 
        print(f"[INFO] Creating {len(files_to_process)} vectorstores concurrently (max {self.max_concurrent_stores} at a time)...")
 
        # Semaphore to limit concurrent vectorstore creation
        semaphore = asyncio.Semaphore(self.max_concurrent_stores)
 
        async def create_single_vectorstore(filename, chunks, file_path):
            async with semaphore:
                loop = asyncio.get_event_loop()
 
                def create_store():
                    # Add filename to each chunk's metadata
                    for chunk in chunks:
                        chunk.metadata["filename"] = filename
 
                    # Create or use existing collection with filename as collection suffix
                    collection_name = f"{self.config.collection_name}_{self.sanitize_collection_name(filename)}"
 
                    # Create PGVector store from documents
                    # Note: PGVector creates its own tables internally
                    try:
                        vectorstore = PGVector.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            collection_name=collection_name,
                            connection=connection_string,
                            pre_delete_collection=True,  # Clear existing data for this file
                            use_jsonb=True  # Use JSONB for metadata (better performance)
                        )
                    except Exception as e:
                        if "already defined" in str(e):
                            # If table already exists, connect to existing collection
                            vectorstore = PGVector(
                                embedding_function=embeddings,
                                collection_name=collection_name,
                                connection=connection_string,
                                use_jsonb=True
                            )
                            # Clear and re-add documents
                            vectorstore.delete_collection()
                            vectorstore = PGVector.from_documents(
                                documents=chunks,
                                embedding=embeddings,
                                collection_name=collection_name,
                                connection=connection_string,
                                use_jsonb=True
                            )
                        else:
                            raise
 
                    # Update file tracking with hash
                    file_hash = self._compute_file_hash(file_path)
                    self._update_file_tracking(filename, file_hash, len(chunks))
 
                    print(f"[INFO] Created pgvector store for '{filename}' ({len(chunks)} chunks)")
                    return filename, vectorstore
 
                return await loop.run_in_executor(self._executor, create_store)
 
        # Create all vectorstores concurrently
        results = await asyncio.gather(*[create_single_vectorstore(filename, chunks, file_path)
                                        for filename, chunks, file_path in files_to_process])
 
        # Build vectorstores dictionary
        for filename, vectorstore in results:
            self.vectorstores[filename] = vectorstore
 
        return self.vectorstores
 
    def _load_existing_vectorstores(self, chunks_by_file):
        """Load existing vectorstores for files that haven't changed.
 
        Args:
            chunks_by_file: Dict mapping filename -> list of chunks
        """
        embeddings = self.get_embeddings()
        connection_string = self.get_connection_string()
        engine = self.get_engine()
 
        for filename in chunks_by_file.keys():
            # Check if collection exists in database
            collection_name = f"{self.config.collection_name}_{self.sanitize_collection_name(filename)}"
 
            try:
                # Verify collection table exists in database
                with engine.connect() as conn:
                    result = conn.execute(
                        text("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables
                                WHERE table_schema = 'public'
                                AND table_name = :table_name
                            )
                        """),
                        {"table_name": f"langchain_pg_collection"}
                    )
                    tables_exist = result.scalar()
 
                    if not tables_exist:
                        # Tables not created yet
                        continue
 
                    # Check if this specific collection exists
                    result = conn.execute(
                        text("""
                            SELECT EXISTS (
                                SELECT 1 FROM langchain_pg_collection
                                WHERE name = :collection_name
                            )
                        """),
                        {"collection_name": collection_name}
                    )
                    collection_exists = result.scalar()
 
                if collection_exists:
                    # Load existing vectorstore
                    vectorstore = PGVector(
                        embeddings=embeddings,
                        collection_name=collection_name,
                        connection=connection_string,
                        use_jsonb=True
                    )
                    self.vectorstores[filename] = vectorstore
                    print(f"[INFO] Loaded existing vectorstore for '{filename}'")
 
            except Exception as e:
                # Collection doesn't exist or is invalid - will be created later
                print(f"[DEBUG] Could not load vectorstore for '{filename}': {e}")
                pass
 
    def get_available_files(self):
        """Get list of available files from vectorstores.
 
        Returns:
            List of filenames
        """
        return sorted(list(self.vectorstores.keys()))

    def cleanup(self):
        """Clean up resources (thread pool executor)."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=True)
            print("[INFO] VectorStoreManager thread pool shut down")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.cleanup()
 
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import whisper
from colorama import Fore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader, TextLoader
 
 
class DocumentProcessor:
    """Handles document and audio file processing, transcription, and chunking."""
 
    def __init__(self, config, max_concurrent_files=5):
        """Initialize the document processor.
 
        Args: 
            config: Config instance (required)
            max_concurrent_files: Maximum number of files to process concurrently
        """
        self.config = config
        self._whisper_model = None
        self.max_concurrent_files = max_concurrent_files
        max_workers = os.cpu_count() or max_concurrent_files
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
 
    def get_whisper_model(self):
        """Get or create Whisper model singleton for efficient reuse."""
        if self._whisper_model is None:
            print(Fore.YELLOW + "[INFO] Loading Whisper 'base' model (first time only)...")
            self._whisper_model = whisper.load_model("base")
            print(Fore.GREEN + "[INFO] Whisper model loaded successfully")
        return self._whisper_model
 
    def _transcribe_audio_sync(self, file_path):
        """Internal synchronous transcription method used by async wrapper.
 
        Args:
            file_path: Path to audio file
 
        Returns:
            Transcribed text string
 
        Raises:
            ValueError: If no speech could be transcribed
        """
        print(Fore.YELLOW + f"[INFO] Transcribing audio: {file_path} (this may take a while)...")
        model = self.get_whisper_model()
        result = model.transcribe(file_path, fp16=False)
        transcript = result["text"].strip()
        if not transcript:
            raise ValueError(f"No speech could be transcribed from {file_path}")
        print(Fore.GREEN + f"[INFO] Transcription complete ({len(transcript)} characters)")
        return transcript
 
    async def transcribe_audio_async(self, file_path):
        """Transcribe audio file asynchronously using thread executor.
 
        Args:
            file_path: Path to audio file
 
        Returns:
            Transcribed text string
 
        Raises:
            ValueError: If no speech could be transcribed
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._transcribe_audio_sync, file_path)
 
    async def load_and_chunk_file_async(self, file_path):
        """Load a single document or audio file and split it into chunks (asynchronous).
 
        Args:
            file_path: Path to file to process
 
        Returns:
            List of LangChain Document chunks
 
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is unsupported or no content loaded
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
 
        ext = os.path.splitext(file_path)[1].lower()
        try:
            # Audio files - use async transcription
            if ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
                transcript = await self.transcribe_audio_async(file_path)
                docs = [Document(page_content=transcript, metadata={"source": file_path})]
 
            # Documents - run in executor to avoid blocking
            else:
                loop = asyncio.get_event_loop()
 
                def load_document():
                    if ext == ".docx":
                        loader = UnstructuredWordDocumentLoader(file_path)
                    elif ext == ".pdf":
                        loader = PyPDFLoader(file_path)
                    elif ext == ".txt":
                        loader = TextLoader(file_path)
                    else:
                        raise ValueError(f"Unsupported file type: {ext}")
                    return loader.load()
 
                docs = await loop.run_in_executor(self._executor, load_document)
 
            if not docs:
                raise ValueError(f"No content loaded from {file_path}")
 
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", "!", "?"]
            )
            chunks = splitter.split_documents(docs)
            return chunks
 
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to process {file_path}: {e}")
            raise
 
    async def process_folder_by_file_async(self, folder_path, vector_manager=None):
        """Scan a folder, process files concurrently, and return chunks grouped by filename.
        Uses incremental processing - only processes new or modified files if vector_manager is provided.
 
        Args:
            folder_path: Path to folder containing documents
            vector_manager: Optional VectorStoreManager for incremental processing
 
        Returns:
            Dict mapping filename -> list of chunks
 
        Raises:
            FileNotFoundError: If folder doesn't exist
            ValueError: If no supported files found
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
 
        files = [f for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1].lower() in self.config.supported_exts]
 
        if not files:
            raise ValueError(f"No supported files found in {folder_path}")
 
        print(Fore.YELLOW + f"[INFO] Scanning folder: {folder_path}")
 
        chunks_by_file = {}
 
        # Determine which files need processing (if vector_manager provided)
        files_to_process = []
        files_skipped = []
 
        if vector_manager:
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                needs_update, reason = vector_manager.check_file_needs_update(file_path, file_name)
 
                if needs_update:
                    files_to_process.append(file_name)
                else:
                    files_skipped.append(file_name)
                    # Initialize empty chunks for unchanged files (will be loaded from DB)
                    chunks_by_file[file_name] = []
 
            if files_skipped:
                print(Fore.YELLOW + f"[INFO] Skipping {len(files_skipped)} unchanged files: {', '.join(files_skipped)}")
 
            if not files_to_process:
                print(Fore.GREEN + "[INFO] All files are up to date - no processing needed!")
                return chunks_by_file
        else:
            # No vector manager - process all files
            files_to_process = files
 
        print(Fore.YELLOW + f"[INFO] Processing {len(files_to_process)} files concurrently (max {self.max_concurrent_files} at a time)...")
 
        # Process files with concurrency limit using semaphore
        semaphore = asyncio.Semaphore(self.max_concurrent_files)
 
        async def process_single_file(file_name):
            async with semaphore:
                file_path = os.path.join(folder_path, file_name)
                print(Fore.CYAN + f"[INFO] Processing file: {file_name}")
                try:
                    chunks = await self.load_and_chunk_file_async(file_path)
                    print(Fore.GREEN + f"[INFO] Added {len(chunks)} chunks from {file_name}")
                    return file_name, chunks
                except Exception as e:
                    print(Fore.RED + f"[ERROR] Skipping {file_name}: {e}")
                    return file_name, []
 
        # Process only the files that need processing
        results = await asyncio.gather(*[process_single_file(file_name) for file_name in files_to_process])
 
        # Build the dictionary from results
        for file_name, chunks in results:
            chunks_by_file[file_name] = chunks
 
        total_chunks = sum(len(chunks) for chunks in chunks_by_file.values())
        print(Fore.GREEN + f"[INFO] Finished processing {len(files_to_process)} files. Total chunks: {total_chunks}")
        return chunks_by_file
 
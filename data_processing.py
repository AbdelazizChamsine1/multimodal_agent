import os
import whisper
from colorama import Fore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader, TextLoader
from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTS

def transcribe_audio(file_path):
    """Transcribe audio file to text using OpenAI Whisper."""
    print(Fore.YELLOW + f"[INFO] Transcribing audio: {file_path} (this may take a while)...")
    model = whisper.load_model("base")
    result = model.transcribe(file_path, fp16=False)
    transcript = result["text"].strip()
    if not transcript:
        raise ValueError(f"No speech could be transcribed from {file_path}")
    print(Fore.GREEN + f"[INFO] Transcription complete ({len(transcript)} characters)")
    return transcript

def load_and_chunk_file(file_path):
    """Load a single document or audio file and split it into chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    try:
        # Audio files
        if ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
            transcript = transcribe_audio(file_path)
            docs = [Document(page_content=transcript, metadata={"source": file_path})]

        # Documents
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext == ".txt":
            loader = TextLoader(file_path)
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if not docs:
            raise ValueError(f"No content loaded from {file_path}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", "!", "?"]
        )
        chunks = splitter.split_documents(docs)
        return chunks

    except Exception as e:
        print(Fore.RED + f"[ERROR] Failed to process {file_path}: {e}")
        raise

def process_folder(folder_path):
    """Scan a folder, process all supported files, and return all chunks."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_chunks = []
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]

    if not files:
        raise ValueError(f"No supported files found in {folder_path}")

    print(Fore.YELLOW + f"[INFO] Scanning folder: {folder_path}")
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        print(Fore.CYAN + f"[INFO] Processing file: {file_name}")
        try:
            chunks = load_and_chunk_file(file_path)
            all_chunks.extend(chunks)
            print(Fore.GREEN + f"[INFO] Added {len(chunks)} chunks from {file_name}")
        except Exception as e:
            print(Fore.RED + f"[ERROR] Skipping {file_name}: {e}")

    print(Fore.GREEN + f"[INFO] Finished processing {len(files)} files. Total chunks: {len(all_chunks)}")
    return all_chunks

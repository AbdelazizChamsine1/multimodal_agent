# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) based document and audio Q&A system. It processes documents (PDF, DOCX, TXT) and audio files (MP3, WAV, M4A, FLAC, OGG), creates vector embeddings, and enables conversational question-answering using Google's Gemini LLM.

## Architecture

### Core Pipeline Flow

1. **Document Ingestion** (data_processing.py)
   - Audio files are transcribed using OpenAI Whisper (base model)
   - Documents are loaded using LangChain loaders
   - All content is chunked using RecursiveCharacterTextSplitter
   - Chunks are stored as LangChain Document objects

2. **Vector Storage** (vectorstore_utils.py)
   - Embeddings: HuggingFace's "sentence-transformers/all-MiniLM-L6-v2"
   - Vector store: FAISS (saved locally to `vectorstore/` directory)
   - Persistent storage allows skipping re-processing on subsequent runs

3. **Q&A Chain** (qa_chain.py)
   - LLM: Google Gemini "gemini-2.0-flash-exp" (temperature=0.2)
   - Retriever: Top-5 similar chunks from vector store
   - Chat history: Last 10 messages (5 Q&A pairs) maintained for context
   - LangChain LCEL syntax for chain composition

4. **Main Loop** (main.py)
   - Loads or creates vector store on startup
   - Interactive CLI for asking questions
   - Maintains conversation history for contextual responses

### Module Responsibilities

- **config.py**: Environment variables, chunking parameters (CHUNK_SIZE=200, CHUNK_OVERLAP=50), supported file extensions
- **data_processing.py**: File loading, audio transcription, text chunking
- **vectorstore_utils.py**: FAISS vector store creation, saving, and loading
- **qa_chain.py**: LangChain QA chain setup with Gemini LLM
- **main.py**: Application entry point and interactive loop

## Development Commands

### Running the Application

```bash
python main.py
```

The application will:
- Check for existing vectorstore in `vectorstore/` directory
- If not found, process all files in `files_folder/` and create vectorstore
- Start interactive Q&A session (type 'exit', 'quit', or 'q' to quit)

### Environment Setup

Required environment variables in `.env`:
```
GEMINI_API_KEY=your_api_key_here
```

### Dependencies

Key dependencies (install via pip):
- langchain, langchain-community, langchain-google-genai, langchain-huggingface
- faiss-cpu (or faiss-gpu)
- openai-whisper
- sentence-transformers
- colorama
- python-dotenv
- Document loaders: unstructured, pypdf, docx

### Testing File Processing

Place test files in `files_folder/` directory. Supported formats:
- Documents: .pdf, .docx, .txt
- Audio: .mp3, .wav, .m4a, .flac, .ogg

To force reprocessing, delete the `vectorstore/` directory before running.

## Important Implementation Notes

### Audio Processing
- Whisper model loads on-demand per audio file (data_processing.py:12)
- Uses "base" model with fp16=False for CPU compatibility
- Transcription can be slow for long audio files

### Vector Store Persistence
- FAISS index is saved locally to avoid reprocessing
- Loading uses `allow_dangerous_deserialization=True` (vectorstore_utils.py:16)
- Delete `vectorstore/` to rebuild from scratch

### Chunking Strategy
- Small chunk size (200) with overlap (50) for granular retrieval
- Separator hierarchy: "\n\n" > "\n" > ". " > " " > "!" > "?"
- Configured in config.py, used in data_processing.py:48-52

### Chat History Management
- Maintains last 5 Q&A pairs (10 messages total) in main.py:46-47
- History passed to QA chain for contextual follow-up questions
- Uses LangChain HumanMessage/AIMessage objects

### Error Handling
- File processing errors are logged but don't stop folder processing (data_processing.py:80)
- Query errors are caught and displayed without crashing the session (main.py:50)
- Missing API key raises ValueError at startup (config.py:10)

## Modifying the System

### Changing the LLM
Edit qa_chain.py:11-15 to use a different model or provider. Update temperature for different response styles (lower = more deterministic).

### Adjusting Retrieval
Change `k` value in qa_chain.py:17 to retrieve more/fewer context chunks.

### Chunking Configuration
Modify CHUNK_SIZE and CHUNK_OVERLAP in config.py. Larger chunks = more context per chunk but less precision.

### Adding File Types
Add extension to SUPPORTED_EXTS in config.py and add loader logic in data_processing.py:20-43.

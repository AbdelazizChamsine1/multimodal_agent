# RAG System with pgvector and Gemini

A Retrieval-Augmented Generation (RAG) system that processes documents and audio files, stores embeddings in PostgreSQL with pgvector, and answers questions using Google's Gemini LLM.

## Features

- **Multi-format Document Processing**: Supports PDF, DOCX, TXT files
- **Audio Transcription**: Processes MP3, WAV, M4A, FLAC, OGG files using OpenAI Whisper
- **Vector Storage**: PostgreSQL with pgvector extension for efficient similarity search
- **Semantic Caching**: Reduces API calls by caching similar queries
- **Reranking**: Improves retrieval quality with cross-encoder reranking
- **Concurrent Processing**: Async file processing and vectorstore creation
- **Incremental Updates**: Only reprocesses modified files based on hash tracking

## Prerequisites

- **Python 3.8+**
- **Docker & Docker Compose**
- **FFmpeg** (required for audio processing)

### Installing FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag_system
```

### 2. Set Up Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
- Get your API key from: https://makersuite.google.com/app/apikey
- Replace `your_gemini_api_key_here` with your actual API key
- Update `POSTGRES_PASSWORD` if desired

### 3. Start PostgreSQL with pgvector (Docker)

```bash
docker-compose up -d
```

This will start a PostgreSQL 18 container with pgvector extension on port **5434**.

**Note:** Port 5434 is used to avoid conflicts with local PostgreSQL installations. If you need a different port, update both `docker-compose.yml` and `.env` file.

### 4. Create Python Virtual Environment

```bash
python -m venv venv
```

**Activate the virtual environment:**

**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 6. Initialize the Database

```bash
python init_db.py
```

This will:
- Create the `rag_vectorstore` database
- Enable the pgvector extension
- Set up metadata tracking tables

### 7. Add Your Documents

Place your files in the `files_folder/` directory:
- Supported formats: `.pdf`, `.docx`, `.txt`, `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`

### 8. Run the Application

```bash
python main.py
```

The system will:
1. Process all files in `files_folder/`
2. Create embeddings and store them in PostgreSQL
3. Start an interactive Q&A session

## Usage

### Interactive Q&A Mode

Once the application starts, you can ask questions:

```
Enter your question (or 'exit' to quit): What is the main topic discussed in the documents?
```

The system will:
- Check the semantic cache for similar questions
- Retrieve relevant document chunks using vector similarity
- Rerank results for better accuracy
- Generate an answer using Gemini LLM
- Provide source references

### Adding New Documents

Simply add files to `files_folder/` and run:

```bash
python main.py
```

The system intelligently:
- Detects new or modified files using hash comparison
- Only processes changed files
- Preserves existing vectorstores for unchanged files

## Project Structure

```
rag_system/
├── config.py                 # Configuration management
├── data_processing.py        # Document/audio processing
├── vectorstore_utils.py      # Vector store management
├── semantic_cache.py         # Query caching
├── qa_chain.py              # Q&A system with Gemini
├── init_db.py               # Database initialization
├── main.py                  # Main application
├── docker-compose.yml       # PostgreSQL container config
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create from .env.example)
├── .env.example            # Environment template
└── files_folder/           # Your documents go here
```

## Configuration

All configuration is managed through environment variables in `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | - | Google Gemini API key |
| `POSTGRES_USER` | ✅ Yes | postgres | Database username |
| `POSTGRES_PASSWORD` | ✅ Yes | - | Database password |
| `POSTGRES_DB` | No | rag_vectorstore | Database name |
| `POSTGRES_HOST` | No | localhost | Database host |
| `POSTGRES_PORT` | No | 5434 | Database port |
| `COLLECTION_NAME` | No | document_embeddings | Vector collection prefix |
| `POSTGRES_POOL_SIZE` | No | 5 | Connection pool size |
| `POSTGRES_MAX_OVERFLOW` | No | 10 | Max overflow connections |
| `CACHE_SIMILARITY_THRESHOLD` | No | 0.85 | Semantic cache threshold |
| `RERANKER_MODEL` | No | cross-encoder/ms-marco-MiniLM-L-12-v2 | Reranker model |
| `RETRIEVE_K` | No | 20 | Documents to retrieve |
| `FINAL_K` | No | 8 | Documents after reranking |

## Troubleshooting

### Port Conflicts

If you have a local PostgreSQL running on port 5434:

1. Change the port in `docker-compose.yml`:
   ```yaml
   ports:
     - "5435:5432"  # Use a different port
   ```

2. Update `.env`:
   ```
   POSTGRES_PORT=5435
   ```

3. Restart Docker:
   ```bash
   docker-compose down
   docker-compose up -d
   python init_db.py
   ```

### Audio Processing Errors

If you see `[WinError 2]` or FFmpeg errors:
- Install FFmpeg (see Prerequisites section)
- Ensure FFmpeg is in your system PATH
- Restart your terminal after installation

### Database Connection Errors

```bash
# Check if Docker container is running
docker ps

# View container logs
docker logs pgvector-18-custom

# Restart container
docker-compose restart
```

### Extension Not Available Error

```bash
# Connect to container and verify pgvector
docker exec pgvector-18-custom psql -U postgres -d rag_vectorstore -c "\dx"
```

You should see the `vector` extension listed.

## Performance Tips

1. **Concurrent Processing**: Adjust `max_concurrent_files` and `max_concurrent_stores` in `main.py`
2. **Chunk Size**: Modify `chunk_size` and `chunk_overlap` in `config.py`
3. **Retrieval Settings**: Tune `RETRIEVE_K` and `FINAL_K` for better accuracy
4. **Cache Threshold**: Adjust `CACHE_SIMILARITY_THRESHOLD` to control cache hits

## Technologies Used

- **LangChain**: Document processing and RAG pipeline
- **Google Gemini**: Large Language Model
- **PostgreSQL + pgvector**: Vector database
- **OpenAI Whisper**: Audio transcription
- **HuggingFace Transformers**: Embeddings and reranking
- **SQLAlchemy**: Database ORM
- **Docker**: Containerization


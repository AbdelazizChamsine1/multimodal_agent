import asyncio
from config import Config
from vectorstore_utils import VectorStoreManager
from data_processing import DocumentProcessor
from qa_chain import QASystem
from colorama import Fore, Style
import sys
 
 
class RAGApplication:
    """Main RAG application that orchestrates document processing and Q&A."""
 
    def __init__(self, folder_path="files_folder", max_concurrent_files=5, max_concurrent_stores=3):
        """Initialize the RAG application.
 
        Args:
            folder_path: Path to folder containing documents
            max_concurrent_files: Maximum number of files to process concurrently
            max_concurrent_stores: Maximum number of vectorstores to create concurrently
        """
        self.folder_path = folder_path
        self.config = Config()
 
        # Initialize components
        self.doc_processor = DocumentProcessor(self.config, max_concurrent_files=max_concurrent_files)
        self.vector_manager = VectorStoreManager(self.config, max_concurrent_stores=max_concurrent_stores)
        self.qa_system = None
 
    async def initialize_async(self):
        """Initialize the application by processing documents and creating vector stores (asynchronous)."""
        print(Fore.YELLOW + "[INFO] Processing files and creating per-file pgvector stores (async mode)...")
 
        # Initialize vectorstore (creates tables)
        self.vector_manager.init_vectorstore()
 
        # Process documents concurrently (with incremental updates - checks hashes first!)
        chunks_by_file = await self.doc_processor.process_folder_by_file_async(
            self.folder_path,
            vector_manager=self.vector_manager
        )
 
        # Create vector stores concurrently (only for new/modified files)
        vectorstores = await self.vector_manager.create_per_file_vectorstores_async(chunks_by_file, self.folder_path)
        print(Fore.GREEN + "[INFO] pgvector stores ready in PostgreSQL!")
 
        # Initialize QA system
        self.qa_system = QASystem(self.vector_manager, self.config)
        self.qa_system.create_qa_chain()
 
        return vectorstores
 
    def display_indexed_files(self):
        """Display the list of indexed files."""
        available_files = self.vector_manager.get_available_files()
        print(Fore.GREEN + "\nDocument/Audio Q&A system ready!")
        print(Fore.CYAN + f"\nIndexed files ({len(available_files)}):")
 
        for i, filename in enumerate(available_files, 1):
            # Identify file type
            ext = filename.split('.')[-1].lower()
            if ext in ['mp3', 'wav', 'm4a', 'flac', 'ogg']:
                file_type = "Audio"
            elif ext == 'pdf':
                file_type = "PDF"
            elif ext == 'docx':
                file_type = "Word"
            elif ext == 'txt':
                file_type = "Text"
            else:
                file_type = "Document"
            print(f"  {i}. {filename} ({file_type})")
 
        print(Fore.YELLOW + "\nTip: Be specific about which file you're asking about (e.g., 'What does the audio say about...?')")
        print(Fore.YELLOW + "     The system will prioritize information from the file you mention.\n")
 
    def run_interactive_loop(self):
        """Run the interactive Q&A loop."""
        while True:
            query = input(Style.RESET_ALL + "Ask a question (or type 'exit'): ")
 
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
 
            if not query.strip():
                continue
 
            try:
                # Use streaming for real-time response display
                print(Fore.GREEN + "\nAnswer: ", end="", flush=True)
 
                full_response = ""
                for chunk in self.qa_system.ask(query, stream=True):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                    sys.stdout.flush()  # Ensure immediate output
 
                print("\n")  # New line after complete response
 
                # Update history
                self.qa_system.update_history(query, full_response)
 
            except Exception as e:
                print(Fore.RED + f"\nError processing query: {e}")
 
    def run(self):
        """Run the complete application workflow."""
        try:
            # Run async initialization
            asyncio.run(self.initialize_async())
 
            self.display_indexed_files()
            self.run_interactive_loop()
        except Exception as e:
            print(Fore.RED + f"Application error: {e}")
 
 
if __name__ == "__main__":
    app = RAGApplication(folder_path="files_folder")
    app.run()
 
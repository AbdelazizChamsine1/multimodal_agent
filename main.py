from config import GEMINI_API_KEY
from data_processing import process_folder
from vectorstore_utils import create_vectorstore, save_vectorstore, load_vectorstore, get_available_files
from qa_chain import create_qa_chain
from colorama import Fore, Style
from langchain_core.messages import HumanMessage, AIMessage
import os

if __name__ == "__main__":
    try:
        folder_path = "files_folder"
        vectorstore_path = "vectorstore"
        
        # Load existing vectorstore or process folder
        if os.path.exists(vectorstore_path):
            print(Fore.YELLOW + f"[INFO] Loading existing vector store from '{vectorstore_path}'")
            vectorstore = load_vectorstore(vectorstore_path)
        else:
            print(Fore.YELLOW + "[INFO] Processing folder and creating vector store...")
            chunks = process_folder(folder_path)
            vectorstore = create_vectorstore(chunks)
            save_vectorstore(vectorstore, vectorstore_path)

        # Display available files
        available_files = get_available_files(vectorstore)
        qa_chain = create_qa_chain(vectorstore, GEMINI_API_KEY, available_files)
        chat_history = []
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

        while True:
            query = input(Style.RESET_ALL + "Ask a question (or type 'exit'): ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            if not query.strip():
                continue

            try:
                response = qa_chain.invoke({"question": query, "chat_history": chat_history})
                print(Fore.GREEN + "\nAnswer:", response, "\n")

                # Update history
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=response))

                # Keep last 5 Q&A pairs
                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]

            except Exception as e:
                print(Fore.RED + f"Error processing query: {e}")

    except Exception as e:
        print(Fore.RED + f"Initialization error: {e}")

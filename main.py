from config import GEMINI_API_KEY
from data_processing import process_folder
from vectorstore_utils import create_vectorstore, save_vectorstore, load_vectorstore
from qa_chain import create_qa_chain
from colorama import Fore
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

        qa_chain = create_qa_chain(vectorstore, GEMINI_API_KEY)
        chat_history = []

        print(Fore.GREEN + "\nDocument/Audio Q&A system ready!\n")

        while True:
            query = input("Ask a question (or type 'exit'): ")
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

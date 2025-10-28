from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from colorama import Fore
import os

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def save_vectorstore(vectorstore, path="vectorstore"):
    vectorstore.save_local(path)
    print(Fore.GREEN + f"[INFO] Vector store saved to '{path}'")

def load_vectorstore(path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print(Fore.GREEN + f"[INFO] Vector store loaded from '{path}'")
    return vectorstore

def get_available_files(vectorstore):
    """Get list of unique source files in the vectorstore."""
    files = set()
    # Access the underlying documents through the docstore
    for doc_id in vectorstore.docstore._dict.keys():
        doc = vectorstore.docstore._dict[doc_id]
        source = doc.metadata.get("source", "")
        if source:
            filename = os.path.basename(source)
            files.add(filename)
    return sorted(list(files))

def create_filtered_retriever(vectorstore, filename=None, k=5):
    """Create a retriever optionally filtered by filename."""
    if filename:
        # Create retriever with metadata filter
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": lambda metadata: os.path.basename(metadata.get("source", "")) == filename
            }
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from colorama import Fore

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

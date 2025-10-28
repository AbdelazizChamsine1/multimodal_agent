from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os
import re

def format_docs(docs):
    """Format documents with source file information."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        source_name = source.split("\\")[-1].split("/")[-1]  # Get filename only
        formatted.append(f"[From: {source_name}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def detect_file_mention(question, available_files):
    """Detect if user is asking about a specific file or file type."""
    question_lower = question.lower()

    # File type mentions
    file_type_map = {
        'audio': ['.mp3', '.wav', '.m4a', '.flac', '.ogg'],
        'pdf': ['.pdf'],
        'document': ['.docx'],
        'text': ['.txt'],
        'word': ['.docx']
    }

    mentioned_files = []

    # Check for specific filename mentions
    for filename in available_files:
        filename_lower = filename.lower()
        # Check if filename (without extension) is mentioned
        name_without_ext = os.path.splitext(filename_lower)[0]
        if name_without_ext in question_lower or filename_lower in question_lower:
            mentioned_files.append(filename)

    # Check for file type mentions (e.g., "the audio", "the PDF")
    if not mentioned_files:
        for file_type, extensions in file_type_map.items():
            if file_type in question_lower:
                for filename in available_files:
                    if any(filename.lower().endswith(ext) for ext in extensions):
                        mentioned_files.append(filename)

    return mentioned_files

def retrieve_with_file_priority(vectorstore, question, available_files, k=10):
    """Retrieve documents with priority given to mentioned files."""
    mentioned_files = detect_file_mention(question, available_files)

    if mentioned_files:
        # Retrieve more documents initially
        retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})
        all_docs = retriever.invoke(question)

        # Separate docs by whether they're from mentioned files
        priority_docs = []
        other_docs = []

        for doc in all_docs:
            source = doc.metadata.get("source", "")
            filename = os.path.basename(source)
            if filename in mentioned_files:
                priority_docs.append(doc)
            else:
                other_docs.append(doc)

        # Return mostly priority docs, with some other docs for context
        if priority_docs:
            # Take up to k-2 from priority, and 2 from others
            result = priority_docs[:k-2] + other_docs[:2]
            return result[:k]

    # Default: no specific file mentioned, use standard retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)

def create_qa_chain(vectorstore, api_key, available_files):
    """Create the QA chain using Gemini and a FAISS vectorstore with smart file-aware retrieval."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.2,
        google_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant answering questions about documents and audio transcripts.
        Use the following context from the documents/transcripts and the conversation history to answer questions.

        IMPORTANT: Each piece of context shows which file it came from using [From: filename].
        - Pay close attention to which file the user is asking about
        - When answering, prioritize information from the relevant file
        - If the user asks about a specific file (e.g., "the audio", "the PDF", "business.txt"), focus on context from that file
        - If multiple files contain relevant information, mention which file each piece of information comes from

        If you don't know the answer based on the context, just say that you don't know.

        Context from documents: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    qa_chain = (
        {
            "context": lambda x: format_docs(
                retrieve_with_file_priority(vectorstore, x["question"], available_files, k=8)
            ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain

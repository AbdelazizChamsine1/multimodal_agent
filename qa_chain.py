from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from colorama import Fore, Style
import os
import time


class QASystem:
    """Manages the question-answering system using LLM and vector stores."""

    def __init__(self, vector_store_manager, config):
        """Initialize the QA system.

        Args:
            vector_store_manager: VectorStoreManager instance
            config: Config instance (required)
        """
        self.config = config
        self.vector_store_manager = vector_store_manager
        self.llm = None
        self.qa_chain = None
        self.chat_history = []
        self.semantic_cache = config.semantic_cache  # Use semantic cache from config
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM (Gemini)."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.2,
            google_api_key=self.config.gemini_api_key,
            streaming=True
        )

    def _get_cached_response(self, question, context_summary=""):
        """Retrieve cached response using semantic similarity.

        Args:
            question: User's question
            context_summary: Summary of retrieved context (optional)

        Returns:
            Cached answer string or None if not found
        """
        return self.semantic_cache.get(question, context_summary)

    def _save_to_cache(self, question, answer, context_summary=""):
        """Save response to semantic cache.

        Args:
            question: User's question
            answer: LLM response
            context_summary: Summary of retrieved context (optional)
        """
        self.semantic_cache.set(question, answer, context_summary)

    def format_docs(self, docs):
        """Format documents with source file information.

        Args:
            docs: List of Document objects

        Returns:
            Formatted string with source information
        """
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            source_name = source.split("\\")[-1].split("/")[-1]  # Get filename only
            formatted.append(f"[From: {source_name}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def detect_file_mention(self, question, available_files):
        """Detect if user is asking about a specific file or file type.

        Args:
            question: User's question
            available_files: List of available filenames

        Returns:
            List of mentioned filenames
        """
        question_lower = question.lower()

        # File type mentions with more variations
        file_type_map = {
            'audio': ['.mp3', '.wav', '.m4a', '.flac', '.ogg'],
            'mp3': ['.mp3'],
            'wav': ['.wav'],
            'pdf': ['.pdf'],
            'document': ['.docx'],
            'text': ['.txt'],
            'word': ['.docx'],
            'docx': ['.docx']
        }

        mentioned_files = []

        # Check for specific filename mentions
        for filename in available_files:
            filename_lower = filename.lower()
            # Check if filename (without extension) is mentioned
            name_without_ext = os.path.splitext(filename_lower)[0]
            if name_without_ext in question_lower or filename_lower in question_lower:
                mentioned_files.append(filename)

        # Check for file type mentions (e.g., "the audio", "the PDF", "mp3 file")
        if not mentioned_files:
            for file_type, extensions in file_type_map.items():
                # Check for various patterns like "audio file", "the mp3", "mp3 file", etc.
                if file_type in question_lower or f"{file_type} file" in question_lower:
                    for filename in available_files:
                        if any(filename.lower().endswith(ext) for ext in extensions):
                            mentioned_files.append(filename)

        return mentioned_files

    def retrieve_from_vectorstores(self, question, k=8):
        """Retrieve documents using separate pgvector vectorstores per file.

        Args:
            question: User's question
            k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        vectorstores = self.vector_store_manager.vectorstores
        available_files = self.vector_store_manager.get_available_files()
        mentioned_files = self.detect_file_mention(question, available_files)

        if mentioned_files:
            # Query ONLY the mentioned file's vectorstore
            all_docs = []
            for filename in mentioned_files:
                if filename in vectorstores:
                    retriever = vectorstores[filename].as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": k}
                    )
                    docs = retriever.invoke(question)
                    all_docs.extend(docs)

            # Return documents from mentioned files only
            if all_docs:
                return all_docs[:k]

        # No specific file mentioned: search across all vectorstores and aggregate results
        all_docs = []
        docs_per_file = max(1, k // len(vectorstores))  # Distribute k across files

        for filename, vectorstore in vectorstores.items():
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": docs_per_file}
            )
            docs = retriever.invoke(question)
            all_docs.extend(docs)

        return all_docs[:k]

    def create_qa_chain(self):
        """Create QA chain using pgvector vectorstores.

        Returns:
            LangChain QA chain
        """
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

        self.qa_chain = (
            {
                "context": lambda x: self.format_docs(
                    self.retrieve_from_vectorstores(x["question"], k=8)
                ),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return self.qa_chain

    def ask(self, question):
        """Ask a question and get a full LLM response (no streaming)."""

        # Ensure QA chain is ready
        if self.qa_chain is None:
            self.create_qa_chain()

        # Retrieve context from vectorstores
        retrieved_docs = self.retrieve_from_vectorstores(question, k=8)
        context_summary = self._get_context_summary(retrieved_docs)

        # Try getting a cached response
        cached_response = self._get_cached_response(question, context_summary)
        if cached_response:
            return cached_response

        # Cache miss → Call the LLM
        start_time = time.time()

        # Invoke the LLM synchronously (no streaming)
        response = self.qa_chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })

        # Save response to cache for future reuse
        self._save_to_cache(question, response, context_summary)

        # Print timing info
        duration = time.time() - start_time
        print(f"{Fore.YELLOW}[DEBUG] LLM response time: {duration:.3f}s{Style.RESET_ALL}")

        return response

    def _get_context_summary(self, docs):
        """Generate a summary of context for cache key.

        Args:
            docs: List of retrieved documents

        Returns:
            String summary of document sources
        """
        if not docs:
            return "no_context"
        # Create a consistent summary based on document sources
        sources = sorted(set(doc.metadata.get("source", "unknown") for doc in docs))
        return "|".join(sources[:3])  # Use top 3 sources for cache key

    def update_history(self, question, answer):
        """Update chat history with the latest Q&A pair.

        Args:
            question: User's question
            answer: Assistant's answer
        """
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        # Keep last 5 Q&A pairs (10 messages)
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []

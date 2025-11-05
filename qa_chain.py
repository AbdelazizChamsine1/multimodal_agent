from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from colorama import Fore, Style
from sentence_transformers import CrossEncoder
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
        self.reranker = None  # Will be initialized lazily
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM (Gemini)."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.2,
            google_api_key=self.config.gemini_api_key,
            streaming=True
        )

    def _initialize_reranker(self):
        """Initialize the reranker model lazily (only when first needed)."""
        if self.reranker is None:
            print(f"{Fore.YELLOW}[INFO] Loading reranker model: {self.config.reranker_model}...{Style.RESET_ALL}")
            self.reranker = CrossEncoder(self.config.reranker_model)
            print(f"{Fore.GREEN}[INFO] Reranker loaded successfully!{Style.RESET_ALL}")

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

    def _rerank_documents(self, question, docs, top_k):
        """Rerank documents using CrossEncoder for better relevance.

        Args:
            question: User's question
            docs: List of Document objects
            top_k: Number of documents to return after reranking

        Returns:
            Reranked list of top_k documents
        """
        if len(docs) <= top_k:
            return docs

        # Ensure reranker is loaded
        self._initialize_reranker()

        # Prepare question-document pairs for reranking
        pairs = [[question, doc.page_content] for doc in docs]

        # Score all pairs with the reranker
        scores = self.reranker.predict(pairs)

        # Sort documents by reranking scores (descending)
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k reranked documents
        reranked_docs = [doc for doc, score in doc_scores[:top_k]]

        # Print reranking info for debugging
        print(f"{Fore.CYAN}[RERANK] {len(docs)} candidates → {top_k} docs (top score: {doc_scores[0][1]:.3f}){Style.RESET_ALL}")

        return reranked_docs

    def retrieve_from_vectorstores(self, question, k=None):
        """Retrieve documents using separate pgvector vectorstores per file with reranking.

        Args:
            question: User's question
            k: Number of final documents to return (uses config.final_k if not provided)

        Returns:
            List of retrieved and reranked documents
        """
        if k is None:
            k = self.config.final_k

        retrieve_k = self.config.retrieve_k  # Number of candidates to retrieve before reranking

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
                        search_kwargs={"k": retrieve_k}  # Retrieve more candidates
                    )
                    docs = retriever.invoke(question)
                    all_docs.extend(docs)

            # Rerank and return top k documents
            if all_docs:
                return self._rerank_documents(question, all_docs, k)

        # No specific file mentioned: search across all vectorstores and aggregate results
        all_docs = []
        docs_per_file = max(2, retrieve_k // len(vectorstores))  # Distribute retrieve_k across files

        for filename, vectorstore in vectorstores.items():
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": docs_per_file}
            )
            docs = retriever.invoke(question)
            all_docs.extend(docs)

        # Rerank all candidates and return top k
        return self._rerank_documents(question, all_docs, k)

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
                    self.retrieve_from_vectorstores(x["question"])
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
        """Ask a question and stream the LLM response."""

        # Ensure QA chain is ready
        if self.qa_chain is None:
            self.create_qa_chain()

        # Check cache first
        # This avoids expensive retrieval and reranking on cache hits
        cached_response = self._get_cached_response(question, context_summary="")
        if cached_response:
            print(f"{Fore.CYAN}[CACHE HIT] Response retrieved from cache (no retrieval/reranking needed){Style.RESET_ALL}\n")
            yield cached_response
            return

        # Cache miss → Call the LLM with streaming (chain will handle retrieval+reranking)
        print(f"{Fore.YELLOW}[CACHE MISS] Generating new response...{Style.RESET_ALL}\n")
        start_time = time.time()

        # Stream the LLM response
        full_response = ""
        for chunk in self.qa_chain.stream({
            "question": question,
            "chat_history": self.chat_history
        }):
            full_response += chunk
            yield chunk

        # Save response to cache for future reuse (question-only caching)
        self._save_to_cache(question, full_response, context_summary="")

        # Print timing info
        duration = time.time() - start_time
        print(f"\n{Fore.YELLOW}[DEBUG] LLM response time: {duration:.3f}s{Style.RESET_ALL}")

        return

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

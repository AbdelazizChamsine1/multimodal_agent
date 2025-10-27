from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(vectorstore, api_key):
    """Create the QA chain using Gemini and a FAISS vectorstore."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.2,
        google_api_key=api_key
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant answering questions about a document or audio transcript.
        Use the following context from the document/transcript and the conversation history to answer questions.
        If you don't know the answer based on the context, just say that you don't know.
        
        Context from document: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    qa_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain

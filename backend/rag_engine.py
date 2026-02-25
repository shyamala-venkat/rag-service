from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from backend.config import ANTHROPIC_API_KEY, PINECONE_API_KEY, PINECONE_INDEX

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,  # matches all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector store
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings
)

# LLM
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    api_key=ANTHROPIC_API_KEY
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context: {context}

Chat History: {chat_history}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def index_documents(chunks):
    """Add document chunks to Pinecone"""
    vectorstore.add_documents(chunks)
    return len(chunks)

def ask(question: str, chat_history: str = ""):
    """Run a RAG query and return the answer"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: chat_history
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain.invoke(question)
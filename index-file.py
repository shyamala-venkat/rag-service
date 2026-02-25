import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# 1. Load your document
loader = PyPDFLoader("Shyamala_Venkatakrishnan_Resume.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# 3. Embed using free HuggingFace model
print("Loading embedding model... (first run may take a minute)")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store in ChromaDB
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)
print("Documents indexed into ChromaDB!")

# 5. Set up Claude as the LLM
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# 6. Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know the answer from the context, say "I don't know".

Context: {context}

Question: {question}
""")

# 7. Build the chain using newer LCEL syntax
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# 8. Ask questions in a loop
print("\nRAG system ready! Type 'quit' to exit.\n")
while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break

    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}\n")
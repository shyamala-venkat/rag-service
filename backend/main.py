from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.rag_engine import ask, index_documents
from backend.document_loader import load_pdf, load_word, load_url, split_documents
from pydantic import BaseModel

app = FastAPI(title="RAG Portfolio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    chat_history: str = ""

@app.get("/")
def root():
    return {"status": "RAG API is running"}

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    docs = load_pdf(file.file)
    chunks = split_documents(docs)
    count = index_documents(chunks)
    return {"message": f"Indexed {count} chunks from {file.filename}"}

@app.post("/upload/word")
async def upload_word(file: UploadFile = File(...)):
    docs = load_word(file.file)
    chunks = split_documents(docs)
    count = index_documents(chunks)
    return {"message": f"Indexed {count} chunks from {file.filename}"}

@app.post("/upload/url")
async def upload_url(url: str = Form(...)):
    docs = load_url(url)
    chunks = split_documents(docs)
    count = index_documents(chunks)
    return {"message": f"Indexed {count} chunks from {url}"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = ask(request.question, request.chat_history)
    return {"answer": answer}
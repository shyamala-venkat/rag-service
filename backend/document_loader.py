from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def load_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file.read())
        tmp_path = f.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)
    return docs

def load_word(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
        f.write(file.read())
        tmp_path = f.name
    loader = Docx2txtLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)
    return docs

def load_url(url: str):
    loader = WebBaseLoader(url)
    return loader.load()

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)
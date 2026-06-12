import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def load_docs(path):
    loader = PyPDFDirectoryLoader(path)
    return loader.load()

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def embed_docs(chunks, db_path):
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    Chroma.from_documents(documents=chunks, embedding=emb, persist_directory=db_path)

def main():
    docs_path = "data/docs"
    db_path = "data/chroma"
    
    docs = load_docs(docs_path)
    chunks = chunk_docs(docs)
    embed_docs(chunks, db_path)

if __name__ == "__main__":
    main()

import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
DATA_PATH = "./data"
DB_PATH = "./vector_db"

def create_vector_db():
    print(" Initializing Knowledge Base Builder...")
    documents = []

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        print(f" Processing PDF: {pdf_file}...")
        loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
        pdf_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents.extend(text_splitter.split_documents(pdf_docs))

    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    for csv_file in csv_files:
        print(f" Processing CSV: {csv_file}...")
        loader = CSVLoader(file_path=os.path.join(DATA_PATH, csv_file), encoding='utf-8') 
        documents.extend(loader.load())

    print(f" Total Information Chunks Created: {len(documents)}")

    print("embedding data... (This may take 1-2 minutes)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print(f" Database Successfully Created at {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()
# ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DB_DIR = "db"
DOCUMENTS_DIR = "documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    """
    Main function to load, split, and embed documents, then store them in ChromaDB.
    """
    # 1. Load documents from the specified directory
    print("Loading documents...")
    loader = DirectoryLoader(
        DOCUMENTS_DIR,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    if not documents:
        print("No PDF documents found. Please add PDFs to the 'documents' folder.")
        return

    # 2. Split documents into smaller chunks
    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks.")

    # 3. Create embeddings
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
    )

    # 4. Create and persist the vector store
    print("Creating and persisting vector store...")
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=DB_DIR
    )
    print(f"Vector store created successfully! Indexed {len(texts)} chunks from {len(documents)} documents.")

if __name__ == "__main__":
    main()
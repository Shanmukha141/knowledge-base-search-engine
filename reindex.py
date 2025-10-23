"""
reindex.py - Rebuild the vector database from all documents in the documents folder.
Run this script whenever you add or remove PDFs to update the search index.
"""

import os
import glob
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURATION ---
DB_DIR = "db"
DOCUMENTS_DIR = "documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    """Reindex all documents in the documents folder."""
    
    print("=" * 60)
    print("REINDEXING KNOWLEDGE BASE")
    print("=" * 60)
    
    # Delete the old database
    print("\n[1/5] Removing old database...")
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR, ignore_errors=True)
        print("  ✓ Old database removed")
    else:
        print("  ✓ No existing database found")
    
    # Initialize embeddings
    print("\n[2/5] Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print(f"  ✓ Using model: {EMBEDDING_MODEL_NAME}")
    
    # Load all documents
    print(f"\n[3/5] Loading documents from '{DOCUMENTS_DIR}'...")
    docs = []
    
    # Load PDFs
    try:
        from pypdf import PdfReader
        pdf_files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
        print(f"  Found {len(pdf_files)} PDF files:")
        
        for path in pdf_files:
            try:
                filename = os.path.basename(path)
                print(f"    • Processing: {filename}")
                reader = PdfReader(path)
                content = "\n".join((page.extract_text() or "") for page in reader.pages)
                
                if content.strip():
                    docs.append(Document(
                        page_content=content, 
                        metadata={"source": path, "filename": filename}
                    ))
                    print(f"      ✓ Loaded {len(reader.pages)} pages")
                else:
                    print(f"      ⚠ No text content found")
            except Exception as e:
                print(f"      ✗ Error: {str(e)}")
    except ImportError:
        print("  ⚠ pypdf not installed. Install it with: pip install pypdf")
    
    # Load text files
    for pattern in ("*.txt", "*.md"):
        text_files = glob.glob(os.path.join(DOCUMENTS_DIR, pattern))
        for path in text_files:
            try:
                filename = os.path.basename(path)
                print(f"    • Processing: {filename}")
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if text.strip():
                    docs.append(Document(
                        page_content=text, 
                        metadata={"source": path, "filename": filename}
                    ))
                    print(f"      ✓ Loaded text file")
            except Exception as e:
                print(f"      ✗ Error: {str(e)}")
    
    if not docs:
        print("\n  ✗ No documents found! Add PDF, TXT, or MD files to the 'documents' folder.")
        return
    
    print(f"\n  ✓ Total documents loaded: {len(docs)}")
    
    # Split into chunks
    print("\n[4/5] Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)
    print(f"  ✓ Created {len(chunks)} chunks from {len(docs)} documents")
    
    # Create vector database
    print("\n[5/5] Creating vector database and indexing chunks...")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Add documents with unique IDs
    ids = [f"{d.metadata.get('filename', 'unknown')}::chunk::{i}" for i, d in enumerate(chunks)]
    db.add_documents(chunks, ids=ids)
    print("  ✓ Chunks indexed successfully")
    
    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    result = db.get()
    
    from collections import Counter
    source_counts = Counter()
    for metadata in result['metadatas']:
        if metadata and 'filename' in metadata:
            source_counts[metadata['filename']] += 1
    
    print(f"\nTotal chunks in database: {len(result['ids'])}")
    print(f"\nChunks per file:")
    for filename, count in sorted(source_counts.items()):
        print(f"  • {filename}: {count} chunks")
    
    print("\n" + "=" * 60)
    print("✓ REINDEXING COMPLETE!")
    print("=" * 60)
    print("\nYou can now restart your Flask app to use the updated index.")

if __name__ == "__main__":
    main()

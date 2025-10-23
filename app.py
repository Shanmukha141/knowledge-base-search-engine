# app.py - FINAL WORKING VERSION

import os
import traceback
import concurrent.futures
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import glob
import shutil  # <-- add
from langchain.text_splitter import RecursiveCharacterTextSplitter  # <-- add
from langchain_core.documents import Document  # <-- add

# Vector store and models
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Flask app
app = Flask(__name__)
load_dotenv()
os.environ["HF_HUB_HTTP_TIMEOUT"] = "120"

# --- CONFIGURATION ---
DB_DIR = "db"
DOCUMENTS_DIR = "documents"
REQUEST_TIMEOUT_SECONDS = 60  # <-- allow more time for first cold start
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = DOCUMENTS_DIR

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Use a smaller model to reduce cold start
LLM_REPO_ID = "google/flan-t5-small"  # <-- changed from base

# --- GLOBALS ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ---------- NEW: document ingestion ----------
def load_local_documents():
    docs = []
    # Load .txt and .md
    for pattern in ("*.txt", "*.md"):
        for path in glob.glob(os.path.join(DOCUMENTS_DIR, pattern)):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": path}))
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")

    # Optional: load .pdf if pypdf is available
    try:
        from pypdf import PdfReader  # pip install pypdf
        for path in glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf")):
            try:
                reader = PdfReader(path)
                content = "\n".join((page.extract_text() or "") for page in reader.pages)
                if content.strip():
                    docs.append(Document(page_content=content, metadata={"source": path}))
            except Exception as e:
                print(f"[WARN] Skipping PDF {path}: {e}")
    except Exception as e:
        print(f"[INFO] PDF support not available (install pypdf to enable): {e}")
    return docs

def ensure_index_built():
    has_vectors = False
    try:
        existing = db.get(limit=1)
        has_vectors = bool(existing and existing.get("ids"))
    except Exception:
        has_vectors = False

    if has_vectors:
        print("[INFO] Existing Chroma index detected.")
        return

    raw_docs = load_local_documents()
    if not raw_docs:
        print(f"[WARN] No documents found in '{DOCUMENTS_DIR}'. Place files there and reindex.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(raw_docs)
    ids = [f"{os.path.basename(d.metadata.get('source','doc'))}::chunk::{i}" for i, d in enumerate(chunks)]
    db.add_documents(chunks, ids=ids)
    print(f"[INFO] Indexed {len(chunks)} chunks from {len(raw_docs)} files.")
# ---------- end ingestion ----------

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HF_TOKEN:
    os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)

# Build index on startup (if empty)
ensure_index_built()

def init_llm():
    # Try remote HF Inference API first (no streaming)
    remote = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="text2text-generation",
        max_new_tokens=128,
        temperature=0.0,
        streaming=False,
        return_full_text=False,
    )
    # Quick ping with timeout to detect StopIteration/endpoint issues
    def _ping():
        return remote.invoke("ok")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_ping)
        try:
            _ = fut.result(timeout=20)
            return remote
        except Exception as e:
            print(f"[WARN] HF endpoint not usable, falling back to local pipeline: {e}")

    # Fallback: local Transformers pipeline (downloads flan-t5-small once)
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        tok = AutoTokenizer.from_pretrained(LLM_REPO_ID)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(LLM_REPO_ID)
        pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok)
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        raise RuntimeError(f"Failed to init both HF endpoint and local pipeline: {e}")

llm = init_llm()

# Prompt and chain unchanged
prompt_template = """
Using these documents, answer the user's question succinctly.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

def _invoke_qa(question: str):
    # RetrievalQA expects input key "query"
    return qa_chain.invoke({"query": question})

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return jsonify({
                "answer": "No indexed documents were found. Please upload or ingest documents and try again.",
                "sources": []
            }), 200

        print("Invoking QA chain...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_invoke_qa, query)
            try:
                result = future.result(timeout=REQUEST_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                return jsonify({
                    "error": f"Timed out after {REQUEST_TIMEOUT_SECONDS}s. Model may be cold-starting; try again."
                }), 504

        print("QA chain returned.")
        answer = result.get("result", "Sorry, I couldn't find an answer.")
        sources = []
        if result.get("source_documents"):
            sources = list({doc.metadata.get("source", "Unknown source") for doc in result["source_documents"]})

        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        print("--- ERROR OCCURRED ---")
        traceback.print_exc()
        print("----------------------")
        return jsonify({"error": f"{type(e).__name__}: {repr(e)}"}), 500

@app.route("/test_llm")
def test_llm():
    try:
        prompt = "Answer with exactly one word: ok"
        out = llm.invoke(prompt)
        return jsonify({"ok": True, "prompt": prompt, "output": str(out)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {repr(e)}"}), 500

@app.route("/reindex", methods=["POST"])
def reindex():
    """Rebuild the vector index from files in the documents folder."""
    try:
        shutil.rmtree(DB_DIR, ignore_errors=True)
        # Recreate vectorstore and retriever
        global db, retriever
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        ensure_index_built()
        return jsonify({"ok": True, "message": "Reindexed from documents/."}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5001, use_reloader=False)

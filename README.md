

# 🧠 Knowledge-Base Search Engine with RAG

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)
[](https://flask.palletsprojects.com/)

An intelligent search engine that ingests multiple PDF documents and uses a **Retrieval-Augmented Generation (RAG)** pipeline to provide synthesized, context-aware answers to user questions. Instead of just finding keywords, this application understands the content of your documents and generates human-like responses.

This project allows you to create your own private, powerful search engine for any collection of documents.

-----

## 📸 Demo
to watch the working video click on the link: https://drive.google.com/file/d/1EheR9FxRXGUPqwskBjheJ3fEN1Tv1_sw/view?usp=drivesdk
<img width="1918" height="1021" alt="image" src="https://github.com/user-attachments/assets/7f854741-f105-41d9-92c8-44d40ebc14c7" />
<img width="1918" height="1013" alt="image" src="https://github.com/user-attachments/assets/84f321a4-bbdb-4bbc-bdf4-669a92e6a948" />
<img width="1918" height="1012" alt="image" src="https://github.com/user-attachments/assets/5f74a259-1176-408e-92b0-f37143ecfdff" />
<img width="1918" height="1015" alt="image" src="https://github.com/user-attachments/assets/7b1388a1-6572-4c22-9feb-09c4d54e921c" />
<img width="1918" height="1015" alt="image" src="https://github.com/user-attachments/assets/17fdcf5d-c16d-4c3e-be1b-bc41c63f089b" />

-----

## ✨ Core Features

  * **Document Ingestion**: Seamlessly process multiple PDF files at once. The system automatically extracts text, splits it into manageable chunks, and prepares it for searching.
  * **Vector Embeddings**: Uses state-of-the-art sentence-transformer models to convert text chunks into numerical vectors, capturing the semantic meaning of the content.
  * **Efficient Vector Storage**: Leverages **ChromaDB** to store and efficiently search through thousands of text embeddings, ensuring fast retrieval of relevant information.
  * **Retrieval-Augmented Generation (RAG)**: Implements a powerful RAG pipeline that:
    1.  **Retrieves** the most relevant document chunks for a given question.
    2.  **Augments** a Large Language Model (LLM) with this specific context.
    3.  **Generates** a coherent, accurate answer based *only* on the provided documents.
  * **Interactive Web Interface**: A clean and simple UI built with **Flask** and Jinja2 allows users to easily submit questions and view the generated answers and their sources.

-----

## ⚙️ How It Works: The RAG Pipeline

The application follows a two-stage process:

### 1\. Ingestion (`ingest.py`)

This is a one-time process you run whenever you add or update your documents.

1.  **Load**: All PDF files from the `/documents` folder are loaded.
2.  **Split**: The text from each document is split into smaller, overlapping chunks. This helps maintain context without overwhelming the language model.
3.  **Embed**: Each chunk is passed through a Hugging Face embedding model (`all-MiniLM-L6-v2`) to create a vector.
4.  **Store**: These vectors and their corresponding text are stored in a local ChromaDB vector database in the `/db` folder.

### 2\. Question-Answering (`app.py`)

This process runs in real-time whenever a user asks a question.

1.  **User Query**: A user submits a question through the web interface.
2.  **Embed Query**: The user's question is converted into a vector using the same embedding model.
3.  **Search & Retrieve**: The application searches the ChromaDB database to find the text chunks whose vectors are most similar to the question's vector.
4.  **Augment & Generate**: The retrieved text chunks (the context) and the user's question are inserted into a prompt. This prompt is then sent to a Large Language Model (e.g., `google/flan-t5-base`) which generates a final answer based on the provided context.

-----

## 🛠️ Tech Stack

  * **Backend Framework**: Flask
  * **LLM Orchestration**: LangChain
  * **LLM & Embeddings**: Hugging Face Hub (Transformers & Sentence-Transformers)
  * **Vector Database**: ChromaDB
  * **PDF Processing**: PyPDF
  * **Frontend**: HTML & Jinja2

-----

## 📂 Project Structure

```
knowledge-base-search-engine/
├── app.py             # Main Flask application with API endpoints
├── ingest.py          # Script to process and embed documents
├── requirements.txt   # List of all Python dependencies
├── README.md          # You are here!
├── documents/         # --- Place your PDF files in this folder ---
└── templates/
    └── index.html     # The HTML file for the web interface
```

-----

## 🚀 Getting Started: Setup and Installation

Follow these steps to get the project running on your local machine.

### 1\. Prerequisites

  * Python 3.9 or higher
  * Git for cloning the repository

### 2\. Clone the Repository

```bash
git clone <your-repository-link>
cd knowledge-base-search-engine
```

### 3\. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
```

### 4\. Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5\. Get Your Hugging Face API Token

The application uses a model from the Hugging Face Hub, which requires a free API token.

1.  **Create an account** on [huggingface.co](https://huggingface.co).
2.  Navigate to **Settings -\> Access Tokens**.
3.  **Generate a new token** with the **`read`** role.
4.  **Important:** The provided `app.py` code requires you to hardcode this token. Open the `app.py` file and paste your token where indicated.

-----

## ▶️ How to Run the Application

### Step 1: Add Your Documents

Place all the PDF files you want to search through into the `/documents` folder. You can start by saving a web page (like a Wikipedia article) as a PDF.

### Step 2: Run the Ingestion Script

This step "teaches" the search engine about your documents. Run this script once, or whenever you add/change the files in the `/documents` folder.

```bash
python ingest.py
```

This will create a `/db` folder containing the vector database.

### Step 3: Run the Web Application

Start the Flask server.

```bash
python app.py
```

### Step 4: Open Your Search Engine\!

Open your web browser and navigate to the URL shown in the terminal, which is usually:

**`http://127.0.0.1:5001`**

You can now ask questions about the content of your PDFs\!

-----

## 🔧 Customization

You can easily customize the application by modifying `app.py`:

  * **Change the LLM**: To use a different model, simply change the `LLM_REPO_ID` variable to another model from the Hugging Face Hub (e.g., `mistralai/Mistral-7B-Instruct-v0.2`).
  * **Customize the Prompt**: Modify the `prompt_template` string to change how the AI is instructed to answer questions. You can alter its tone, format, or constraints.
  * **Adjust Retrieval**: Change the `k` value in the `retriever` definition to fetch more or fewer document chunks for context.

-----

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Feel free to check the issues page if you want to contribute.

-----

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.



# Knowledge-base Search Engine using RAG

This project implements a search engine that ingests multiple PDF documents and uses a Retrieval-Augmented Generation (RAG) pipeline to answer user questions based on the document content.

## Objective

Search across documents and provide synthesized answers using an LLM-based RAG system.

## Features

- **Document Ingestion**: Process and embed text from multiple PDF documents.
- **Vector Storage**: Uses ChromaDB to store text embeddings for efficient retrieval.
- **Retrieval-Augmented Generation (RAG)**: Retrieves relevant document chunks and uses a Large Language Model (LLM) to synthesize a coherent answer.
- **Web Interface**: A simple frontend built with Flask to submit queries and view answers.

## Project Structure
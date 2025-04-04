# Local RAG Agent with DeepSeek & Ollama

## Overview

This project implements a Retrieval-Augmented Generation (RAG) agent that allows users to query their local text documents using a powerful Large Language Model (DeepSeek) run locally via Ollama. It leverages LangChain for orchestration, FAISS for efficient vector storage and retrieval, and Ollama for local LLM and embedding model hosting.

The primary goal is to provide accurate, context-aware answers based on a custom knowledge base, demonstrating a practical application of RAG architecture.

## Features

* **Retrieval-Augmented Generation (RAG):** Implements the core RAG pipeline to enhance LLM responses with information retrieved from local documents.
* **Local LLM Integration:** Utilizes Ollama to run the `deepseek-llm` model (or other compatible models) locally, ensuring privacy and control.
* **Local Embeddings:** Employs Ollama to run the `nomic-embed-text` model for generating text embeddings locally.
* **Vector Store:** Uses FAISS (CPU) for creating, saving, and loading a persistent vector index of the document knowledge base, enabling efficient similarity searches.
* **Knowledge Base:** Ingests and processes local `.pdf` files from a specified data directory.
* **Modular Design:** Built with a clear, modular Python structure (`rag_core` package) separating concerns like configuration, data loading, vector store management, LLM interaction, and RAG chain logic.
* **Command-Line Interface (CLI):** Provides an interactive CLI (`main_cli.py`) for users to ask questions and receive answers directly in the terminal.
* **Web UI Interface:** Provides a Streamlit UI(`app.py`) to chat with the model in a visually pleasing way
* **Streaming Responses:** LLM responses are streamed token-by-token to the CLI or Web UI for a better user experience.
* **Fallback Mechanism:** If no relevant context is found in the documents, the agent can fall back to using the base LLM's knowledge.
* **Ollama Integration Checks:** Includes utility functions to verify Ollama service status and model availability before starting.

## Technology Stack

* **Language:** Python 3.x
* **Core Framework:** LangChain
* **LLM Hosting:** Ollama
* **LLM Model:** DeepSeek (`deepseek-llm:latest` or similar)
* **Embedding Model:** Nomic Embed Text (`nomic-embed-text:latest` via Ollama)
* **Vector Database:** FAISS (`faiss-cpu`)
* **Core Libraries:** `langchain`, `langchain-community`, `ollama`, `faiss-cpu`

## How It Works (High-Level RAG Flow)

1.  **Load:** Pdf documents (`.pdf`) are loaded from the `rag_data` directory.
2.  **Split:** Documents are split into smaller, manageable chunks.
3.  **Embed:** Text chunks are converted into numerical vector embeddings using the local embedding model via Ollama.
4.  **Store:** Embeddings are stored in a FAISS vector index, which is persisted locally in the `vectorstore_db` directory.
5.  **Retrieve:** When a user asks a question, the query is embedded, and FAISS retrieves the most relevant document chunks based on vector similarity.
6.  **Augment & Generate:** The retrieved chunks (context) and the original question are formatted into a prompt and sent to the local DeepSeek LLM (via Ollama) to generate the final answer.

## Key Concepts Demonstrated

* Retrieval-Augmented Generation (RAG) Architecture
* Large Language Model (LLM) Integration (Ollama, DeepSeek)
* Vector Embeddings and Similarity Search
* Vector Database Management (FAISS)
* Local Model Inference (Privacy-preserving AI)
* Modular Python Project Structure
* Command-Line Interface Development
* Using LangChain for AI Application Orchestration
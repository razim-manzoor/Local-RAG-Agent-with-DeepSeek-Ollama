import os

# --- Core Configuration ---

# Project Root Directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

# Data and Vector Store Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "rag_data")
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore_db")

# Ollama Model Names
MODEL_NAME = "deepseek-llm:latest"
EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"

# RAG Parameters
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 150 
SEARCH_K = 4 

# Prompt Template
RAG_PROMPT_TEMPLATE = """
**Context:**
{context}

**Based ONLY on the context provided above, answer the following question:**
Question: {question}

**Answer:**
"""

# Simple LLM Fallback Prompt (if no RAG context)
SIMPLE_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
import logging
import ollama
from typing import List
from langchain.schema import Document # Use langchain.schema for Document type hint

logger = logging.getLogger(__name__)

def check_ollama_model_availability(model_name: str) -> bool:
    """Checks if the specified Ollama model is available."""
    try:
        logger.info(f"Checking for Ollama model: {model_name}")
        ollama.show(model_name)
        logger.info(f"Model '{model_name}' found.")
        return True
    except Exception as e:
        logger.error(f"Model '{model_name}' not found or Ollama not running: {e}")
        logger.error(f"Please ensure Ollama is running and the model is pulled (`ollama pull {model_name}`).")
        return False

def check_ollama_status() -> bool:
    """Checks if the Ollama service is running."""
    try:
        ollama.list()
        logger.info("Ollama service is running.")
        return True
    except Exception as e:
        logger.error(f"Ollama service not responding: {e}")
        logger.error("Please ensure the Ollama application or service is running.")
        return False

def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single string for the prompt context."""
    if not docs:
        return "No context documents were found or retrieved."
    # Ensure metadata is accessed safely
    formatted_docs = []
    for i, doc in enumerate(docs):
        metadata_str = str(doc.metadata) if doc.metadata else "{}"
        page_content_str = doc.page_content if doc.page_content else ""
        formatted_docs.append(f"Source {i+1} (metadata: {metadata_str}):\n{page_content_str}")
    return "\n\n".join(formatted_docs)
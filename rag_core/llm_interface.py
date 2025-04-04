import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Import config variables
from .config import MODEL_NAME, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

def get_ollama_embeddings() -> OllamaEmbeddings | None:
    """Initializes and returns the Ollama embeddings model."""
    logger.info(f"Initializing Ollama embeddings with model: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        # Test connection during initialization
        _ = embeddings.embed_query("Test query for embedding model connection.")
        logger.info("Ollama embeddings initialized successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize Ollama embeddings for model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
        logger.error("Ensure the embedding model is pulled and Ollama is running.")
        return None

def get_ollama_llm(streaming: bool = False) -> Ollama | None:
    """Initializes and returns the Ollama LLM."""
    logger.info(f"Initializing Ollama LLM with model: {MODEL_NAME}")
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
    try:
        llm = Ollama(
            model=MODEL_NAME,
            callback_manager=CallbackManager(callbacks) if callbacks else None,
            # Add other parameters like temperature, num_ctx if needed
        )
        logger.info(f"Ollama LLM '{MODEL_NAME}' initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM for model '{MODEL_NAME}': {e}", exc_info=True)
        logger.error("Ensure the LLM model is pulled and Ollama is running.")
        return None
import sys
import time
import logging

# Setup basic logging for the CLI application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary functions from our rag_core package
from rag_core.config import SEARCH_K
from rag_core.utils import check_ollama_status, check_ollama_model_availability
from rag_core.data_loader import load_documents, split_documents
from rag_core.llm_interface import get_ollama_embeddings, get_ollama_llm
from rag_core.vector_store import create_or_load_vectorstore
from rag_core.rag_chain import setup_rag_chain
from rag_core.config import MODEL_NAME, EMBEDDING_MODEL_NAME

def run_cli():
    """Runs the command-line interface for the RAG agent."""
    logger.info("Starting Command-Line RAG Agent...")

    # --- Initial Checks ---
    if not check_ollama_status():
        sys.exit(1)
    if not check_ollama_model_availability(MODEL_NAME):
        sys.exit(1)
    if not check_ollama_model_availability(EMBEDDING_MODEL_NAME):
        # Allow continuing without embedding model, vector store creation will fail gracefully
        logger.warning(f"Embedding model '{EMBEDDING_MODEL_NAME}' check failed. RAG features might be unavailable.")

    # --- Load Embeddings and LLM ---
    embeddings = get_ollama_embeddings()
    # For CLI, enable streaming output from the LLM
    llm = get_ollama_llm(streaming=True)

    if not llm:
        logger.error("Failed to initialize the main LLM. Exiting.")
        sys.exit(1)

    # --- Load, Split, and Create/Load Vector Store ---
    retriever = None
    if embeddings: 
        documents = load_documents()
        text_chunks = split_documents(documents)
        vectorstore = create_or_load_vectorstore(text_chunks, embeddings)

        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
            logger.info(f"Retriever setup with k={SEARCH_K}")
        else:
            logger.warning("Failed to setup vector store/retriever. RAG will be disabled.")
    else:
        logger.warning("Embeddings failed to load. RAG features disabled.")

    # --- Setup RAG Chain (will adapt based on retriever availability) ---
    rag_chain = setup_rag_chain(llm, retriever)

    # --- Interactive Query Loop ---
    logger.info("RAG Agent Ready. Enter 'quit', 'exit', or 'bye' to stop.")
    while True:
        try:
            query = input("\nAsk your question: ")
            if query.lower() in ["quit", "exit", "bye"]:
                break
            if not query.strip():
                continue

            logger.info(f"Processing query: '{query}'")
            start_time = time.time()

            print("\nAssistant:")
            # Invoke the chain - it expects a string query
            # The streaming callback in get_ollama_llm handles printing to console
            full_response = ""
            for chunk in rag_chain.stream(query):
                 # Streaming is handled by the callback, but we can collect if needed
                 full_response += chunk
                 # We don't print here because the callback does it.
                 pass # Callback handles printing

            # Add a newline after the streaming finishes if needed (callback might handle it)
            print()

            end_time = time.time()
            logger.info(f"Query processed in {end_time - start_time:.2f} seconds.")

            # Optionally log retrieved context (if needed for debugging)
            # if retriever:
            #     try:
            #         retrieved_docs = retriever.invoke(query)
            #         logger.debug(f"Retrieved Context: {[doc.metadata for doc in retrieved_docs]}")
            #     except Exception as e:
            #          logger.debug(f"Could not retrieve context for logging: {e}")


        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}", exc_info=True)
            print("\nSorry, an error occurred. Please try again.")

    logger.info("RAG Agent stopped.")

if __name__ == "__main__":
    run_cli()

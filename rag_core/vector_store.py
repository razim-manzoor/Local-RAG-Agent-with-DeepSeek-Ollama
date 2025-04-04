import os
import time
import logging
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document

# Import config variables
from .config import VECTORSTORE_DIR

logger = logging.getLogger(__name__)

def create_or_load_vectorstore(
    chunks: List[Document],
    embeddings: OllamaEmbeddings,
    store_path: str = VECTORSTORE_DIR
) -> Optional[FAISS]:
    """Creates a FAISS vector store or loads it if it exists."""
    vectorstore = None
    if os.path.exists(store_path):
        logger.info(f"Attempting to load existing vector store from: {store_path}")
        try:
            vectorstore = FAISS.load_local(
                store_path,
                embeddings,
                allow_dangerous_deserialization=True # Needed for custom embeddings
            )
            logger.info("Vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}. Will try to recreate if chunks are provided.")
            vectorstore = None

    # If loading failed or store doesn't exist, try creating a new one
    if not vectorstore:
        if not chunks:
            logger.warning("No document chunks provided and no existing vector store found. Cannot create or load vector store.")
            return None

        logger.info(f"Creating new vector store with {len(chunks)} chunks...")
        try:
            start_time = time.time()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            end_time = time.time()
            logger.info(f"Vector store created in {end_time - start_time:.2f} seconds.")

            logger.info(f"Saving vector store to: {store_path}")
            os.makedirs(store_path, exist_ok=True)
            vectorstore.save_local(store_path)
            logger.info("Vector store saved successfully.")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to create or save vector store at {store_path}: {e}", exc_info=True)
            return None

    return vectorstore
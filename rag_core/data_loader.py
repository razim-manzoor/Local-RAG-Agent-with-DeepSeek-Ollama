import os
import logging
from typing import List
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Import config variables
from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def load_documents(data_directory: str = DATA_DIR) -> List[Document]:
    """Loads documents (.pdf) from the specified directory."""
    logger.info(f"Loading PDF documents from: {data_directory}")
    if not os.path.exists(data_directory) or not os.listdir(data_directory):
        logger.warning(f"Data directory '{data_directory}' is empty or does not exist.")
        return []  # Return empty list if no data

    loader = DirectoryLoader(
        data_directory,
        glob="**/*.pdf",  # Load .pdf files
        loader_cls=PyPDFLoader,
        use_multithreading=True,
        show_progress=False,  # Set to True for verbose loading in CLI
        recursive=True
    )
    try:
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} PDF documents.")
        if not documents:
            logger.warning(f"No PDF documents found in {data_directory}.")
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents from {data_directory}: {e}", exc_info=True)
        return []  # Return empty list on error

def split_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks."""
    if not documents:
        logger.info("No documents to split.")
        return []

    logger.info(f"Splitting {len(documents)} documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    try:
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Failed to split documents: {e}", exc_info=True)
        return []  # Return empty list on error
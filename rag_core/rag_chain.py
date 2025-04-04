import logging
from typing import Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.llms import Ollama
from langchain_community.vectorstores.utils import DistanceStrategy # Used by FAISS retriever
from langchain_community.vectorstores import VectorStore # Type hint for retriever parent class
from langchain.schema.retriever import BaseRetriever # More specific type hint

# Import config and utilities
from .config import RAG_PROMPT_TEMPLATE, SIMPLE_PROMPT_TEMPLATE, SEARCH_K
from .utils import format_docs

logger = logging.getLogger(__name__)

def setup_rag_chain(llm: Ollama, retriever: Optional[BaseRetriever]) -> Any: # Using Any for the complex Runnable type
    """Sets up the RAG chain using LangChain Expression Language (LCEL)."""

    if retriever:
        # RAG Chain implementation
        logger.info("Setting up RAG chain with retriever...")
        prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        # Define the RAG pipeline using LCEL
        rag_chain = (
            # RunnableParallel allows running retriever and passing question simultaneously
            {"context": retriever, "question": RunnablePassthrough()}
            # Assign formatted context to the dictionary
            | RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            # Feed the dictionary with context and question to the prompt
            | prompt
            # Send the formatted prompt to the LLM
            | llm
            # Parse the LLM output string
            | StrOutputParser()
        )
        logger.info("RAG chain setup complete.")
        return rag_chain
    else:
        # Fallback: Simple LLM chain if no retriever is available
        logger.warning("Setting up a simple LLM chain (no retriever available).")
        prompt = PromptTemplate(
            template=SIMPLE_PROMPT_TEMPLATE,
            input_variables=["question"]
        )
        simple_chain = (
            # Expects a simple string question as input
            {"question": RunnablePassthrough()} # Wrap input string in dict for consistency if needed downstream
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Simple LLM chain setup complete.")
        return simple_chain
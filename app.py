import streamlit as st
import logging
import os

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary functions and variables from our rag_core package
from rag_core.config import MODEL_NAME, EMBEDDING_MODEL_NAME, DATA_DIR, VECTORSTORE_DIR, SEARCH_K
from rag_core.utils import check_ollama_model_availability, format_docs 
from rag_core.data_loader import load_documents, split_documents
from rag_core.llm_interface import get_ollama_embeddings, get_ollama_llm
from rag_core.vector_store import create_or_load_vectorstore
from rag_core.rag_chain import setup_rag_chain

# --- File Upload Functionality ---

def save_uploaded_file(uploaded_file):
    """
    Saves an uploaded PDF file to the DATA_DIR folder.
    
    This function ensures that the DATA_DIR exists and writes the file in binary mode.
    """
    if uploaded_file is not None:
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file: {file_path}")
        return file_path
    return None

# --- Streamlit Caching for Expensive Resources ---

@st.cache_resource
def load_pipeline():
    """Loads embeddings, LLM, vector store, retriever and sets up the RAG chain."""
    st.info("Initializing RAG pipeline... (Cached)")

    # 1. Check Models (Basic check, detailed check in utils)
    if not check_ollama_model_availability(MODEL_NAME) or not check_ollama_model_availability(EMBEDDING_MODEL_NAME):
        st.error("One or more Ollama models not found. Please pull them.", icon="🚨")
        st.stop()

    # 2. Load Embeddings & LLM
    embeddings = get_ollama_embeddings()
    llm = get_ollama_llm(streaming=False)  # Streamlit handles streaming via write_stream

    if not embeddings or not llm:
        st.error("Failed to initialize Ollama models. Check Ollama status and logs.", icon="🚨")
        st.stop()

    # 3. Load/Process Documents and Vector Store
    retriever = None
    with st.spinner(f"Loading/Creating vector store from '{DATA_DIR}'..."):
        documents = load_documents()
        text_chunks = split_documents(documents)
        vectorstore = create_or_load_vectorstore(text_chunks, embeddings)

    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
        st.success(f"Retriever ready (using {VECTORSTORE_DIR}).")
    else:
        st.warning(f"Could not load/create vector store from '{DATA_DIR}'. RAG disabled.", icon="⚠️")

    # 4. Setup RAG Chain
    with st.spinner("Setting up RAG chain..."):
        rag_chain = setup_rag_chain(llm, retriever)  # Adapts based on retriever
    st.success("Pipeline ready!")

    return rag_chain, retriever  # Return both for potential use

# --- Streamlit App UI ---

st.set_page_config(page_title="Modular RAG Agent", layout="wide")
st.title("📄 Deepseek RAG Agent")
st.markdown(f"Using Ollama (`{MODEL_NAME}`) and data from `{DATA_DIR}`.")

# Sidebar - File Uploader and Options
with st.sidebar:
    st.header("Options")
    
    # File uploader widget for PDFs
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf:
        save_uploaded_file(uploaded_pdf)
        st.success(f"Uploaded file '{uploaded_pdf.name}' saved to '{DATA_DIR}'.")

    # Button to refresh the pipeline (loads newly uploaded documents)
    if st.button("Refresh Pipeline"):
        st.experimental_rerun()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Pipeline Status")
    st.markdown(f"**LLM Model:** `{MODEL_NAME}`")
    st.markdown(f"**Embedding Model:** `{EMBEDDING_MODEL_NAME}`")
    st.markdown(f"**Data Directory:** `{DATA_DIR}`")
    st.markdown(f"**Vector Store:** `{VECTORSTORE_DIR}`")
    st.markdown(f"**Retriever Active:** {'Yes' if 'retriever' in locals() and 'retriever' else 'No'}")
    st.markdown(f"**Search K:** `{SEARCH_K}`")

# Load the pipeline using the cached function
try:
    rag_chain, retriever = load_pipeline()
except Exception as e:
    st.error(f"Fatal error during pipeline initialization: {e}", icon="🚨")
    logger.error("Pipeline initialization failed.", exc_info=True)
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input and process query
if prompt := st.chat_input("Ask your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = ""
        try:
            response_stream = rag_chain.stream(prompt)
            full_response = st.write_stream(response_stream)

            if retriever:
                with st.expander("Retrieved Context Snippets"):
                    retrieved_docs = retriever.invoke(prompt)
                    if retrieved_docs:
                        for i, doc in enumerate(retrieved_docs):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'N/A')
                            st.caption(f"Source {i+1}: {os.path.basename(source)} (Page: {page})")
                            st.markdown(f"> {doc.page_content[:350]}...")
                    else:
                        st.write("No relevant context found in documents for this query.")

        except Exception as e:
            logger.error(f"Error processing query '{prompt}': {e}", exc_info=True)
            full_response = f"Sorry, an error occurred: {e}"
            st.error(full_response, icon="🔥")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
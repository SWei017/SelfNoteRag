import streamlit as st
from main import SelfNoteRAGPipeline
import json
import os
from urllib.parse import quote

def load_config():
    with open("config.json", "r") as f:
        config = json.load(f)
    return config

def load_rag_pipeline(config):
    rag_pipeline = SelfNoteRAGPipeline(config)
    return rag_pipeline

def save_config(config):
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)


# Initialize only once
if "config" not in st.session_state:
    st.session_state.config = load_config()

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = load_rag_pipeline(st.session_state.config)

# Access later using:
config = st.session_state.config
rag_pipeline = st.session_state.rag_pipeline

# wider ui
st.set_page_config(layout="wide")

st.markdown(
    '''
    <style>
    div.stButton > button {
        font-size: 1.2rem;
        padding: 0.75em 0.5em;
        width: 100%;
        margin-bottom: 0.5em;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# ----------------- START OF SIDEBAR --------------------
# Sidebar UI
st.sidebar.header("SelfNoteRAG")

# Tabs for navigation using vertical buttons
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Chunking Panel"
    
if st.sidebar.button("Chat"):
    st.session_state["active_page"] = "Chat"
if st.sidebar.button("Chunking Panel"):
    st.session_state["active_page"] = "Chunking Panel"


page = st.session_state["active_page"]

# Settings in an expander
with st.sidebar.expander("⚙️ Settings", expanded=False):
    with st.form("settings_form"):
        # Embedding Model
        embedding_model = config.get("embedding_model", "all-minilm")
        config["embedding_model"] = st.selectbox(
            label="Embedding Model",
            options=["all-minilm", "sentence-transformers", "openai", "cohere"],
            index=["all-minilm", "sentence-transformers", "openai", "cohere"].index(embedding_model)
        )

        # Generator Model
        generator_model = config.get("generator_model", "llama3.2")
        config["generator_model"] = st.selectbox(
            label="Generator Model",
            options=["llama3.2", "gpt-3.5-turbo", "gpt-4", "claude-3"],
            index=["llama3.2", "gpt-3.5-turbo", "gpt-4", "claude-3"].index(generator_model)
        )

        # Vector Store Folder Path
        vector_store_folderpath = config.get("vector_store_folderpath", "")
        config["vector_store_folderpath"] = st.text_input(
            label="Vector Store Folder Path",
            value=vector_store_folderpath,
            help="Path to store the FAISS index"
        )

        # Document Folder
        vault_directory = config.get("vault_directory", "")
        config["vault_directory"] = st.text_input(
            label="Document Folder",
            value=vault_directory,
            help="Path to your documents folder"
        )

        # Chunk Size
        chunk_size = config.get("chunk_size", 500)
        config["chunk_size"] = st.number_input(
            label="Chunk Size",
            value=chunk_size,
            help="Size of text chunks for embedding",
            step=10
        )

        # Chunk overlap
        chunk_overlap = config.get("chunk_overlap", 100)
        config["chunk_overlap"] = st.number_input(
            label="Chunk Overlap",
            value=chunk_overlap,
            help="Overlap size of text chunks for embedding",
            step=10
        )


        saved = st.form_submit_button("💾 Save Settings")
        # Save updated config
        if saved:
            save_config(config)
            st.success("Settings saved successfully!")

        # Reload RAG pipeline with new config
        if st.form_submit_button("🔄 Reload Pipeline"):
            rag_pipeline = load_rag_pipeline(config)
            st.success("Pipeline reloaded!")
# ----------------- END OF SIDEBAR --------------------

col0, col1, col2, col3 = st.columns([15, 35, 35, 15])
col0_1, col1_1, col2_1 = st.columns([15, 70, 15])

if page == "Chat":
    with col1:
        st.title("💬 Text Input & Output Demo")

        # Input from user
        user_input = st.text_area("Enter your question or text:", height=150)

        # Action button
        if st.button("Submit"):
            if user_input.strip():
                content, response = rag_pipeline.ask(user_input)

                # Store in session_state
                st.session_state["last_response"] = response
                st.session_state["last_content"] = content

        # Output display containers
        output_container = st.container()
        

        # Render the result if exists
        if "last_response" in st.session_state:
            with output_container:
                st.markdown("### 📤 Output")
                st.write(st.session_state["last_response"])



    with col2:
        # for document info
        content_container = st.container()
        if "last_response" in st.session_state:
            with content_container:
                encoded_file = quote(os.path.basename(str(rag_pipeline.retrieved_document_filepath)))
                vault_name = os.path.basename(vault_directory)

                obsidian_url = f"obsidian://open?vault={quote(vault_name)}&file={encoded_file}"
                st.markdown("### 📤 Content")
                st.write(f"[🔗 Open in Obsidian]({obsidian_url})", unsafe_allow_html=True)
                st.markdown(st.session_state["last_content"])

elif page == "Chunking Panel":
    with col1_1:
        st.title("Embedding Info")
        # button to retrain
        if st.button("Embed"):
            rag_pipeline.embed(config["vault_directory"])
            st.session_state["chunked_docs"] = rag_pipeline.get_chunkings()

        if "chunked_docs" in st.session_state:
            st.subheader("Chunked Documents by Title")
            st.json(st.session_state["chunked_docs"])





import streamlit as st
from main import SelfNoteRAGPipeline
import json

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

config = load_config()
rag_pipeline = load_rag_pipeline(config)

st.title("💬 Text Input & Output Demo")


# ----------------- START OF SIDEBAR --------------------
# Sidebar UI
st.sidebar.header("Settings")

# Embedding Model
embedding_model = config.get("embedding_model", "all-minilm")
config["embedding_model"] = st.sidebar.selectbox(
    label="Embedding Model",
    options=["all-minilm", "sentence-transformers", "openai", "cohere"],
    index=["all-minilm", "sentence-transformers", "openai", "cohere"].index(embedding_model)
)

# Generator Model
generator_model = config.get("generator_model", "llama3.2")
config["generator_model"] = st.sidebar.selectbox(
    label="Generator Model",
    options=["llama3.2", "gpt-3.5-turbo", "gpt-4", "claude-3"],
    index=["llama3.2", "gpt-3.5-turbo", "gpt-4", "claude-3"].index(generator_model)
)

# Vector Store Folder Path
vector_store_folderpath = config.get("vector_store_folderpath", "")
config["vector_store_folderpath"] = st.sidebar.text_input(
    label="Vector Store Folder Path",
    value=vector_store_folderpath,
    help="Path to store the FAISS index"
)

# Document Folder
document_folder = config.get("document_folder", "")
config["document_folder"] = st.sidebar.text_input(
    label="Document Folder",
    value=document_folder,
    help="Path to your documents folder"
)

# Chunk Size
chunk_size = config.get("chunk_size", "500")
config["chunk_size"] = st.sidebar.text_input(
    label="Chunk Size",
    value=chunk_size,
    help="Size of text chunks for embedding"
)

# Save updated config
if st.sidebar.button("💾 Save Settings"):
    save_config(config)
    st.sidebar.success("Settings saved successfully!")

# Reload RAG pipeline with new config
if st.sidebar.button("🔄 Reload Pipeline"):
    rag_pipeline = load_rag_pipeline(config)
    st.sidebar.success("Pipeline reloaded!")

# ----------------- END OF SIDEBAR --------------------

left, middle, right = st.columns([10,15,0.5])

# button to retrain
if left.button("Embed"):
    rag_pipeline.embed(config["document_folder"])

# Input from user
user_input = middle.text_area("Enter your question or text:", height=150)

# Action button
if middle.button("Submit"):
    # Simulated processing (replace this with your logic)
    content, response = rag_pipeline.ask(user_input)
    # Output display
    middle.markdown("### 📤 Output")
    middle.write(response)

    middle.markdown("### 📤 Content")
    middle.write(content)



    middle.markdown("### 📤 Content")
    middle.write(content)



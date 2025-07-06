# SelfNoteRAG

SelfNoteRAG is a retrieval-augmented generation (RAG) system for semantic search over your Obsidian vault, powered by local LLMs and vector search.

⚠️ Note: This project is currently in active development. Features, dependencies, and structure may change. Contributions, suggestions, and feedback are welcome!

## Features
- **Basic RAG system (embed, retrieve, prompt)** using local LLMs (Ollama) and vector databases (FAISS)
- **Streamlit web interface** for chat and chunking panel
- **Configurable models and chunking parameters** via sidebar settings
- **Obsidian integration**: open retrieved notes directly in Obsidian

### Required Python Packages
There is no `requirements.txt` yet. Install these manually:
- streamlit
- langchain
- langchain_community
- langchain_core
- langchain_ollama
- sentence-transformers
- faiss-cpu
- watchdog
- markdown
- (and any other dependencies as needed)

### Required LLM backend
- Ollama must be installed and running with your desired local model (e.g., llama3).

## Configuration
Edit `config.json` to set your embedding/generator models, vector store path, and your Obsidian vault directory:
```json
{
  "embedding_model": "all-minilm",
  "generator_model": "llama3.2",
  "vector_store_folderpath": "C:/Projects/SelfNoteRAG/faiss_index",
  "vault_directory": "C:/Users/yourname/Documents/Obsidian Vault",
  "chunk_size": 700,
  "chunk_overlap": 220
}
```
You can also adjust these settings in the Streamlit sidebar.

## Usage
### Quick Start (Windows)
1. Activate your environment and run:
   ```sh
   streamlit run app.py --server.fileWatcherType none
   ```
2. Open the Streamlit UI in your browser (usually at http://localhost:8501).
3. Use the sidebar to configure models and folders, then use the Chat or Chunking Panel tabs.

### Embedding Notes
- Use the **Chunking Panel** tab and click the Embed button to process your notes and build the vector store.

### Querying
- In the **Chat** tab, ask questions and get responses based on relevant notes.
- Retrieved chunks are reranked and passed to the LLM for generation.
- Click on any source link to open the corresponding note in Obsidian.

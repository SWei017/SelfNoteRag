from .base import Generator
from langchain_community.vectorstores import FAISS
from embeddings.base import Embedding
from langchain_community.llms import Ollama 
from typing import Tuple

class OllamaLocalGenerator(Generator):
    """Abstract base class for text splitters."""
    def __init__(self, embedding: Embedding, model: str = "llama3.2"):
        super().__init__()
        self.embedding = embedding
        self.vector_store = self.embedding.vector_store
        self.llm = Ollama(model=model)

    def ask(self, query: str) -> Tuple[str, str]:
        """Ask llm."""
        query_vector = self.vector_store.similarity_search(query, k=1)

        context = "\n\n".join([doc.page_content for doc in query_vector])
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        llm = Ollama(model="llama3.2")
        response = llm.predict(prompt)
        return (context, response)

    def load_vector_store(self, filepath: str) -> FAISS:
        return self.embedding.load_vector_store()
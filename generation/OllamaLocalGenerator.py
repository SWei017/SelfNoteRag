from .base import Generator
from retrieval.Retriever import Retriever
from langchain_community.vectorstores import FAISS
from embeddings.base import Embedding
from langchain_community.llms import Ollama 
from typing import Tuple

class OllamaLocalGenerator(Generator):
    """Abstract base class for text splitters."""
    def __init__(self, embedding: Embedding, retrieval: Retriever, model: str = "llama3.2"):
        super().__init__()
        self.embedding = embedding
        self.vector_store = self.embedding.vector_store
        self.llm = Ollama(model=model)
        self.retrieval = retrieval

    def ask(self, query: str) -> Tuple[str, str]:
        """Ask llm."""

        context = self.retrieval.retrieve(query=query)

        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        llm = Ollama(model="llama3.2")
        response = llm.invoke(prompt)
        return (context, response)

    def load_vector_store(self) -> FAISS:
        return self.embedding.load_vector_store()
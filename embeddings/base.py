from abc import ABC, abstractmethod

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

class Embedding(ABC):
    """Abstract base class for text splitters."""
    def __init__(self):
        self.vector_store = None
        
    @abstractmethod
    def embed(self, documents: List[Document]) -> FAISS:
        """Embed document into vectors."""
        pass

    @abstractmethod
    def load_vector_store(self) -> FAISS:
        pass
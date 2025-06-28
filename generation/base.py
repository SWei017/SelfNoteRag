from abc import ABC, abstractmethod

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

class Generator(ABC):
    """Abstract base class for text splitters."""
    
    @abstractmethod
    def ask(self, prompt: str) -> str:
        """Ask llm."""
        pass

    @abstractmethod
    def load_vector_store(self) -> FAISS:
        pass
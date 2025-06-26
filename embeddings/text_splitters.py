
from abc import ABC, abstractmethod

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class BaseTextSplitter(ABC):
    """Abstract base class for text splitters."""
    
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        pass
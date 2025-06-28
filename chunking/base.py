from abc import ABC, abstractmethod

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from pathlib import Path

class Chunking(ABC):

    def __init__(self, documents: List[Document] = None):
        self.documents = documents if documents else []

    @abstractmethod
    def split_text(self, folder_path: str) -> List[Document]:
        pass
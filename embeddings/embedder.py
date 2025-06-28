from .base import Embedding
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from typing import List
from uuid import uuid4
from pathlib import Path

class OllamaLocalEmbedding(Embedding):
    def __init__(self, vector_store_folderpath: str, model: str = "all-minilm"):
        super().__init__()
        self.embedding = OllamaEmbeddings(model=model)

        self.index = faiss.IndexFlatL2(len(self.embedding.embed_query("hello world"))) 
        self.folderpath = vector_store_folderpath
        self.vector_store = self.load_vector_store()

    def embed(self, documents: List[Document]) -> FAISS:
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        self.vector_store.save_local(self.folderpath)
        return self.vector_store

    def load_vector_store(self) -> FAISS:
        folder = Path(self.folderpath)
        if not folder.is_dir():
            return FAISS(
                            embedding_function=self.embedding,
                            index=self.index,
                            docstore=InMemoryDocstore(),
                            index_to_docstore_id={},
                        )
        
        return FAISS.load_local(
            self.folderpath, self.embedding, allow_dangerous_deserialization=True
        )
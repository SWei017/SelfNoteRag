from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from typing import List
import torch, gc

class Retriever():
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', device='cpu') # cross-encoder for reranking

    def retrieve(self, query: str, title: str = None) -> Document:
        documents = self.vector_store.similarity_search(query, k=5)

        # rerank
        document = self.rerank(query, documents)

        return document
    
    def rerank(self, query: str, documents: List[Document]) -> Document:
        """Rerank top-k rseults and return 1st rank as Document"""
        corpus = [doc.page_content for doc in documents]
        ranks = self.reranker.rank(query, corpus)

        document = documents[ranks[0]['corpus_id']] # only take the highest rank

        return document
    
    def __del__(self):
        """Destructor"""
        del self.reranker
        gc.collect()
        print(f"{self} disposed.")


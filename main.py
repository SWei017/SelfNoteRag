from generation.OllamaLocalGenerator import OllamaLocalGenerator
from chunking.text_splitters import MarkdownTextSplitter, RecursiveTextSplitter
from embeddings.embedder import OllamaLocalEmbedding
from reader.MarkdownReader import MarkdownReader
from retrieval.Retriever import Retriever
from typing import Dict, List


class SelfNoteRAGPipeline:
    def __init__(self, config):
        reader = MarkdownReader()

        chunk_size = config.get("chunk_size", 500)
        chunk_overlap = config.get("chunk_overlap", 100)

        self.splitter = RecursiveTextSplitter(
            reader=reader, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap)
        self.embedder = OllamaLocalEmbedding(vector_store_folderpath=config["vector_store_folderpath"])
        self.retrieval = Retriever(self.embedder.vector_store)
        self.generator = OllamaLocalGenerator(embedding=self.embedder)
        
        self.load_vector_store()

    def load_vector_store(self):
        self.vector_store = self.embedder.load_vector_store()
        
    def embed(self, folder_path):
        documents = self.splitter.split_text(folder_path)
        self.vector_store = self.embedder.embed(documents)

    def ask(self, query: str):
        retrieved_document = self.retrieval.retrieve(query=query)

        self.retrieved_document_filepath = retrieved_document.metadata.get("filepath", "")
        context, response = self.generator.ask(retrieved_document.page_content, query)

        return context, response
    
    def get_chunkings(self) -> Dict[str, List[str]]:
        return self.splitter.get_chunkings()
    





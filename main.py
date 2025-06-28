from generation.OllamaLocalGenerator import OllamaLocalGenerator
from chunking.text_splitters import MarkdownTextSplitter, RecursiveTextSplitter
from embeddings.embedder import OllamaLocalEmbedding
from reader.MarkdownReader import MarkdownReader
from retrieval.Retriever import Retriever

class SelfNoteRAGPipeline:
    def __init__(self, config):
        reader = MarkdownReader()
        self.splitter = RecursiveTextSplitter(reader=reader)
        self.embedder = OllamaLocalEmbedding(vector_store_folderpath=config["vector_store_folderpath"])
        self.retrieval = Retriever(self.embedder.vector_store)
        self.generator = OllamaLocalGenerator(embedding=self.embedder, retrieval=self.retrieval)
        
        self.load_vector_store()

    def load_vector_store(self):
        self.vector_store = self.embedder.load_vector_store()
        
    
    def embed(self, folder_path):
        documents = self.splitter.split_text(folder_path)
        self.vector_store = self.embedder.embed(documents)

    def ask(self, query: str):
        context, response = self.generator.ask(query)

        return context, response




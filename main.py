from config import CONFIG
from generation.OllamaLocalGenerator import OllamaLocalGenerator
from chunking.text_splitters import MarkdownTextSplitter, RecursiveTextSplitter
from embeddings.embedder import OllamaLocalEmbedding
from reader.MarkdownReader import MarkdownReader

class SelfNoteRAGPipeline:
    def __init__(self, config):
        self.config = config
        reader = MarkdownReader()
        self.splitter = RecursiveTextSplitter(reader=reader)
        self.embedder = OllamaLocalEmbedding(vector_store_folderpath=self.config["vector_store_folderpath"])
        self.generator = OllamaLocalGenerator(embedding=self.embedder)
        
        self.load_vector_store()

    def load_vector_store(self):
        self.vector_store = self.embedder.load_vector_store()
        
    
    def embed(self, folder_path):
        documents = self.splitter.split_text(folder_path)
        self.vector_store = self.embedder.embed(documents)

    def ask(self, query: str):
        context, response = self.generator.ask(query)

        return context, response




from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from .base import Chunking
import markdown
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from reader.base import Reader

class NaiveTextSplitter(Chunking):
    def __init__(self, reader: Reader):
        self.reader = reader

    def split_text(self, folder_path) -> List[Document]:
        filepaths = self.reader.read_folder(folder_path)
        for filepath in filepaths:
            f = open(filepath, 'r')
            htmlmarkdown=markdown.markdown( f.read() )
            # htmlmarkdowns.append(htmlmarkdown)
            
            self.documents.append(Document(
                page_content=htmlmarkdown,
                metadata={"Title": "html"},
            ))

        return self.documents
    
class MarkdownTextSplitter(Chunking):
    def __init__(self, reader: Reader, headers_to_split_on: List[Tuple] = None):
        self.reader = reader
        default_header_to_split_on = [
                                        ("#", "Header 1"),
                                        ("##", "Header 2"),
                                        ("###", "Header 3"),
                                    ]
        headers_to_split_on = headers_to_split_on if headers_to_split_on else default_header_to_split_on
        self.splitter = MarkdownTextSplitter(headers_to_split_on)

    def split_text(self, folder_path: str) -> List[Document]:
        filepaths = self.reader.read_folder(folder_path)
        for filepath in filepaths:
            if filepath.is_file():
                with open(filepath, 'r', encoding='utf-8') as f:
                    markdown_text = f.read()
                    
                    md_header_splits = self.splitter.split_text(markdown_text)

                    for document in md_header_splits:
                        self.documents.append(document)

        return self.documents
    
class RecursiveTextSplitter(Chunking):
    def __init__(self, reader: Reader, documents: List[Document] = None, chunk_size: int = 500, chunk_overlap: int = 100, separators: List[str] = None):
        super().__init__(documents)
        self.reader = reader
        separators = separators if separators else ["\n\n", "\n", "#", "##", "###"]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators = separators
        )

    def split_text(self, folder_path: str) -> List[Document]:
        filepaths = self.reader.read_folder(folder_path)
        documents = []
        for filepath in filepaths:
            if filepath.is_file():
                with open(filepath, 'r', encoding='utf-8') as f:
                    filename = filepath.stem
                    markdown_text = f.read()
                    
                    doc = Document(page_content=markdown_text, metadata={"filepath": filepath, "title": filename})
                    documents.append(doc)

        self.documents = self.splitter.split_documents(documents)
        
        return self.documents
    
    def get_chunkings(self) -> Dict[str, List[str]]:
        documents_dict = {}
        for doc in self.documents:
            title =  doc.metadata.get('title')
            if title:
                documents_dict.setdefault(doc.metadata['title'], []).append(doc.page_content)

        return documents_dict
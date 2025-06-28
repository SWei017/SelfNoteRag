from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama 
import markdown
import faiss
from langchain_text_splitters import MarkdownHeaderTextSplitter

from uuid import uuid4

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from typing import List, Dict
from config import CONFIG

def load():
    embeddings = OllamaEmbeddings(model="all-minilm")
    new_vector_store = FAISS.load_local(
        CONFIG["vector_store_filename"], embeddings, allow_dangerous_deserialization=True
    )

    print(f"Vector store loaded")
    return new_vector_store

def ask(prompt: str):
    llm = Ollama(model="llama3.2")
    response = llm.predict(prompt)

    return response

def embed(filepath: str):
    documents = []
    for file in Path(rf'{filepath}').rglob('*.md'):
        if file.is_file():
            f = open(file, 'r')
            htmlmarkdown=markdown.markdown( f.read() )
            # htmlmarkdowns.append(htmlmarkdown)
            
            documents.append(Document(
                page_content=htmlmarkdown,
                metadata={"Title": "html"},
            ))

    embeddings = OllamaEmbeddings(model="all-minilm")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))  # L2 = Euclidean distance

    # langchain vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    vector_store.save_local(CONFIG["vector_store_filename"])

    print(f"Vector store save to {CONFIG["vector_store_filename"]}")
    return vector_store

def embed_with_markdown_splitter(filepath: str):
    documents = []
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    for file in Path(rf'{filepath}').rglob('*.md'):
        if file.is_file():
            with open(file, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
                
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
                md_header_splits = markdown_splitter.split_text(markdown_text)

                for document in md_header_splits:
                    documents.append(document)

    embeddings = OllamaEmbeddings(model="all-minilm")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world one two three")))  # L2 = Euclidean distance

    # langchain vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    vector_store.save_local(CONFIG["vector_store_filename"])

    print(f"Vector store save to {CONFIG["vector_store_filename"]}")
    return vector_store

def embed_with_recursive_splitter(filepath: str):
        documents = []
        for file in Path(rf'{filepath}').rglob('*.md'):
            if file.is_file():
                with open(file, 'r', encoding='utf-8') as f:
                    
                    filename = file.stem
                    markdown_text = f.read()
                    
                    doc = Document(page_content=markdown_text, metadata={"Title": filename})
                    documents.append(doc)


        recursive_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100,
                        separators=["\n\n", "\n", "#", "##", "###"])
        splitted_doc = recursive_splitter.split_documents(documents)

        embeddings = OllamaEmbeddings(model="all-minilm")
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world one two three")))  # L2 = Euclidean distance

        # langchain vector store
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        uuids = [str(uuid4()) for _ in range(len(splitted_doc))]
        vector_store.add_documents(documents=splitted_doc, ids=uuids)

        vector_store.save_local(CONFIG["vector_store_filename"])

        print(f"Vector store save to {CONFIG["vector_store_filename"]}")
        return vector_store
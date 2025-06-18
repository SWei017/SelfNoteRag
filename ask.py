from utils import embed, load
from config import CONFIG
import sys
from langchain_community.llms import Ollama 

def main(query: str):
    vector_store = load()

    query_vector = vector_store.similarity_search(query, k=1)

    context = "\n\n".join([doc.page_content for doc in query_vector])
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    llm = Ollama(model="llama3.2")
    response = llm.predict(prompt)

    return context, response

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <argument>")
        sys.exit(1)

    arg = sys.argv[1]

    context, response = main(arg)
    print(f"Context:\n {context}")
    print(f"Response:\n {response}")

class Retriever():
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, title: str = None) -> str:
        query_vector = self.vector_store.similarity_search(query, k=1)

        # metadata filter?
        context = ''
        if title:
            context = "\n\n".join([doc.page_content for doc in query_vector if doc.metadata['title'].startswith(title)])
        else:
            context = "\n\n".join([doc.page_content for doc in query_vector])

        return context
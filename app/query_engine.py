def query_pdf(vector_store, query):
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

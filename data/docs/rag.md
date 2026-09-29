# Retrieval augmented generation

Retrieval augmented generation (RAG) grounds an answer in passages fetched from a document collection. A typical pipeline loads documents, splits them into chunks, builds a search index, retrieves relevant chunks, and sends those chunks with the question to a language model. A useful answer cites the source passages. If retrieval finds no support, the system should say that it cannot answer from the documents.

Chunk size and overlap change retrieval behavior. Small chunks can improve precision but lose context. Large chunks can preserve context but add irrelevant text and increase prompt cost. Evaluate settings against labeled questions rather than selecting them by intuition.

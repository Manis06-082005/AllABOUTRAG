import bm25s

documents = [
    "Python is used for machine learning",
    "SQL is used for databases"
]

tokens = bm25s.tokenize(documents)
retriever=bm25s.BM25()
indexing=retriever.index(tokens)

query="What is Used for Machine leanring"
tokenized_query=bm25s.tokenize(query)
results, scores = retriever.retrieve(tokenized_query, k=2)
print("\nResults:")
for i, score in zip(results[0], scores[0]):
    print(f"Score: {score:.4f} | {documents[i]}")
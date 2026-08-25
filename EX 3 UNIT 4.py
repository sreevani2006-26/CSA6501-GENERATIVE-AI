import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Artificial Intelligence enables machines to perform intelligent tasks.",
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses multi-layer neural networks.",
    "Natural Language Processing helps computers understand human language.",
    "Python is widely used for machine learning and data science.",
    "FAISS is used for efficient similarity search and vector retrieval."
]

document_embeddings = model.encode(documents)

document_embeddings = np.array(document_embeddings).astype("float32")
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

print("Documents stored in vector database:", index.ntotal)
query = input("\nEnter your search query: ")
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")
k = 3
distances, indices = index.search(query_embedding, k)
print(f"\nTop {k} Retrieved Documents:\n")

for rank, (distance, idx) in enumerate(
    zip(distances[0], indices[0]), start=1
):
    print(f"Rank {rank}")
    print("Document:", documents[idx])
    print("Distance:", round(float(distance), 4))
    print("-" * 50)

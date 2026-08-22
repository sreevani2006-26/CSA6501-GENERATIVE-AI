import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Documents to store in the vector database
documents = [
    "Artificial Intelligence enables machines to perform intelligent tasks.",
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses multi-layer neural networks.",
    "Natural Language Processing helps computers understand human language.",
    "Python is widely used for machine learning and data science.",
    "FAISS is used for efficient similarity search and vector retrieval."
]

# Generate document embeddings
document_embeddings = model.encode(documents)

# Convert embeddings to float32
document_embeddings = np.array(document_embeddings).astype("float32")

# Create FAISS vector database
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Store document embeddings in the vector database
index.add(document_embeddings)

print("Documents stored in vector database:", index.ntotal)

# Get user query
query = input("\nEnter your search query: ")

# Generate query embedding
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# Number of top results to retrieve
k = 3

# Perform top-k similarity search
distances, indices = index.search(query_embedding, k)

# Display retrieved documents
print(f"\nTop {k} Retrieved Documents:\n")

for rank, (distance, idx) in enumerate(
    zip(distances[0], indices[0]), start=1
):
    print(f"Rank {rank}")
    print("Document:", documents[idx])
    print("Distance:", round(float(distance), 4))
    print("-" * 50)

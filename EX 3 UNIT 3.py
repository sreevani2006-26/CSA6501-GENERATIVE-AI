import faiss
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Sample documents
documents = [
    "Artificial Intelligence enables machines to perform intelligent tasks.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Python is widely used for data science and machine learning.",
    "Natural Language Processing helps computers understand human language.",
    "Football is a popular sport played by two teams."
]

# 3. Generate embeddings
document_embeddings = model.encode(documents)

# Convert embeddings to float32 for FAISS
document_embeddings = document_embeddings.astype("float32")

# 4. Create the FAISS vector database
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 5. Add document embeddings to the database
index.add(document_embeddings)

print("Number of documents stored:", index.ntotal)

# 6. Get search query
query = input("\nEnter your search query: ")

# 7. Generate query embedding
query_embedding = model.encode([query])
query_embedding = query_embedding.astype("float32")

# 8. Retrieve top 3 similar documents
k = 3
distances, indices = index.search(query_embedding, k)

# 9. Display results
print("\nTop Similar Documents:\n")

for i in range(k):
    document_index = indices[0][i]
    distance = distances[0][i]

    print(f"Rank {i + 1}")
    print("Document:", documents[document_index])
    print("Similarity Distance:", round(float(distance), 4))
    print("-" * 50)

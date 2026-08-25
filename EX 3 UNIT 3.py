import faiss
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Artificial Intelligence enables machines to perform intelligent tasks.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Python is widely used for data science and machine learning.",
    "Natural Language Processing helps computers understand human language.",
    "Football is a popular sport played by two teams."
]

document_embeddings = model.encode(documents)
document_embeddings = document_embeddings.astype("float32")

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

print("Number of documents stored:", index.ntotal)

query = input("\nEnter your search query: ")
query_embedding = model.encode([query])
query_embedding = query_embedding.astype("float32")
k = 3
distances, indices = index.search(query_embedding, k)
print("\nTop Similar Documents:\n")
for i in range(k):
    document_index = indices[0][i]
    distance = distances[0][i]

    print(f"Rank {i + 1}")
    print("Document:", documents[document_index])
    print("Similarity Distance:", round(float(distance), 4))
    print("-" * 50)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
query = input("Enter your search query: ")
query_embedding = model.encode([query])
similarity_scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

ranked_results = sorted(
    enumerate(similarity_scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 3 Semantic Search Results:\n")

for index, score in ranked_results[:3]:
    print("Document:", documents[index])
    print("Cosine Similarity:", round(float(score), 4))
    print("-" * 50)

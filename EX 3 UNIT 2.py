from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create a collection of documents
documents = [
    "Artificial Intelligence enables machines to perform intelligent tasks.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Python is widely used for data science and machine learning.",
    "Natural Language Processing helps computers understand human language.",
    "Football is a popular sport played by two teams."
]

# 3. Generate embeddings for all documents
document_embeddings = model.encode(documents)

# 4. Get the search query from the user
query = input("Enter your search query: ")

# 5. Generate embedding for the query
query_embedding = model.encode([query])

# 6. Calculate cosine similarity between query and documents
similarity_scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

# 7. Get documents in descending order of similarity
ranked_results = sorted(
    enumerate(similarity_scores),
    key=lambda x: x[1],
    reverse=True
)

# 8. Display top 3 most relevant documents
print("\nTop 3 Semantic Search Results:\n")

for index, score in ranked_results[:3]:
    print("Document:", documents[index])
    print("Cosine Similarity:", round(float(score), 4))
    print("-" * 50)

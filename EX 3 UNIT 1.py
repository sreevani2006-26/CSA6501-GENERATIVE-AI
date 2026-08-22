from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
documents = [
    "Artificial Intelligence is transforming the world.",
    "Machine learning is a branch of artificial intelligence.",
    "Python is a popular programming language.",
    "Deep learning uses neural networks.",
    "I like to play football."
]

# Generate embeddings for documents
document_embeddings = model.encode(documents)

# User search query
query = input("Enter your search query: ")

# Generate embedding for query
query_embedding = model.encode([query])

# Calculate cosine similarity
similarity_scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

# Sort results from highest similarity to lowest
results = sorted(
    enumerate(similarity_scores),
    key=lambda x: x[1],
    reverse=True
)

# Display results
print("\nSemantic Search Results:\n")

for index, score in results:
    print("Text:", documents[index])
    print("Similarity Score:", round(float(score), 4))
    print()

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 1. Documents
documents = [
    "Artificial Intelligence enables machines to perform tasks that normally require human intelligence.",
    "Machine Learning is a subset of Artificial Intelligence that learns patterns from data.",
    "Natural Language Processing enables computers to understand and process human language.",
    "Retrieval-Augmented Generation combines information retrieval with text generation.",
    "FAISS is a library used for efficient similarity search on vector embeddings."
]

# 2. Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Generate document embeddings
document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")

# 4. Create FAISS vector database
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

print("Documents stored:", index.ntotal)

# 5. Load text generation model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# 6. Get question from user
question = input("\nAsk a question: ")

# 7. Convert question into embedding
question_embedding = embedding_model.encode(
    [question],
    convert_to_numpy=True
).astype("float32")

# 8. Retrieve top 2 relevant documents
k = 2
distances, indices = index.search(question_embedding, k)

# 9. Create context from retrieved documents
context = " ".join([documents[i] for i in indices[0]])

# 10. Create RAG prompt
prompt = f"""
Answer the question using only the given context.

Context:
{context}

Question:
{question}

Answer:
"""

# 11. Generate answer
result = generator(
    prompt,
    max_new_tokens=50
)

# 12. Display results
print("\n--- RAG Document Question Answering System ---")

print("\nQuestion:")
print(question)

print("\nRetrieved Documents:")
for rank, i in enumerate(indices[0], start=1):
    print(f"{rank}. {documents[i]}")

print("\nAnswer:")
print(result[0]["generated_text"])

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

documents = [
    "Artificial Intelligence enables machines to perform tasks that normally require human intelligence.",
    "Machine Learning is a subset of Artificial Intelligence that learns patterns from data.",
    "Natural Language Processing enables computers to understand and process human language.",
    "Retrieval-Augmented Generation combines information retrieval with text generation.",
    "FAISS is a library used for efficient similarity search on vector embeddings."
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")
dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

print("Documents stored:", index.ntotal)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)
question = input("\nAsk a question: ")
question_embedding = embedding_model.encode(
    [question],
    convert_to_numpy=True
).astype("float32")
k = 2
distances, indices = index.search(question_embedding, k)
context = " ".join([documents[i] for i in indices[0]])
prompt = f"""
Answer the question using only the given context.

Context:
{context}

Question:
{question}

Answer:
"""
result = generator(
    prompt,
    max_new_tokens=50
)
print("\n--- RAG Document Question Answering System ---")

print("\nQuestion:")
print(question)

print("\nRetrieved Documents:")
for rank, i in enumerate(indices[0], start=1):
    print(f"{rank}. {documents[i]}")

print("\nAnswer:")
print(result[0]["generated_text"])

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --------------------------------------------------
# 1. DOCUMENT LOADING
# --------------------------------------------------

document = """
Artificial Intelligence is a branch of computer science that enables
machines to perform tasks that normally require human intelligence.

Machine Learning is a subset of Artificial Intelligence. It allows
computers to learn patterns from data without being explicitly programmed.

Natural Language Processing, also called NLP, enables computers to
understand, interpret, and process human language.

Retrieval-Augmented Generation, known as RAG, combines document retrieval
with answer generation. A RAG system first retrieves relevant information
and then uses that information to generate an answer.

FAISS is a library developed for efficient similarity search and retrieval
of vector embeddings.
"""

print("Document loaded successfully!")

# --------------------------------------------------
# 2. TEXT CHUNKING
# --------------------------------------------------

def chunk_text(text, chunk_size=150):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


chunks = chunk_text(document, chunk_size=30)

print("\nNumber of chunks:", len(chunks))

# --------------------------------------------------
# 3. GENERATE EMBEDDINGS
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
).astype("float32")

print("Embeddings generated successfully!")

# --------------------------------------------------
# 4. CREATE FAISS VECTOR DATABASE
# --------------------------------------------------

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Store chunk embeddings
index.add(chunk_embeddings)

print("Chunks stored in FAISS:", index.ntotal)

# --------------------------------------------------
# 5. LOAD ANSWER GENERATION MODEL
# --------------------------------------------------

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("Answer generation model loaded!")

# --------------------------------------------------
# 6. GET USER QUESTION
# --------------------------------------------------

question = input("\nAsk a question: ")

# --------------------------------------------------
# 7. RETRIEVAL
# --------------------------------------------------

query_embedding = embedding_model.encode(
    [question],
    convert_to_numpy=True
).astype("float32")

# Retrieve top 2 relevant chunks
k = 2

distances, indices = index.search(
    query_embedding,
    k
)

# Combine retrieved chunks
retrieved_chunks = [
    chunks[i] for i in indices[0]
]

context = " ".join(retrieved_chunks)

# --------------------------------------------------
# 8. RAG ANSWER GENERATION
# --------------------------------------------------

prompt = f"""
Answer the question using only the information given below.

Context:
{context}

Question:
{question}

Answer:
"""

result = generator(
    prompt,
    max_new_tokens=80
)

answer = result[0]["generated_text"]

# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n====================================")
print("       RAG QUESTION ANSWERING")
print("====================================")

print("\nQuestion:")
print(question)

print("\nRetrieved Chunks:")
for rank, chunk in enumerate(retrieved_chunks, start=1):
    print(f"\nChunk {rank}:")
    print(chunk)

print("\nGenerated Answer:")
print(answer)

print("\n====================================")

import warnings
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import pipeline


# -------------------------------------------------
# 1. CREATE DOMAIN-SPECIFIC DOCUMENTS
# -------------------------------------------------

documents = [
    Document(
        page_content="Artificial Intelligence is a branch of computer science "
                      "that enables machines to perform tasks requiring human intelligence."
    ),

    Document(
        page_content="Machine Learning is a subset of Artificial Intelligence "
                      "that allows computers to learn patterns from data."
    ),

    Document(
        page_content="Natural Language Processing, or NLP, enables computers "
                      "to understand, interpret and process human language."
    ),

    Document(
        page_content="Deep Learning uses artificial neural networks with multiple "
                      "layers to learn complex patterns from large amounts of data."
    ),

    Document(
        page_content="Retrieval Augmented Generation, or RAG, combines document "
                      "retrieval with language generation to provide answers using relevant information."
    ),

    Document(
        page_content="FAISS is a library used for efficient similarity search "
                      "and retrieval of vector embeddings."
    )
]


# -------------------------------------------------
# 2. CREATE EMBEDDINGS
# -------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------------------------
# 3. CREATE FAISS VECTOR DATABASE
# -------------------------------------------------

print("Creating vector database...")

vector_db = FAISS.from_documents(
    documents,
    embeddings
)

print("Vector database created successfully!")


# -------------------------------------------------
# 4. LOAD ANSWER GENERATION MODEL
# -------------------------------------------------

print("Loading chatbot model...")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("Chatbot model loaded successfully!")


# -------------------------------------------------
# 5. CHATBOT FUNCTION
# -------------------------------------------------

def chatbot(question):

    # Retrieve top 2 relevant documents
    retrieved_docs = vector_db.similarity_search(
        question,
        k=2
    )

    # Combine retrieved information
    context = "\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Create prompt
    prompt = f"""
You are a domain-specific chatbot for Artificial Intelligence and
Natural Language Processing.

Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    result = generator(
        prompt,
        max_new_tokens=80
    )

    answer = result[0]["generated_text"]

    return answer, retrieved_docs


# -------------------------------------------------
# 6. START CHATBOT
# -------------------------------------------------

print("\n======================================")
print("   AI & NLP DOMAIN-SPECIFIC CHATBOT")
print("======================================")

print("Ask questions about AI, ML, NLP, Deep Learning, RAG or FAISS.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    answer, retrieved_docs = chatbot(question)

    print("\nChatbot:", answer)

    print("\nRetrieved Documents:")

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc.page_content}")

    print("\n" + "-" * 60)

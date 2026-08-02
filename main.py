# ==========================================
# IMPORTS
# ==========================================

from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# STEP 1: READ TEXT FILE
# ==========================================

file_path = "data/ml_notes.txt"

with open(file_path, "r", encoding="utf-8") as file:
    full_text = file.read()


# ==========================================
# STEP 2: CHUNKING FUNCTION
# ==========================================

def create_chunks(text, chunk_size=500, overlap=100):

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ==========================================
# STEP 3: CREATE CHUNKS
# ==========================================

chunks = create_chunks(full_text)

print(f"\nTotal Chunks Created: {len(chunks)}")


# ==========================================
# STEP 4: LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding model loaded successfully!")


# ==========================================
# STEP 5: CREATE EMBEDDINGS
# ==========================================

print("\nCreating embeddings...")

embeddings = embedding_model.encode(chunks)

print("Embeddings created successfully!")
print(f"Embedding Shape: {embeddings.shape}")


# ==========================================
# STEP 6: CREATE FAISS INDEX
# ==========================================

embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(embeddings))

print("\nFAISS index created successfully!")
print(f"Total vectors stored: {index.ntotal}")


# ==========================================
# STEP 7: SETUP GROQ CLIENT
# ==========================================

print("\nSetting up Groq client...")

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )

client = Groq(api_key=groq_api_key)

print("Groq client ready!")


# ==========================================
# STEP 8: USER QUERY
# ==========================================

query = input("\nEnter your question: ")


# ==========================================
# STEP 9: CONVERT QUERY TO EMBEDDING
# ==========================================

query_embedding = embedding_model.encode([query])


# ==========================================
# STEP 10: SEARCH RELEVANT CHUNKS
# ==========================================

top_k = 3

distances, indices = index.search(
    np.array(query_embedding),
    top_k
)


# ==========================================
# STEP 11: RETRIEVE CONTEXT
# ==========================================

retrieved_chunks = []

for idx in indices[0]:
    retrieved_chunks.append(chunks[idx])

context = "\n\n".join(retrieved_chunks)


# ==========================================
# STEP 12: CREATE PROMPT
# ==========================================

prompt = f"""
Use the context below to answer the question.

Only use information from the context.
If the answer is not available in the context, say:
"I don't have enough information to answer this."

Context:
{context}

Question:
{query}

Answer:
"""


# ==========================================
# STEP 13: GENERATE RESPONSE VIA GROQ
# ==========================================

print("\nGenerating answer...")

chat_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions using only the provided context."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ==========================================
# STEP 14: DISPLAY ANSWER
# ==========================================

print("\n========== AI ANSWER ==========\n")

print(chat_response.choices[0].message.content)
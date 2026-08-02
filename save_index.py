# ==========================================
# save_index.py
# Run this ONCE to build and save the index
# ==========================================

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# ==========================================
# STEP 1: READ TEXT FILE
# ==========================================

with open("data/ml_notes.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# ==========================================
# STEP 2: CHUNKING
# ==========================================

def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = create_chunks(full_text)
print(f"Total Chunks: {len(chunks)}")

# ==========================================
# STEP 3: CREATE EMBEDDINGS
# ==========================================

print("Creating embeddings...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(chunks)
print(f"Embeddings shape: {embeddings.shape}")

# ==========================================
# STEP 4: CREATE AND SAVE FAISS INDEX
# ==========================================

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

os.makedirs("saved_index", exist_ok=True)

# Save FAISS index
faiss.write_index(index, "saved_index/index.faiss")

# Save chunks
with open("saved_index/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("\nIndex saved successfully!")
print("Files saved in saved_index/ folder:")
print("  - index.faiss")
print("  - chunks.pkl")
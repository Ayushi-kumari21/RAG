
# ==========================================
# IMPORTS
# ==========================================

import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import numpy as np
import fitz  # pymupdf
import pytesseract
from PIL import Image
import io
from audio_recorder_streamlit import audio_recorder
import os
# Tell pytesseract where tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # <-- your key
    return embedding_model, client

embedding_model, client = load_models()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        page_text = page.get_text()
        if len(page_text.strip()) > 50:
            text += page_text
        else:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text
    return text


def create_chunks(text, chunk_size=300, overlap=75):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def generate_pdf_summary(text, filename, client):
    sample_text = text[:6000]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a document summarizer. Given text from a document, provide a comprehensive summary covering: 1) Main topic, 2) Key subtopics covered, 3) Important concepts, 4) Conclusions or findings. Be thorough but concise."
            },
            {
                "role": "user",
                "content": f"Summarize this document called '{filename}':\n\n{sample_text}"
            }
        ]
    )
    return response.choices[0].message.content


def is_broad_question(query, client):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """Classify if the question is BROAD or SPECIFIC.
BROAD questions ask about: topics covered, summary, overview, comparison between documents, common themes, what the document is about.
SPECIFIC questions ask about: specific facts, definitions, methods, numbers, names, explanations of specific concepts.
Reply with only one word: BROAD or SPECIFIC."""
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    result = response.choices[0].message.content.strip().upper()
    return "BROAD" in result


def transcribe_audio(audio_bytes, client):
    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("audio.wav", audio_bytes, "audio/wav"),
        )
        return transcription.text
    except Exception as e:
        return f"Error: {str(e)}"


def expand_query(user_question, client):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a search query optimizer. Rewrite the user's question into 3 different search queries that would help find relevant information in a document. Return only the 3 queries, one per line, nothing else."
            },
            {
                "role": "user",
                "content": f"Original question: {user_question}"
            }
        ]
    )
    expanded = response.choices[0].message.content.strip()
    queries = [user_question] + expanded.split("\n")[:3]
    return queries


def search_index(query, index, chunks, sources, embedding_model, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx],
            "source": sources[idx],
            "distance": distances[0][i]
        })
    return results


def multi_query_search(queries, index, chunks, sources, embedding_model):
    all_results = []
    seen_chunks = set()
    for query in queries:
        results = search_index(query, index, chunks, sources, embedding_model)
        for r in results:
            if r["chunk"] not in seen_chunks:
                seen_chunks.add(r["chunk"])
                all_results.append(r)
    all_results.sort(key=lambda x: x["distance"])
    return all_results[:6]


def generate_answer(query, client):
    history = []
    for msg in st.session_state.messages[:-1]:
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    broad = is_broad_question(query, client)

    if broad:
        st.caption("Using document summaries...")
        summary_context = ""
        for filename, summary in st.session_state.pdf_summaries.items():
            summary_context += f"\n\n=== {filename} ===\n{summary}"

        prompt = f"""Use the document summaries below to answer the question.
Be thorough and mention specific details from each document where relevant.

Document Summaries:
{summary_context}

Question:
{query}

Answer:"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about documents."
                },
                *history,
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content
        st.write(answer)
        st.caption(f"Sources: {', '.join(st.session_state.pdf_summaries.keys())}")

        with st.expander(" View document summaries used"):
            for filename, summary in st.session_state.pdf_summaries.items():
                st.markdown(f"**{filename}**")
                st.write(summary)
                st.divider()

    else:
        st.caption("🔍 Searching specific chunks...")
        queries = expand_query(query, client)
        results = multi_query_search(
            queries,
            st.session_state.index,
            st.session_state.chunks,
            st.session_state.chunk_sources,
            embedding_model
        )

        context = "\n\n".join([r["chunk"] for r in results])
        unique_sources = list(set([r["source"] for r in results]))

        prompt = f"""Use the context below to answer the question.
Only use information from the context. If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question:
{query}

Answer:"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions using only the provided context. Be detailed and thorough."
                },
                *history,
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content
        st.write(answer)
        st.caption(f"📄 Sources: {', '.join(unique_sources)}")

        with st.expander("🔍 View retrieved context"):
            for i, r in enumerate(results):
                st.markdown(f"**Chunk {i+1}** — `{r['source']}`")
                st.text(r["chunk"])
                st.divider()

    return answer


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_sources" not in st.session_state:
    st.session_state.chunk_sources = []

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

if "pdf_summaries" not in st.session_state:
    st.session_state.pdf_summaries = {}

if "audio_transcribed" not in st.session_state:
    st.session_state.audio_transcribed = ""


# ==========================================
# SIDEBAR - PDF UPLOAD
# ==========================================

with st.sidebar:
    st.header(" Upload Your PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        for f in uploaded_files:
            st.write(f"{f.name}")

    process_btn = st.button("Process PDFs", type="primary")

    if process_btn and uploaded_files:
        with st.spinner("Reading and indexing PDFs..."):
            all_chunks = []
            all_sources = []
            pdf_summaries = {}

            for pdf_file in uploaded_files:
                st.write(f"Processing: {pdf_file.name}")
                text = extract_text_from_pdf(pdf_file)

                if len(text.strip()) < 50:
                    st.warning(f" Could not extract text from {pdf_file.name}")
                    continue

                st.write(f" Summarizing: {pdf_file.name}")
                summary = generate_pdf_summary(text, pdf_file.name, client)
                pdf_summaries[pdf_file.name] = summary

                chunks = create_chunks(text)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_sources.append(pdf_file.name)

                st.write(f"{pdf_file.name} — {len(chunks)} chunks")

            if all_chunks:
                embeddings = embedding_model.encode(
                    all_chunks,
                    show_progress_bar=True
                )
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings))

                st.session_state.chunks = all_chunks
                st.session_state.chunk_sources = all_sources
                st.session_state.index = index
                st.session_state.pdf_summaries = pdf_summaries
                st.session_state.pdfs_processed = True
                st.session_state.messages = []

                st.success("All PDFs processed!")
                st.write(f"Total chunks: {len(all_chunks)}")

                st.divider()
                st.subheader("PDF Summaries")
                for filename, summary in pdf_summaries.items():
                    with st.expander(f"{filename}"):
                        st.write(summary)
            else:
                st.error("No text could be extracted from any PDF.")

    elif process_btn and not uploaded_files:
        st.warning("Please upload at least one PDF first.")

    st.divider()
    if st.session_state.pdfs_processed:
        st.success("Index ready")
        st.write(f"Chunks in memory: {len(st.session_state.chunks)}")
        st.write(f"PDFs summarized: {len(st.session_state.pdf_summaries)}")
    else:
        st.info("Upload PDFs and click Process to begin")


# ==========================================
# WELCOME MESSAGE
# ==========================================

st.title(" RAG Assistant")

if not st.session_state.pdfs_processed:
    st.markdown("""
     **Hello! I'm your RAG Assistant.**

    I can help you understand any document instantly.

    **Here's how to get started:**

    1. Upload your PDFs using the sidebar on the left
    2. Click **Process PDFs** to index them
    3. Type or speak your question below

    **What I can do:**
    - Summarize entire documents
    - Answer specific questions
    - Compare multiple documents
    - Tell you exactly which document the answer came from
    """)

else:
    # Show quick suggestions once PDFs are loaded
    st.markdown("**Try asking:**")
    col1, col2 = st.columns(2)
    with col1:
        st.info("What are the main topics in these documents?")
        st.info("What is [specific concept]?")
    with col2:
        st.info("Compare the two documents")
        st.info(" Summarize the key findings")


# ==========================================
# CHAT HISTORY
# ==========================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ==========================================
# AUDIO INPUT
# ==========================================

st.divider()
st.markdown("**Or speak your question:**")

audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e74c3c",
    neutral_color="#3498db",
    icon_size="2x"
)

if audio_bytes and len(audio_bytes) > 1000:
    with st.spinner("Transcribing..."):
        transcribed = transcribe_audio(audio_bytes, client)
    if transcribed and not transcribed.startswith("Error"):
        st.success(f"You said: **{transcribed}**")
        st.session_state.audio_transcribed = transcribed


# ==========================================
# TEXT INPUT
# ==========================================

query = st.chat_input("Ask a question about your PDFs...")

if st.session_state.audio_transcribed and not query:
    query = st.session_state.audio_transcribed
    st.session_state.audio_transcribed = ""

if query:

    if not st.session_state.pdfs_processed:
        st.warning("Please upload and process PDFs first using the sidebar.")

    else:

        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_answer(query, client)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

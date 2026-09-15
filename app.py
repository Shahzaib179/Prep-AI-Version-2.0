import hashlib
import re
import shutil
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(
    page_title="Prep AI V2",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 700
DEFAULT_OVERLAP = 120
DEFAULT_TOP_K = 8
GROQ_MODEL = "llama-3.3-70b-versatile"

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]


# -----------------------------
# Cached models/resources
# -----------------------------
@st.cache_resource
def load_embedding_model():
    """Load Sentence Transformers once per app process."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def get_groq_client():
    """Create the Groq client from Streamlit secrets."""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml."
        )
    return Groq(api_key=api_key)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_path: str, filename: str) -> list[dict]:
    """Extract PDF text page-by-page."""
    reader = PdfReader(file_path)
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            records.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": page_number,
                }
            )

    return records


def extract_docx(file_path: str, filename: str) -> list[dict]:
    """Extract DOCX paragraphs. DOCX text does not reliably expose page numbers."""
    document = Document(file_path)
    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs)

    if not text:
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ]


def extract_txt(file_path: str, filename: str) -> list[dict]:
    """Extract a plain-text file."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ]


def extract_md(file_path: str, filename: str) -> list[dict]:
    """Extract Markdown as readable text."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None,
        }
    ]


def extract_document(file_path: str, filename: str) -> list[dict]:
    """Route a supported document to its extraction function."""
    extension = Path(filename).suffix.lower()

    extractors = {
        ".pdf": extract_pdf,
        ".docx": extract_docx,
        ".txt": extract_txt,
        ".md": extract_md,
    }

    if extension not in extractors:
        raise ValueError(f"Unsupported file type: {extension}")

    return extractors[extension](file_path, filename)


# -----------------------------
# Chunking
# -----------------------------
def chunk_text_records(
    records: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Create overlapping word-based chunks.

    Metadata is copied to every chunk, so filename/page remain available
    during search and source display.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []

    for record in records:
        words = record["text"].split()

        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "filename": record["filename"],
                        "page": record.get("page"),
                    }
                )

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_vector_index(chunks: list[dict]):
    """Embed all chunks once and build a FAISS cosine-similarity index."""
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def fingerprint_chunks(chunks: list[dict]) -> str:
    """Create a stable fingerprint so unchanged documents are not re-embedded."""
    digest = hashlib.sha256()

    for chunk in chunks:
        digest.update(chunk["filename"].encode("utf-8"))
        digest.update(str(chunk.get("page")).encode("utf-8"))
        digest.update(chunk["text"].encode("utf-8"))

    return digest.hexdigest()


# -----------------------------
# Keyword search
# -----------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "with", "from", "by", "is", "are", "was", "were", "be", "as",
    "at", "it", "this", "that", "these", "those", "what", "which",
    "who", "how", "why", "when", "where", "about", "explain",
}


def important_words(text: str) -> list[str]:
    """Return simple keyword candidates from the user's topic."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if len(word) > 2 and word not in STOPWORDS]


def keyword_scores(query: str, chunks: list[dict]) -> np.ndarray:
    """Score each chunk by simple keyword overlap."""
    query_words = set(important_words(query))
    scores = np.zeros(len(chunks), dtype="float32")

    if not query_words:
        return scores

    for i, chunk in enumerate(chunks):
        chunk_words = set(important_words(chunk["text"]))
        overlap = query_words.intersection(chunk_words)
        scores[i] = len(overlap) / len(query_words)

    return scores


# -----------------------------
# Hybrid semantic + keyword search
# -----------------------------
def hybrid_search(
    query: str,
    chunks: list[dict],
    index,
    top_k: int = DEFAULT_TOP_K,
    semantic_weight: float = 0.75,
) -> list[dict]:
    """
    Combine FAISS semantic similarity and keyword overlap.

    Both scores are normalized before the weighted combination.
    """
    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    semantic_k = min(len(chunks), max(top_k * 3, top_k))
    semantic_scores, semantic_indices = index.search(
        query_embedding,
        semantic_k,
    )

    semantic = np.zeros(len(chunks), dtype="float32")
    for score, idx in zip(semantic_scores[0], semantic_indices[0]):
        if idx >= 0:
            semantic[idx] = score

    # Convert cosine similarity from roughly [-1, 1] to [0, 1].
    semantic = (semantic + 1.0) / 2.0

    keywords = keyword_scores(query, chunks)

    if semantic.max() > semantic.min():
        semantic_norm = (semantic - semantic.min()) / (
            semantic.max() - semantic.min()
        )
    else:
        semantic_norm = semantic

    if keywords.max() > keywords.min():
        keyword_norm = (keywords - keywords.min()) / (
            keywords.max() - keywords.min()
        )
    else:
        keyword_norm = keywords

    final_scores = (
        semantic_weight * semantic_norm
        + (1.0 - semantic_weight) * keyword_norm
    )

    ranked_indices = np.argsort(final_scores)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        result = dict(chunks[idx])
        result["semantic_score"] = float(semantic_norm[idx])
        result["keyword_score"] = float(keyword_norm[idx])
        result["hybrid_score"] = float(final_scores[idx])
        results.append(result)

    return results


# -----------------------------
# Source formatting
# -----------------------------
def format_source_label(source: dict) -> str:
    page = source.get("page")
    page_text = f"Page {page}" if page else "Page not available"
    return f"{source['filename']} — {page_text}"


def build_context(results: list[dict]) -> str:
    """Build a clearly separated context block for the LLM."""
    blocks = []

    for i, result in enumerate(results, start=1):
        page = result.get("page")
        page_text = str(page) if page else "N/A"

        blocks.append(
            f"[SOURCE {i}]\n"
            f"Filename: {result['filename']}\n"
            f"Page: {page_text}\n"
            f"Text:\n{result['text']}"
        )

    return "\n\n".join(blocks)


# -----------------------------
# Groq generation
# -----------------------------
def ask_groq(
    user_request: str,
    retrieved_chunks: list[dict],
    mode: str,
) -> str:
    """Ask Groq to answer strictly from retrieved context."""
    client = get_groq_client()
    context = build_context(retrieved_chunks)

    if mode == "MCQ":
        system_prompt = """
You are Prep AI, an MDCAT-level exam preparation assistant.

You MUST use only the supplied CONTEXT. Do not use outside knowledge.
If the context does not contain enough information, say so clearly and
do not invent facts.

Create high-quality MDCAT-style multiple choice questions.
Use only facts, relationships, definitions, mechanisms, examples, and
details that are explicitly supported by the context.

Respect the requested difficulty level:
- Easy: direct recall and straightforward understanding.
- Medium: conceptual understanding, comparison, application, and interpretation.
- Hard: multi-step reasoning, closely related concepts, and challenging distractors.
- Mixed MDCAT Level: naturally mix easy, medium, and hard questions.

For each question:
- Give 4 options: A, B, C, D.
- Give the correct answer.
- Give a one-sentence explanation based only on the context.
- Include a source number such as [SOURCE 2] when possible.

Return the maximum number requested if the context supports it.
Avoid duplicate questions and avoid questions that require information
outside the context.
"""
    else:
        system_prompt = """
You are Prep AI, an exam preparation assistant.

Answer ONLY from the supplied CONTEXT.
Do not use outside knowledge, even if you know the answer.
If the requested information is not present or cannot be supported by
the context, say: "This information is not available in the provided
documents."

For a normal answer, be clear, concise, and student-friendly.
When useful, cite the source number such as [SOURCE 1].
"""

    prompt = f"""
STUDENT REQUEST:
{user_request}

CONTEXT:
{context}
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=8000,
    )

    return completion.choices[0].message.content


# -----------------------------
# Google Drive loading
# -----------------------------
def download_drive_source(drive_url: str) -> tuple[Path, bool]:
    """
    Download a public/shared Google Drive file or folder.

    gdown is used because it supports Google Drive sharing URLs.
    The user must have made the file/folder accessible to the app.
    """
    drive_url = drive_url.strip()

    if not drive_url:
        raise ValueError("Please paste a Google Drive link.")

    temp_dir = Path(tempfile.mkdtemp(prefix="prep_ai_drive_"))

    is_folder = "/folders/" in drive_url

    if is_folder:
        output_dir = temp_dir / "drive_folder"
        output_dir.mkdir(parents=True, exist_ok=True)

        gdown.download_folder(
            url=drive_url,
            output=str(output_dir),
            quiet=True,
            use_cookies=False,
        )
        return output_dir, True

    # Let gdown keep the original Drive filename. We temporarily change the
    # working directory so the downloaded file stays inside our temp folder.
    import os

    previous_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        downloaded = gdown.download(
            url=drive_url,
            output=None,
            quiet=True,
        )
    finally:
        os.chdir(previous_cwd)

    if not downloaded:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            "Could not download the Google Drive file. "
            "Make sure the sharing link is accessible."
        )

    downloaded_path = Path(downloaded)
    if not downloaded_path.is_absolute():
        downloaded_path = temp_dir / downloaded_path

    return downloaded_path, False


def collect_supported_files(source_path: Path, is_folder: bool) -> list[Path]:
    """Find supported documents in a downloaded Drive file/folder."""
    if not is_folder:
        return (
            [source_path]
            if source_path.suffix.lower() in SUPPORTED_EXTENSIONS
            else []
        )

    files = [
        path
        for path in source_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


# -----------------------------
# Processing pipeline
# -----------------------------
def process_files(file_paths: list[tuple[Path, str]]) -> tuple[list[dict], list[dict]]:
    """Extract and chunk all files using the same pipeline."""
    all_records = []
    extraction_summary = []

    for file_path, display_filename in file_paths:
        records = extract_document(str(file_path), display_filename)
        all_records.extend(records)

        extraction_summary.append(
            {
                "filename": display_filename,
                "pages_or_sections": len(records),
                "characters": sum(len(r["text"]) for r in records),
            }
        )

    chunks = chunk_text_records(all_records)

    return extraction_summary, chunks


def reset_index():
    for key in [
        "chunks",
        "faiss_index",
        "embeddings",
        "document_fingerprint",
        "documents",
        "last_results",
        "last_answer",
    ]:
        st.session_state.pop(key, None)


# -----------------------------
# Session state
# -----------------------------
for key, default in {
    "chunks": [],
    "faiss_index": None,
    "embeddings": None,
    "document_fingerprint": None,
    "documents": [],
    "last_results": [],
    "last_answer": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------
# UI
# -----------------------------
st.title("📚 Prep AI V2")
st.caption(
    "Advanced RAG for exam preparation — local documents + Google Drive + "
    "hybrid semantic/keyword search + Groq."
)

with st.sidebar:
    st.header("⚙️ RAG Settings")

    chunk_size = st.slider(
        "Chunk size (words)",
        min_value=300,
        max_value=1200,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
    )

    overlap = st.slider(
        "Chunk overlap (words)",
        min_value=0,
        max_value=300,
        value=DEFAULT_OVERLAP,
        step=20,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=3,
        max_value=15,
        value=DEFAULT_TOP_K,
    )

    semantic_weight = st.slider(
        "Semantic search weight",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
    )

    st.divider()
    st.info(
        "Embeddings are created only when the document set changes. "
        "The Sentence Transformer model is also cached."
    )

# Document sources
st.header("1. Add study material")

local_files = st.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

drive_url = st.text_input(
    "Google Drive file or folder link",
    placeholder="Paste a shared Google Drive link here",
)

col1, col2 = st.columns(2)

with col1:
    process_button = st.button(
        "🚀 Process Documents",
        type="primary",
        use_container_width=True,
    )

with col2:
    clear_button = st.button(
        "🧹 Clear Documents",
        use_container_width=True,
    )

if clear_button:
    reset_index()
    st.rerun()

if process_button:
    temp_upload_dir = Path(tempfile.mkdtemp(prefix="prep_ai_uploads_"))
    files_to_process = []

    # Save local uploads to temporary files.
    for uploaded_file in local_files or []:
        destination = temp_upload_dir / uploaded_file.name
        destination.write_bytes(uploaded_file.getvalue())
        files_to_process.append((destination, uploaded_file.name))

    # Download Drive source if supplied.
    if drive_url.strip():
        try:
            drive_path, is_folder = download_drive_source(drive_url)
            drive_files = collect_supported_files(drive_path, is_folder)

            if not drive_files:
                st.warning(
                    "No supported PDF, DOCX, TXT or MD files were found in "
                    "the Google Drive source."
                )

            for path in drive_files:
                files_to_process.append((path, path.name))

        except Exception as exc:
            st.error(f"Google Drive error: {exc}")

    if not files_to_process:
        st.warning("Add at least one local document or Google Drive source.")
    else:
        try:
            with st.spinner("Extracting text and creating chunks..."):
                summary, chunks = process_files(files_to_process)

            if not chunks:
                st.error("No readable text was found in the supplied documents.")
            else:
                new_fingerprint = fingerprint_chunks(chunks)

                if (
                    st.session_state.document_fingerprint == new_fingerprint
                    and st.session_state.faiss_index is not None
                ):
                    st.info("These documents are already processed. Reusing embeddings.")
                else:
                    with st.spinner(
                        "Creating Sentence Transformers embeddings and FAISS index..."
                    ):
                        index, embeddings = build_vector_index(chunks)

                    st.session_state.chunks = chunks
                    st.session_state.faiss_index = index
                    st.session_state.embeddings = embeddings
                    st.session_state.document_fingerprint = new_fingerprint
                    st.session_state.documents = summary
                    st.session_state.last_results = []
                    st.session_state.last_answer = ""

                st.success(
                    f"Processed {len(files_to_process)} document(s) and created "
                    f"{len(chunks)} chunks."
                )

        except Exception as exc:
            st.error(f"Document processing error: {exc}")

# Document information
if st.session_state.documents:
    st.header("2. Extracted document information")

    for doc in st.session_state.documents:
        st.write(
            f"**{doc['filename']}** — "
            f"{doc['characters']:,} characters, "
            f"{doc['pages_or_sections']} extracted page/section unit(s)"
        )

    st.metric("Created chunks", len(st.session_state.chunks))

# Study controls
if st.session_state.faiss_index is not None:
    st.header("3. Study with Prep AI")

    st.write(
        "The fields below help the retriever focus on the chapter/topic "
        "already available in your uploaded material."
    )

    col1, col2 = st.columns(2)

    with col1:
        chapter = st.text_input(
            "Chapter name (optional)",
            placeholder="e.g. Cell Biology",
        )

        topic = st.text_input(
            "Topic / concept",
            placeholder="e.g. Mitochondria and cellular respiration",
        )

    with col2:
        mode = st.selectbox(
            "Generation mode",
            ["MCQ", "Answer / Explanation"],
        )

        difficulty = st.selectbox(
            "Difficulty level",
            ["Easy", "Medium", "Hard", "Mixed MDCAT Level"],
            index=3,
            disabled=(mode != "MCQ"),
        )

        if mode == "MCQ":
            mcq_count = st.slider(
                "Maximum MCQs",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
            )
        else:
            mcq_count = 20

    extra_request = st.text_area(
        "Optional instruction",
        placeholder=(
            "Example: Focus on definitions, mechanisms and common MDCAT traps."
        ),
    )

    generate_button = st.button(
        "🧠 Generate",
        type="primary",
        use_container_width=True,
    )

    if generate_button:
        if not topic.strip() and not chapter.strip():
            st.warning("Enter at least a chapter name or topic.")
        else:
            retrieval_query_parts = []

            if chapter.strip():
                retrieval_query_parts.append(f"Chapter: {chapter.strip()}")

            if topic.strip():
                retrieval_query_parts.append(f"Topic: {topic.strip()}")

            if extra_request.strip():
                retrieval_query_parts.append(extra_request.strip())

            retrieval_query = "\n".join(retrieval_query_parts)

            with st.spinner("Searching your documents..."):
                results = hybrid_search(
                    query=retrieval_query,
                    chunks=st.session_state.chunks,
                    index=st.session_state.faiss_index,
                    top_k=top_k,
                    semantic_weight=semantic_weight,
                )

            if not results:
                st.warning("No relevant document chunks were found.")
            else:
                if mode == "MCQ":
                    user_request = (
                        f"Chapter: {chapter or 'Not specified'}\n"
                        f"Topic: {topic or 'Not specified'}\n"
                        f"Difficulty level: {difficulty}\n"
                        f"Create up to {mcq_count} MDCAT-level MCQs with answer key "
                        "from the provided context only.\n"
                        f"Additional instruction: {extra_request or 'None'}"
                    )
                else:
                    user_request = (
                        f"Chapter: {chapter or 'Not specified'}\n"
                        f"Topic: {topic or 'Not specified'}\n"
                        f"Student request: {extra_request or 'Explain the topic clearly.'}"
                    )

                with st.spinner("Generating answer from retrieved context..."):
                    try:
                        answer = ask_groq(
                            user_request=user_request,
                            retrieved_chunks=results,
                            mode=mode,
                        )

                        st.session_state.last_results = results
                        st.session_state.last_answer = answer
                    except Exception as exc:
                        st.error(f"Groq error: {exc}")

# Answer + retrieved sources
if st.session_state.last_answer:
    st.header("4. Prep AI Answer")
    st.markdown(st.session_state.last_answer)

    st.header("5. Retrieved Sources")
    st.caption(
        "These are the chunks retrieved by the hybrid semantic + keyword search."
    )

    for i, source in enumerate(st.session_state.last_results, start=1):
        with st.expander(
            f"Source {i}: {format_source_label(source)} "
            f"(hybrid score: {source['hybrid_score']:.3f})"
        ):
            st.write(
                f"**Filename:** {source['filename']}\n\n"
                f"**Page:** {source.get('page') or 'Not available'}\n\n"
                f"**Semantic score:** {source['semantic_score']:.3f}\n\n"
                f"**Keyword score:** {source['keyword_score']:.3f}"
            )
            st.text_area(
                "Retrieved text",
                value=source["text"],
                height=180,
                key=f"source_text_{i}_{source['hybrid_score']}",
            )

st.divider()
st.caption(
    "Prep AI V2 uses Sentence Transformers + FAISS for retrieval and Groq "
    "for context-grounded generation."
)

import os
import re
import io
import json
import hashlib
import random
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Prep AI App V2",
    page_icon="📚",
    layout="wide",
)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes: bytes, filename: str):
    records = []
    reader = PdfReader(io.BytesIO(file_bytes))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            records.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": page_number,
                }
            )
    return records


def extract_docx(file_bytes: bytes, filename: str):
    document = Document(io.BytesIO(file_bytes))
    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs).strip()
    return (
        [{"text": text, "filename": filename, "page": None}]
        if text
        else []
    )


def extract_txt(file_bytes: bytes, filename: str):
    text = file_bytes.decode("utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", text).strip()
    return (
        [{"text": text, "filename": filename, "page": None}]
        if text
        else []
    )


def extract_md(file_bytes: bytes, filename: str):
    text = file_bytes.decode("utf-8", errors="ignore")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`>-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return (
        [{"text": text, "filename": filename, "page": None}]
        if text
        else []
    )


def extract_document(file_bytes: bytes, filename: str):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    raise ValueError(f"Unsupported file type: {extension}")


# -----------------------------
# Chunking
# -----------------------------
def chunk_text_records(records, chunk_size=900, overlap=150):
    chunks = []

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be >= 0 and smaller than chunk size.")

    for record in records:
        words = record["text"].split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
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
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_vector_index(chunks):
    model = load_embedding_model()
    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def fingerprint_chunks(chunks):
    payload = json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


# -----------------------------
# Hybrid retrieval
# -----------------------------
def important_words(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "is", "are", "was", "were", "with", "from", "by", "as", "at",
        "what", "which", "who", "how", "why", "when", "where", "that",
        "this", "these", "those", "chapter", "topic",
    }
    return {word for word in words if len(word) > 2 and word not in stopwords}


def keyword_scores(query, chunks):
    query_words = important_words(query)
    scores = []

    for chunk in chunks:
        chunk_words = important_words(chunk["text"])
        if not query_words:
            score = 0.0
        else:
            score = len(query_words & chunk_words) / len(query_words)
        scores.append(score)

    return np.array(scores, dtype="float32")


def hybrid_search(query, chunks, index, top_k=8, semantic_weight=0.7):
    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    semantic_scores, semantic_indices = index.search(
        query_embedding, min(top_k * 3, len(chunks))
    )

    semantic_scores = semantic_scores[0]
    semantic_indices = semantic_indices[0]

    keyword = keyword_scores(query, chunks)

    candidates = []
    for semantic_score, idx in zip(semantic_scores, semantic_indices):
        if idx < 0:
            continue

        combined = (
            semantic_weight * float(semantic_score)
            + (1 - semantic_weight) * float(keyword[idx])
        )
        candidates.append((combined, float(semantic_score), float(keyword[idx]), int(idx)))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[:top_k]


def format_source_label(chunk):
    if chunk.get("page"):
        return f"{chunk['filename']} — page {chunk['page']}"
    return chunk["filename"]


def build_context(chunks, results):
    context_parts = []

    for source_number, (_, _, _, idx) in enumerate(results, start=1):
        chunk = chunks[idx]
        context_parts.append(
            f"[SOURCE {source_number}]\n"
            f"File: {chunk['filename']}\n"
            f"Page: {chunk.get('page') or 'N/A'}\n"
            f"Text: {chunk['text']}"
        )

    return "\n\n".join(context_parts)


# -----------------------------
# Groq
# -----------------------------
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", None)

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to Streamlit secrets or environment variables."
        )

    return Groq(api_key=api_key)


def ask_groq(topic, context, mode, difficulty="Mixed MDCAT Level", count=10, instruction=""):
    client = get_groq_client()

    if mode == "MCQ":
        system_prompt = f"""
You are an expert MDCAT exam question writer.

Create up to {count} high-quality single-best-answer MCQs using ONLY the supplied
SOURCE CONTEXT. Do not use outside knowledge. If the context does not support
enough questions, generate fewer questions rather than inventing information.

Difficulty: {difficulty}
- Easy: direct recall and straightforward understanding.
- Medium: conceptual understanding, comparison, application, or interpretation.
- Hard: multi-step reasoning and challenging but fair distractors.
- Mixed MDCAT Level: a mixture of easy, medium, and hard questions.

Each question must have exactly four options: A, B, C, D.
There must be exactly one correct answer.

Return ONLY valid JSON in this exact structure:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "correct_answer": "A",
      "explanation": "Short explanation based only on the source context.",
      "source": 1
    }}
  ]
}}

Rules:
- The correct_answer must be one of A, B, C, D.
- source must be the SOURCE number that best supports the question.
- Do not reveal the answer anywhere except correct_answer.
- Keep questions at MDCAT level.
- Avoid duplicate questions.
- Do not invent facts outside the context.
"""
    else:
        system_prompt = """
You are an expert study assistant.
Answer the student's request ONLY from the supplied SOURCE CONTEXT.
If the answer is not available in the context, clearly say that it is not
available in the provided material.
"""

    user_prompt = (
        f"Student topic/chapter: {topic}\n"
        f"Difficulty level: {difficulty}\n"
        f"Additional instruction: {instruction or 'None'}\n\n"
        f"SOURCE CONTEXT:\n{context}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2 if mode == "MCQ" else 0.1,
        max_tokens=8000,
    )

    return response.choices[0].message.content.strip()


def parse_mcq_json(raw_text):
    text = raw_text.strip()

    # Handle accidental Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # If extra text surrounds JSON, extract the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    data = json.loads(text)
    questions = data.get("questions", [])

    if not isinstance(questions, list):
        raise ValueError("Groq returned an invalid questions list.")

    cleaned = []

    for i, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        options = item.get("options", {})
        correct = str(item.get("correct_answer", "")).strip().upper()
        explanation = str(item.get("explanation", "")).strip()
        source = item.get("source")

        if not question:
            continue

        if not isinstance(options, dict):
            continue

        normalized_options = {}
        for letter in ["A", "B", "C", "D"]:
            value = str(options.get(letter, "")).strip()
            if not value:
                break
            normalized_options[letter] = value

        if len(normalized_options) != 4:
            continue

        if correct not in normalized_options:
            continue

        try:
            source_number = int(source)
        except (TypeError, ValueError):
            source_number = None

        cleaned.append(
            {
                "id": f"q{i}",
                "question": question,
                "options": normalized_options,
                "correct_answer": correct,
                "explanation": explanation,
                "source": source_number,
            }
        )

    if not cleaned:
        raise ValueError("No valid MCQs were returned by Groq.")

    return cleaned


# -----------------------------
# File processing
# -----------------------------
def process_files(file_items, chunk_size, overlap):
    all_records = []
    document_info = []

    for item in file_items:
        filename = item["name"]
        file_bytes = item["bytes"]

        records = extract_document(file_bytes, filename)
        all_records.extend(records)

        total_chars = sum(len(record["text"]) for record in records)
        pages = sorted(
            {
                record.get("page")
                for record in records
                if record.get("page") is not None
            }
        )

        document_info.append(
            {
                "filename": filename,
                "characters": total_chars,
                "pages": len(pages) if pages else None,
            }
        )

    chunks = chunk_text_records(
        all_records,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return all_records, chunks, document_info


def load_drive_file(drive_url):
    downloaded = gdown.download(
        url=drive_url,
        output=None,
        quiet=True,
    )

    if not downloaded:
        raise ValueError(
            "Google Drive download failed. Make sure the file is shared correctly."
        )

    path = Path(downloaded)
    return {
        "name": path.name,
        "bytes": path.read_bytes(),
    }


# -----------------------------
# Session state
# -----------------------------
defaults = {
    "chunks": [],
    "index": None,
    "embeddings": None,
    "fingerprint": None,
    "document_info": [],
    "last_results": [],
    "last_context": "",
    "last_answer": "",
    "quiz": [],
    "quiz_submitted": False,
    "quiz_answers": {},
    "quiz_score": None,
    "quiz_nonce": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# UI
# -----------------------------
st.title("📚 Prep AI App V2")
st.caption("RAG-based exam preparation with semantic search, keyword search, and interactive MCQ quizzes.")

with st.sidebar:
    st.header("Retrieval Settings")

    chunk_size = st.slider(
        "Chunk size (words)",
        min_value=300,
        max_value=1800,
        value=900,
        step=100,
    )

    overlap = st.slider(
        "Chunk overlap (words)",
        min_value=0,
        max_value=400,
        value=150,
        step=25,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=15,
        value=8,
    )

    semantic_weight = st.slider(
        "Semantic search weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
    )

    st.divider()
    st.info(
        "For Groq, add GROQ_API_KEY to .streamlit/secrets.toml or your environment."
    )


st.subheader("1. Add study material")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, TXT, or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

drive_url = st.text_input(
    "Optional Google Drive file link",
    placeholder="Paste a shareable Google Drive file link here",
)

file_items = []

if uploaded_files:
    for uploaded in uploaded_files:
        file_items.append(
            {
                "name": uploaded.name,
                "bytes": uploaded.getvalue(),
            }
        )

if drive_url.strip():
    try:
        with st.spinner("Downloading Google Drive file..."):
            drive_item = load_drive_file(drive_url.strip())

        extension = Path(drive_item["name"]).suffix.lower()
        if extension not in {".pdf", ".docx", ".txt", ".md"}:
            st.warning(
                f"Google Drive file type '{extension}' is not supported. "
                "Use PDF, DOCX, TXT, or MD."
            )
        else:
            file_items.append(drive_item)

    except Exception as exc:
        st.error(f"Google Drive error: {exc}")


if file_items:
    if st.button("Process documents", type="primary"):
        try:
            with st.spinner("Extracting text, chunking, and building FAISS index..."):
                records, chunks, document_info = process_files(
                    file_items,
                    chunk_size,
                    overlap,
                )

                if not chunks:
                    raise ValueError("No text could be extracted from the supplied files.")

                index, embeddings = build_vector_index(chunks)
                fp = fingerprint_chunks(chunks)

                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["embeddings"] = embeddings
                st.session_state["fingerprint"] = fp
                st.session_state["document_info"] = document_info

                # A new document set means a new quiz.
                st.session_state["quiz"] = []
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_score"] = None

            st.success(
                f"Processed {len(document_info)} document(s) and created "
                f"{len(chunks)} chunks."
            )

        except Exception as exc:
            st.error(f"Processing error: {exc}")


if st.session_state["document_info"]:
    st.subheader("2. Extracted document information")

    for info in st.session_state["document_info"]:
        page_text = (
            f"{info['pages']} page(s)"
            if info["pages"] is not None
            else "page number not available"
        )
        st.write(
            f"**{info['filename']}** — "
            f"{info['characters']:,} extracted characters — {page_text}"
        )

    st.caption(
        f"Total chunks available for retrieval: {len(st.session_state['chunks'])}"
    )


st.subheader("3. Study / Quiz")

topic = st.text_input(
    "Chapter / topic",
    placeholder="Example: Cell membrane, Genetics, Chemical bonding...",
)

mode = st.selectbox(
    "Mode",
    ["MCQ", "Answer / Explanation"],
)

difficulty = st.selectbox(
    "Difficulty level",
    ["Easy", "Medium", "Hard", "Mixed MDCAT Level"],
    index=3,
    disabled=(mode != "MCQ"),
)

mcq_count = st.number_input(
    "Number of MCQs",
    min_value=1,
    max_value=100,
    value=20,
    step=1,
    disabled=(mode != "MCQ"),
)

instruction = st.text_area(
    "Optional instruction",
    placeholder="Example: Focus on conceptual questions and avoid calculations.",
)

generate_label = "Generate Quiz" if mode == "MCQ" else "Get Answer"

if st.button(generate_label, type="primary"):
    if not st.session_state["chunks"] or st.session_state["index"] is None:
        st.warning("Please process at least one study document first.")
    elif not topic.strip():
        st.warning("Please enter a chapter or topic.")
    else:
        try:
            with st.spinner("Searching your material and generating content..."):
                results = hybrid_search(
                    topic,
                    st.session_state["chunks"],
                    st.session_state["index"],
                    top_k=top_k,
                    semantic_weight=semantic_weight,
                )

                context = build_context(
                    st.session_state["chunks"],
                    results,
                )

                if mode == "MCQ":
                    # Add a fresh nonce to encourage a different quiz on retakes.
                    st.session_state["quiz_nonce"] += 1
                    varied_instruction = (
                        f"{instruction}\n"
                        f"Create a fresh quiz variation #{st.session_state['quiz_nonce']}. "
                        "Avoid repeating obvious question wording from previous generations."
                    )

                    raw_answer = ask_groq(
                        topic=topic,
                        context=context,
                        mode=mode,
                        difficulty=difficulty,
                        count=int(mcq_count),
                        instruction=varied_instruction,
                    )

                    quiz = parse_mcq_json(raw_answer)

                    # Keep the quiz limited to the requested count.
                    quiz = quiz[: int(mcq_count)]

                    st.session_state["quiz"] = quiz
                    st.session_state["quiz_submitted"] = False
                    st.session_state["quiz_answers"] = {}
                    st.session_state["quiz_score"] = None
                    st.session_state["last_results"] = results
                    st.session_state["last_context"] = context
                    st.session_state["last_answer"] = ""

                else:
                    answer = ask_groq(
                        topic=topic,
                        context=context,
                        mode=mode,
                        difficulty=difficulty,
                        count=1,
                        instruction=instruction,
                    )

                    st.session_state["quiz"] = []
                    st.session_state["quiz_submitted"] = False
                    st.session_state["quiz_answers"] = {}
                    st.session_state["quiz_score"] = None
                    st.session_state["last_results"] = results
                    st.session_state["last_context"] = context
                    st.session_state["last_answer"] = answer

        except Exception as exc:
            st.error(f"Generation error: {exc}")


# -----------------------------
# Interactive quiz
# -----------------------------
if st.session_state["quiz"]:
    st.divider()
    st.subheader("📝 Interactive MCQ Quiz")

    quiz = st.session_state["quiz"]

    st.write(
        f"**{len(quiz)} question(s)** — select one answer for each question, "
        "then click **Submit Quiz**."
    )

    if not st.session_state["quiz_submitted"]:
        with st.form("mcq_quiz_form"):
            for number, question in enumerate(quiz, start=1):
                st.markdown(f"### Question {number}")
                st.write(question["question"])

                options = question["options"]
                selected = st.radio(
                    "Choose an answer:",
                    options=["Not attempted", "A", "B", "C", "D"],
                    format_func=lambda value, opts=options: (
                        "Not attempted"
                        if value == "Not attempted"
                        else f"{value}. {opts[value]}"
                    ),
                    index=0,
                    key=f"quiz_answer_{question['id']}_{st.session_state['quiz_nonce']}",
                )

                st.session_state["quiz_answers"][question["id"]] = selected

            submitted = st.form_submit_button(
                "Submit Quiz",
                type="primary",
            )

        if submitted:
            answers = st.session_state["quiz_answers"]
            correct = sum(
                1
                for question in quiz
                if answers.get(question["id"]) == question["correct_answer"]
            )

            attempted = sum(
                1
                for question in quiz
                if answers.get(question["id"]) in {"A", "B", "C", "D"}
            )

            total = len(quiz)
            percentage = (correct / total * 100) if total else 0.0

            st.session_state["quiz_score"] = {
                "correct": correct,
                "wrong": attempted - correct,
                "unattempted": total - attempted,
                "attempted": attempted,
                "total": total,
                "percentage": percentage,
            }
            st.session_state["quiz_submitted"] = True
            st.rerun()

    else:
        score = st.session_state["quiz_score"]

        st.success("Quiz submitted!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score", f"{score['correct']} / {score['total']}")
        col2.metric("Percentage", f"{score['percentage']:.1f}%")
        col3.metric("Wrong", score["wrong"])
        col4.metric("Unattempted", score["unattempted"])

        if score["percentage"] >= 80:
            st.success("Excellent performance!")
        elif score["percentage"] >= 60:
            st.info("Good performance. Review the incorrect questions below.")
        else:
            st.warning("Keep practicing. Review the explanations and sources below.")

        st.divider()
        st.subheader("Quiz Review")

        for number, question in enumerate(quiz, start=1):
            user_answer = st.session_state["quiz_answers"].get(
                question["id"], "Not attempted"
            )
            correct_answer = question["correct_answer"]

            if user_answer == correct_answer:
                status = "✅ Correct"
            elif user_answer == "Not attempted":
                status = "⚪ Unattempted"
            else:
                status = "❌ Incorrect"

            st.markdown(f"### {number}. {status}")
            st.write(question["question"])

            for letter, option_text in question["options"].items():
                marker = ""
                if letter == correct_answer:
                    marker = " — **Correct answer**"
                elif letter == user_answer:
                    marker = " — **Your answer**"

                st.write(f"**{letter}.** {option_text}{marker}")

            st.write(
                f"**Your answer:** "
                f"{user_answer if user_answer != 'Not attempted' else 'Not attempted'}"
            )
            st.write(f"**Correct answer:** {correct_answer}")

            if question["explanation"]:
                st.info(f"**Explanation:** {question['explanation']}")

            source_number = question.get("source")
            results = st.session_state["last_results"]
            chunks = st.session_state["chunks"]

            if (
                isinstance(source_number, int)
                and 1 <= source_number <= len(results)
            ):
                _, semantic_score, keyword_score, idx = results[source_number - 1]
                chunk = chunks[idx]

                st.markdown("**Source:**")
                st.caption(format_source_label(chunk))
                st.write(chunk["text"])

            st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("🔄 Generate New Quiz"):
                st.session_state["quiz_nonce"] += 1
                st.session_state["quiz"] = []
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_score"] = None
                st.rerun()

        with col_b:
            if st.button("📖 Hide Quiz / Start Again"):
                st.session_state["quiz"] = []
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_score"] = None
                st.rerun()


# -----------------------------
# Normal answer mode
# -----------------------------
if st.session_state["last_answer"]:
    st.divider()
    st.subheader("Answer")
    st.markdown(st.session_state["last_answer"])


# -----------------------------
# Retrieved sources
# -----------------------------
if st.session_state["last_results"] and not st.session_state["quiz"]:
    st.divider()
    st.subheader("Retrieved RAG Sources")

    for number, (_, semantic_score, keyword_score, idx) in enumerate(
        st.session_state["last_results"],
        start=1,
    ):
        chunk = st.session_state["chunks"][idx]

        with st.expander(
            f"Source {number}: {format_source_label(chunk)}"
        ):
            st.write(chunk["text"])
            st.caption(
                f"Semantic score: {semantic_score:.3f} | "
                f"Keyword score: {keyword_score:.3f}"
            )

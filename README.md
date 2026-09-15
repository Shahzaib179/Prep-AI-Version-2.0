# Prep AI V2

Prep AI V2 is a simple Streamlit-based RAG study assistant for exam preparation.

It accepts:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Google Drive file links
- Google Drive folder links

The app extracts document text, creates overlapping chunks, embeds the chunks with Sentence Transformers, stores them in a FAISS index, and uses hybrid semantic + keyword search before sending the retrieved context to Groq.

## Main RAG pipeline

```text
Student documents
      |
      v
Text extraction
      |
      v
Overlapping chunks + filename/page metadata
      |
      v
Sentence Transformers embeddings
      |
      v
FAISS vector index
      |
      +----------------------+
      |                      |
      v                      v
Semantic search        Keyword search
      |                      |
      +----------+-----------+
                 |
                 v
          Hybrid ranking
                 |
                 v
       Top relevant chunks
                 |
                 v
              Groq
                 |
                 v
       Answer / MDCAT MCQs
                 |
                 v
        Retrieved sources
```

## Project files

```text
prep_ai_v2/
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install Python

Python 3.10+ is recommended.

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure the Groq API key

The API key is deliberately NOT stored in `app.py`.

Create this file:

```text
.streamlit/secrets.toml
```

Put:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Do not commit `secrets.toml` to Git.

For Streamlit Community Cloud, add `GROQ_API_KEY` in the app's Secrets settings.

## 5. Run the app

```bash
streamlit run app.py
```

## How the document pipeline works

### PDF

`extract_pdf()` uses `pypdf` and extracts each PDF page separately.

Every extracted record keeps:

```python
{
    "text": "...",
    "filename": "biology.pdf",
    "page": 4
}
```

### DOCX

`extract_docx()` uses `python-docx`.

DOCX files do not reliably expose printed page numbers through normal document parsing, so their page metadata is `None`.

### TXT

`extract_txt()` reads the complete text file and keeps the filename.

### MD

`extract_md()` reads the Markdown file as text and keeps the filename.

## Chunking

`chunk_text_records()` splits extracted text into overlapping word chunks.

Default:

- Chunk size: 700 words
- Overlap: 120 words

Every chunk keeps:

```python
{
    "text": "...",
    "filename": "biology.pdf",
    "page": 4
}
```

The UI displays the total number of chunks created.

## Embeddings

Sentence Transformers uses:

```text
all-MiniLM-L6-v2
```

The model is loaded with Streamlit `st.cache_resource`.

Document chunks are embedded once when the document set changes.

The app creates a SHA-256 fingerprint of the chunk content and metadata. If the same documents are processed again, the existing FAISS index and embeddings in Streamlit session state are reused.

This prevents embeddings from being recreated for every question.

## FAISS

FAISS stores normalized chunk embeddings in an `IndexFlatIP` index.

Because the vectors are normalized, inner product is equivalent to cosine similarity for the semantic search.

## Hybrid search

`hybrid_search()` combines:

1. Sentence Transformer semantic similarity.
2. Simple keyword overlap.

The sidebar lets you control the semantic-search weight.

Default:

```text
75% semantic
25% keyword
```

The final ranking keeps the original chunk metadata.

## Groq

Groq receives:

- Chapter
- Topic
- Optional student instruction
- Retrieved document chunks

The system prompt tells the model to answer ONLY from the supplied context.

If the information is not present, the model is instructed to say:

```text
This information is not available in the provided documents.
```

For MCQ mode, the model is instructed to produce:

- MDCAT-style questions
- Four options: A, B, C, D
- Correct answer
- Short explanation
- Source number where possible

## Google Drive

Paste a Google Drive sharing link into the app.

The app uses `gdown` to download public/shared Google Drive files and folders.

Supported downloaded files:

```text
.pdf
.docx
.txt
.md
```

For a Google Drive folder, supported files inside the folder are sent through exactly the same pipeline as local uploads:

```text
Drive
  -> extraction
  -> chunking
  -> embeddings
  -> FAISS
  -> hybrid search
  -> Groq
```

### Important Google Drive requirement

The Drive file or folder must be accessible to the account/service making the download request. A private Drive link that requires an authenticated Google session will not automatically work with this simple public-link implementation.

For production use with private Google Drive files, replace the `gdown` section with the Google Drive API and OAuth/service-account authentication.

## MDCAT question generation

Choose:

```text
Generation mode -> MCQ
```

Then enter:

- Chapter name
- Topic/concept
- Optional instruction
- Maximum number of MCQs

The app retrieves the most relevant chunks first, then asks Groq to generate as many questions as the requested maximum that can be supported by the retrieved context.

This is intentionally context-grounded so the model does not invent facts from outside the student's documents.

## Source display

After generation, the app displays the retrieved sources below the answer.

Each source shows:

- Filename
- Page number when available
- Semantic score
- Keyword score
- Hybrid score
- Full retrieved text chunk

This makes the RAG pipeline easier for students and developers to inspect.

## Why Streamlit session state and caching are used

Streamlit reruns the Python script when users interact with widgets.

The app uses:

### `st.session_state`

Stores the current:

- Chunks
- FAISS index
- Embeddings
- Document fingerprint
- Document summary
- Last answer
- Last retrieved sources

Therefore, asking another question does not rebuild the document index.

### `st.cache_resource`

Caches:

- Sentence Transformer model
- Groq client

This avoids repeatedly loading expensive resources.

## Security

Never put your Groq key directly in `app.py`.

Correct:

```python
api_key = st.secrets.get("GROQ_API_KEY")
```

and:

```text
.streamlit/secrets.toml
```

Incorrect:

```python
GROQ_API_KEY = "gsk_..."
```

Also add `.streamlit/secrets.toml` to `.gitignore`:

```gitignore
.streamlit/secrets.toml
.venv/
__pycache__/
```

## Simple architecture

Everything is intentionally kept in one `app.py` so it is easy to teach and explain.

The important functions are:

```text
extract_pdf()
extract_docx()
extract_txt()
extract_md()

extract_document()

chunk_text_records()

build_vector_index()

keyword_scores()

hybrid_search()

ask_groq()

download_drive_source()

process_files()
```

## Suggested future V3 improvements

Once this version is stable, the next version could add:

- Persistent FAISS indexes on disk
- SQLite/Chroma/Qdrant document storage
- Google Drive OAuth for private files
- Better PDF table extraction
- DOCX heading/page-aware chunking
- Reranking with a cross-encoder
- Query expansion
- Multi-query retrieval
- Metadata filters
- Conversation memory
- MCQ difficulty controls
- MCQ validation and duplicate detection
- Subject-specific MDCAT templates
- Question history
- Export to PDF/DOCX
- Student performance analytics
- Flashcards
- Spaced repetition

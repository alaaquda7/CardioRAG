# CardioRAG 🫀
### AI-Powered Cardiology Research Assistant

A Retrieval-Augmented Generation (RAG) system that answers medical questions from peer-reviewed cardiology research — with automatic citations, zero hallucination.

---

## Why CardioRAG?

Most RAG systems are built on general-purpose components. CardioRAG is different — every component was chosen for a specific reason:

**BioBERT over general-purpose embeddings**
General models treat "HFrEF" and "heart failure with reduced ejection fraction" as unrelated.
BioBERT was trained on PubMed — it understands clinical terminology natively.

**CrossEncoder Reranker after retrieval**
Cosine similarity retrieves similar vectors, not necessarily relevant answers.
The CrossEncoder reads the question and chunk together — scoring true relevance.
Result: retrieve 10, rerank to best 3.

**Query Expansion (3 phrasings)**
Medical concepts have multiple terminologies.
One query phrasing misses content phrased differently in the paper.
Expanding to 3 phrasings maximizes retrieval coverage.

**Strict citation enforcement**
Every answer must reference Source N or say "not available in the provided documents."
This eliminates hallucination by design — not by hope.

**chunk_size=600 instead of default 1000**
Smaller chunks preserve sentence integrity and produce more precise page-level citations.

Every decision was made intentionally — not copied from a tutorial.

---

# Data

This project uses 25 peer-reviewed cardiology papers.

## Sources

### ESC Guidelines (5 papers)
- Heart Failure Guidelines 2023 — escardio.org
- Acute Coronary Syndromes 2023 — escardio.org
- Hypertension Guidelines 2024 — escardio.org
- Atrial Fibrillation Guidelines 2024 — escardio.org
- Cardiovascular & Diabetes 2023 — escardio.org

### PubMed Central (12 papers)
- Search: pubmed.ncbi.nlm.nih.gov/pmc
- Filter: Free Full Text + 2021-2024
- Topics: Troponin, Heart Failure, MI, Cardiac Imaging

### ArXiv (8 papers)
- Search: arxiv.org
- Topics: AI in ECG, Deep Learning Cardiology, NLP Clinical Notes


---


## The Challenge: Why Medical RAG is Hard

Building RAG on medical text is not the same as building it on general documents:

**1. Terminology Gap**
A general embedding model treats "HFrEF" and "heart failure" as unrelated.
BioBERT was trained on PubMed abstracts — it understands clinical synonyms natively.

**2. Query-Document Mismatch**
A doctor asks: *"What biomarkers confirm MI?"*
The paper says: *"Elevated Troponin I levels are diagnostic of myocardial infarction."*
Query Expansion bridges this gap by generating alternative phrasings before retrieval.

**3. Chunk Boundary Problem**
RecursiveCharacterTextSplitter can cut a sentence mid-way.
Tuning chunk_size=600 with chunk_overlap=120 preserves sentence integrity
while keeping chunks small enough for precise citations.

**4. Retrieval ≠ Relevance**
Cosine similarity finds similar vectors — not necessarily the most relevant answer.
The CrossEncoder reads the question and chunk together, scoring true relevance.

---

## Pipeline

```
PDF Files (25 papers)
        ↓
pypdf → extract text per page
        ↓
RecursiveCharacterTextSplitter
chunk_size=600, chunk_overlap=120
        ↓
BioBERT Embeddings
(pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb)
        ↓
ChromaDB → stored on Google Drive (persistent)
        ↓
User Question
        ↓
Query Expansion → 3 alternative phrasings via LLaMA-3
        ↓
Retrieval → top 10 chunks per query (deduplicated)
        ↓
CrossEncoder Reranking → top 3 most relevant chunks
        ↓
LLaMA-3.1-8b via Groq → answer generation
        ↓
Answer + Source File + Page Number
```

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF Loading | `pypdf` |
| Text Splitting | `RecursiveCharacterTextSplitter` (LangChain) |
| Embeddings | `BioBERT` via HuggingFace |
| Vector Store | `ChromaDB` (persisted to Google Drive) |
| Reranker | `CrossEncoder` — cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | `LLaMA-3.1-8b-instant` via Groq API |
| Framework | `LangChain` (Official Docs pattern) |
| Environment | `Google Colab` — T4 GPU |

---

## Data Sources

**25 peer-reviewed documents** across 3 categories:

| Source | Count | Content |
|---|---|---|
| 🏛️ ESC Guidelines | 5 | Heart Failure 2023, ACS 2023, Hypertension 2024, AFib 2024, Diabetes & CVD 2023 |
| 🔬 PubMed Central | 12 | Troponin studies, MI management, Cardiac biomarkers, Cardiac imaging |
| ⚡ ArXiv | 8 | AI in ECG, Deep Learning for cardiac diagnosis, NLP in clinical notes |

---

## Configuration

```python
CONFIG = {
    "chunk_size": 600,          # Tuned down from LangChain default (1000) for medical precision
    "chunk_overlap": 120,       # 20% overlap preserves sentence continuity
    "embedding_model": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "llm_model": "llama-3.1-8b-instant",
    "top_k_retrieval": 10,      # Cast wide net first
    "top_k_reranked": 3,        # Rerank to precision
}
```

---

## Getting Started

### 1. Open in Google Colab
```
Runtime → Change runtime type → T4 GPU
```

### 2. Add Groq API Key
```
Left panel → 🔑 Secrets → Add new secret
Name:  GROQ_API_KEY
Value: your_key_from_console.groq.com
Notebook access: ON
```

### 3. Run cells in order (1 → 11)

### 4. Ask a question
```python
ask("What are the diagnostic criteria for acute heart failure?")
```

---

## Example Output

```
Question: What are the diagnostic criteria for acute heart failure?
============================================================
Queries generated: 4
Unique chunks retrieved: 14
After reranking: top 3 chunks selected

Answer:
The diagnostic criteria for acute heart failure involve a multifaceted
approach integrating clinical evaluation, laboratory testing, and imaging.

Clinical assessment includes evaluation of:
- Dyspnea and orthopnea [Source 1]
- Fatigue and reduced exercise tolerance [Source 2]
- Peripheral edema [Source 3]

ESC guidelines recommend BNP and NT-proBNP as key biomarkers
to confirm diagnosis and evaluate severity [Source 1].

Sources:
============================================================
  🏛️ [1] esc_heart_failure_2023.pdf — Page 23
  🔬 [2] pub_cardiac_imaging_heart_failure_2025.pdf — Page 5
  🏛️ [3] esc_cardiovascular_diabetes_2023.pdf — Page 88
```

---

## Evaluation

Self-evaluation pipeline using LLM-as-judge (no external libraries):

| Criterion | Score |
|---|---|
| Accuracy | 5/5 |
| Completeness | 4/5 |
| Overall | 4.3/5 |
| Verdict | PASS ✅ |

---

## Roadmap

- [x] PDF ingestion and chunking
- [x] BioBERT medical embeddings
- [x] ChromaDB persistent vector store
- [x] CrossEncoder reranking
- [x] Query expansion
- [x] Automatic citations (filename + page)
- [x] Google Drive persistence
- [x] Self-evaluation pipeline
- [ ] Streamlit web interface
- [ ] Cloud vector store (Pinecone / Qdrant)
- [ ] Support for 100+ documents

---

## References

Built following [LangChain Official RAG Documentation](https://docs.langchain.com/oss/python/langchain/retrieval) as the architectural base, with domain-specific enhancements for medical text.

---

## Author

**Alaa ALQudah**  
AI & Robotics Engineer

[![LinkedIn]([https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/alaa-qudah-2444ba344?utm_source=share_via&utm_content=profile&utm_medium=member_ios))
[![GitHub](https://github.com/alaaquda7)

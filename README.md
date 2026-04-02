# 🏛️ LLM Agentic Legal Information Retrieval

**Kaggle Competition:** [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)

> Given an English legal question, retrieve the most relevant Swiss legal citations (statutes + court decisions) from a German-language corpus. Evaluated on **Macro F1** at citation level.

---

## 🗂️ Repo Structure

```
kaggle-swiss-legal-retrieval/
├── notebooks/
│   └── baseline_bm25.ipynb       # Kaggle-ready notebook (Phase 1)
├── solutions/
│   └── baseline_bm25.py          # Same logic as .py for reference
├── README.md
└── .gitignore
```

---

## 🔍 Task Overview

| | Details |
|---|---|
| **Input** | English legal questions |
| **Output** | Semicolon-separated Swiss legal citation strings |
| **Corpus** | `laws_de.csv` (statutes) + `court_considerations.csv` (court decisions, ~2.4GB) |
| **Metric** | Macro F1 (citation-level precision & recall, averaged per query) |
| **Constraint** | Kaggle notebook must run **offline** in ≤ 12 hours |

---

## 🚀 Approach Roadmap

### Phase 1 — BM25 Baseline ✅
- Keyword search (BM25Okapi) over laws corpus
- No GPU needed, runs in minutes
- Establishes a leaderboard baseline

### Phase 2 — Multilingual Embeddings (planned)
- Translate English queries → German using `Helsinki-NLP/opus-mt-en-de` (offline)
- Encode all docs with `intfloat/multilingual-e5-small`
- FAISS approximate nearest-neighbor index for fast retrieval
- Add court decisions corpus

### Phase 3 — Hybrid + Reranking (planned)
- Combine BM25 + embedding scores
- Cross-encoder reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384`)
- Dynamic TOP_K per query

### Phase 4 — Agentic Retrieval (planned)
- Query expansion with small LLM (Mistral 7B quantized)
- Iterative retrieval: retrieve → verify → expand
- Citation-graph traversal

---

## 📊 Results

| Phase | Val Macro F1 | Notes |
|---|---|---|
| BM25 Baseline | TBD | Laws corpus only, TOP_K=5 |
| Multilingual Embeddings | TBD | - |

---

## 🛠️ Local Setup (for dev/testing)

```bash
pip install rank_bm25 pandas numpy
python solutions/baseline_bm25.py
```

Data files not included (download from Kaggle competition page).

---

## 📎 References

- [LEXam Dataset](https://huggingface.co/datasets/LEXam-Benchmark/LEXam) — training queries source
- [Competition Starter Repo](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)
- BM25: Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*

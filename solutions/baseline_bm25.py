"""
=================================================================
Phase 1 Baseline: BM25 Retrieval for Swiss Legal Citation Task
=================================================================

Competition: LLM Agentic Legal Information Retrieval (Kaggle)
https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval

Approach:
- BM25Okapi keyword retrieval over laws_de.csv corpus
- No GPU, no embeddings — just fast keyword matching
- Cross-lingual limitation: queries are English, corpus is German
  → Phase 2 will fix this with multilingual embeddings

Expected val Macro F1: low (cross-lingual mismatch) but it gets us
on the leaderboard to iterate from.
=================================================================
"""

# ── DEPENDENCIES ─────────────────────────────────────────────
# pip install rank_bm25 pandas numpy
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re
import os

# ── CONFIG ───────────────────────────────────────────────────
DATA_DIR = "/kaggle/input/llm-agentic-legal-information-retrieval"
OUTPUT_PATH = "/kaggle/working/submission.csv"

# How many citations to return per query.
# Controls precision/recall tradeoff:
#   Too low  → low recall  (miss correct citations)
#   Too high → low precision (include wrong citations)
TOP_K = 5

# Whether to include the large court decisions file (~2.4GB).
# Set to True once the BM25 baseline is running — adds many more valid citations.
USE_COURT = False


# ── LOAD DATA ────────────────────────────────────────────────
print("Loading test queries...")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"  → {len(test_df)} test queries")

print("Loading laws corpus...")
laws_df = pd.read_csv(f"{DATA_DIR}/laws_de.csv")
print(f"  → {len(laws_df)} law snippets")

if USE_COURT:
    print("Loading court considerations (big file, be patient)...")
    court_df = pd.read_csv(f"{DATA_DIR}/court_considerations.csv")
    print(f"  → {len(court_df)} court decisions")
    corpus_df = pd.concat([laws_df, court_df], ignore_index=True)
else:
    corpus_df = laws_df

print(f"Total corpus: {len(corpus_df):,} documents\n")


# ── TOKENIZER ────────────────────────────────────────────────
def simple_tokenize(text: str) -> list[str]:
    """
    Lowercase + regex tokenizer.

    Keeps:
    - Latin letters (a-z)
    - German umlauts (ä, ö, ü, ß) — important for German legal text
    - Digits
    - Dots and slashes — critical for citation IDs like 'BGE 139 I 2 E. 3.1'

    Phase 2 note: replace this with a proper multilingual tokenizer
    (e.g. from transformers) or first translate queries to German.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = re.findall(r"[a-z0-9äöüß./]+", text)
    return tokens


# ── BUILD BM25 INDEX ─────────────────────────────────────────
# Concatenate citation string + text so citation keywords also influence scoring.
# Example: "Art. 11 Abs. 2 OR  Wer …" helps if query mentions "OR" (Obligationenrecht).
print("Building BM25 index (this takes a minute)...")
corpus_texts = corpus_df["citation"].fillna("") + " " + corpus_df["text"].fillna("")
tokenized_corpus = [simple_tokenize(doc) for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 index ready!\n")


# ── RETRIEVE ─────────────────────────────────────────────────
print(f"Retrieving top-{TOP_K} citations per query...")
predictions = []

for _, row in test_df.iterrows():
    query_id = row["query_id"]
    query    = row["query"]

    query_tokens = simple_tokenize(query)

    if not query_tokens:
        # Edge case: empty or unparseable query
        predictions.append({"query_id": query_id, "predicted_citations": ""})
        continue

    # Score every document in the corpus
    scores = bm25.get_scores(query_tokens)  # shape: (len(corpus_df),)

    # Sort descending, take top-K
    top_k_idx = np.argsort(scores)[::-1][:TOP_K]

    # Drop documents with zero BM25 score (no keyword overlap at all)
    top_k_idx = [i for i in top_k_idx if scores[i] > 0]

    top_citations = corpus_df.iloc[top_k_idx]["citation"].tolist()

    predictions.append({
        "query_id": query_id,
        "predicted_citations": ";".join(top_citations)
    })


# ── WRITE SUBMISSION ─────────────────────────────────────────
submission_df = pd.DataFrame(predictions)
submission_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Submission saved → {OUTPUT_PATH}")
print(f"   Rows: {len(submission_df)}\n")
print(submission_df.head(5).to_string())


# ── LOCAL VALIDATION (val.csv) ───────────────────────────────
val_path = f"{DATA_DIR}/val.csv"
if os.path.exists(val_path):
    print("\n" + "="*50)
    print("Evaluating on val.csv (10 English queries)...")
    val_df = pd.read_csv(val_path)

    val_preds = []
    for _, row in val_df.iterrows():
        tokens = simple_tokenize(row["query"])
        if not tokens:
            val_preds.append(set())
            continue
        scores  = bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:TOP_K]
        top_idx = [i for i in top_idx if scores[i] > 0]
        val_preds.append(set(corpus_df.iloc[top_idx]["citation"].tolist()))

    f1_scores = []
    for i, row in val_df.iterrows():
        gold = set(str(row["gold_citations"]).split(";"))
        pred = val_preds[i]

        if not gold and not pred:
            f1_scores.append(1.0)
        elif not gold or not pred:
            f1_scores.append(0.0)
        else:
            tp        = len(gold & pred)
            precision = tp / len(pred) if pred else 0.0
            recall    = tp / len(gold) if gold else 0.0
            denom     = precision + recall
            f1_scores.append(2 * precision * recall / denom if denom > 0 else 0.0)

    print(f"Val Macro F1 : {np.mean(f1_scores):.4f}")
    print(f"Per-query F1 : {[round(s, 3) for s in f1_scores]}")
    print("="*50)

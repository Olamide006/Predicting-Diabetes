# build_rag.py
# Reads papers.json, builds TF-IDF vectors, stores in FAISS index
# Run this ONCE — it saves the index to disk

import json
import pickle
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

PAPERS_FILE  = "papers.json"
INDEX_FILE   = "rag_index.faiss"
META_FILE    = "rag_metadata.pkl"

def load_papers():
    with open(PAPERS_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")
    return papers

def build_documents(papers):
    """Combine title + abstract into one searchable document per paper."""
    docs = []
    metadata = []
    for p in papers:
        text = f"{p['title']}. {p['abstract']}"
        docs.append(text)
        metadata.append({
            "pmid":   p["pmid"],
            "title":  p["title"],
            "year":   p["year"],
            "term":   p["search_term"],
            "abstract": p["abstract"]
        })
    return docs, metadata

def build_index(docs):
    """Convert documents to TF-IDF vectors and store in FAISS."""
    print("Building TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)   # include bigrams like "blood glucose"
    )
    tfidf_matrix = vectorizer.fit_transform(docs)

    # Convert sparse TF-IDF to dense float32 (FAISS requirement)
    dense = tfidf_matrix.toarray().astype(np.float32)

    # Normalise so cosine similarity = dot product (faster search)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1
    dense = dense / norms

    dim = dense.shape[1]
    print(f"Vector dimensions: {dim}")

    # Build FAISS flat index (exact search, fine for 173 docs)
    index = faiss.IndexFlatIP(dim)  # IP = inner product = cosine similarity
    index.add(dense)
    print(f"FAISS index built with {index.ntotal} vectors")

    return index, vectorizer

def save(index, vectorizer, metadata):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "metadata": metadata}, f)
    print(f"Saved index  → {INDEX_FILE}")
    print(f"Saved metadata → {META_FILE}")

def main():
    papers            = load_papers()
    docs, metadata    = build_documents(papers)
    index, vectorizer = build_index(docs)
    save(index, vectorizer, metadata)
    print("\nAll done! RAG index is ready.")

if __name__ == "__main__":
    main()
# build_rag.py
# Reads PDF papers from papers/ subfolders, builds TF-IDF FAISS index

import os
import pickle
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import pymupdf  # pip install pymupdf

INDEX_FILE = "rag_index.faiss"
META_FILE  = "rag_metadata.pkl"
PAPERS_DIR = "papers"

FEATURE_FOLDERS = [
    ""
    "age",
    "bmi",
    "sex",
    "family_history",
    "gestational_diabetes",
    "physical_activity",
    "hypertension",
]

def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF file."""
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  ⚠️  Could not read {pdf_path}: {e}")
        return ""

def load_papers_from_pdfs():
    """Walk through papers/ subfolders and extract text from each PDF."""
    docs     = []
    metadata = []
    total    = 0

    for folder in FEATURE_FOLDERS:
        # folder_path = os.path.join(PAPERS_DIR, folder)
        folder_path = os.path.join(PAPERS_DIR, folder) if folder else PAPERS_DIR
        if not os.path.exists(folder_path):
            print(f"⚠️  Folder not found: {folder_path}")
            continue

        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        print(f"\n📁 {folder}/ — {len(pdf_files)} PDFs found")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            text     = extract_text_from_pdf(pdf_path)

            if not text or len(text) < 100:
                print(f"  ⚠️  Skipping (too short): {pdf_file}")
                continue

            # Use first 300 chars as title preview
            title_preview = text[:200].replace("\n", " ").strip()

            docs.append(text)
            metadata.append({
                "title":    pdf_file.replace(".pdf", ""),
                "abstract": text[:1000],  # first 1000 chars as abstract
                "full_text": text,
                "feature":  folder,
                "source":   pdf_path,
                "year":     "2015-2025",
                "pmid":     pdf_file.replace(".pdf", ""),
            })
            total += 1
            print(f"  ✅ {pdf_file[:60]}")

    print(f"\n✅ Total papers loaded: {total}")
    return docs, metadata

def build_index(docs):
    """Convert documents to TF-IDF vectors and store in FAISS."""
    print("\nBuilding TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(docs)

    dense = tfidf_matrix.toarray().astype(np.float32)

    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1
    dense = dense / norms

    dim = dense.shape[1]
    print(f"Vector dimensions: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(dense)
    print(f"FAISS index built with {index.ntotal} vectors")

    return index, vectorizer

def save(index, vectorizer, metadata):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "metadata": metadata}, f)
    print(f"\n✅ Saved index    → {INDEX_FILE}")
    print(f"✅ Saved metadata → {META_FILE}")

def main():
    docs, metadata    = load_papers_from_pdfs()
    if not docs:
        print("❌ No documents found! Check your papers/ folder.")
        return
    index, vectorizer = build_index(docs)
    save(index, vectorizer, metadata)
    print("\n🎉 RAG index built from your papers and ready!")

if __name__ == "__main__":
    main()
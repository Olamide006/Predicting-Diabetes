# fetch_papers.py  (updated - with retry and save-as-you-go)

from Bio import Entrez
import json
import time
import os

Entrez.email = "your_email@example.com"  # keep your email here

SEARCH_TERMS = [
    "BMI diabetes risk African",
    "family history type 2 diabetes risk",
    "physical activity diabetes prevention",
    "hypertension diabetes risk factor",
    "gestational diabetes future risk",
    "age diabetes onset risk",
    "prediabetes risk factors",
    "diabetes prevention lifestyle",
]

OUTPUT_FILE = "papers.json"

def load_existing():
    """Load papers already saved so we don't re-fetch them."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} existing papers from {OUTPUT_FILE}")
        return papers, {p["pmid"] for p in papers}
    return [], set()

def save(papers):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

def fetch_with_retry(term, max_results=25, retries=3):
    """Fetch papers, retrying up to 3 times on network error."""
    for attempt in range(1, retries + 1):
        try:
            handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results,
                                     datetype="pdat", mindate="2015", maxdate="2025")
            record = Entrez.read(handle)
            handle.close()
            ids = record["IdList"]

            if not ids:
                return []

            time.sleep(0.5)

            handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            papers = []
            for article in records["PubmedArticle"]:
                try:
                    title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
                    abstract_parts = article["MedlineCitation"]["Article"].get("Abstract", {})
                    abstract = str(abstract_parts.get("AbstractText", [""])[0])
                    year = str(article["MedlineCitation"]["Article"]["Journal"]
                               ["JournalIssue"]["PubDate"].get("Year", "unknown"))
                    pmid = str(article["MedlineCitation"]["PMID"])

                    if len(abstract) > 100:
                        papers.append({
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract,
                            "year": year,
                            "search_term": term
                        })
                except Exception:
                    continue

            return papers

        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < retries:
                wait = 5 * attempt
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"  Skipping '{term}' after {retries} failed attempts.")
                return []

def main():
    all_papers, seen_pmids = load_existing()
    done_terms = {p["search_term"] for p in all_papers}

    for term in SEARCH_TERMS:
        if term in done_terms:
            print(f"Skipping (already done): '{term}'")
            continue

        print(f"\nSearching: '{term}'")
        papers = fetch_with_retry(term)
        new = 0
        for paper in papers:
            if paper["pmid"] not in seen_pmids:
                all_papers.append(paper)
                seen_pmids.add(paper["pmid"])
                new += 1

        save(all_papers)  # save after every term
        print(f"  Kept {new} new papers. Total so far: {len(all_papers)}")
        time.sleep(2)  # slightly longer pause between terms

    print(f"\nDone! Total unique papers: {len(all_papers)}")

if __name__ == "__main__":
    main()
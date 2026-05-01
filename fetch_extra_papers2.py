from Bio import Entrez
import json
import time

Entrez.email = "your_email@example.com"

EXTRA_TERMS = [
  
    "hypertension diabetes comorbidity risk management",
]

OUTPUT_FILE = "papers.json"

def load_existing():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)
    return papers, {p["pmid"] for p in papers}

def save(papers):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

def fetch(term, retries=3):
    for attempt in range(1, retries + 1):
        try:
            handle  = Entrez.esearch(db="pubmed", term=term, retmax=25,
                                     datetype="pdat", mindate="2015", maxdate="2025")
            record  = Entrez.read(handle)
            handle.close()
            ids     = record["IdList"]
            if not ids:
                return []
            time.sleep(0.5)
            handle  = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            papers  = []
            for article in records["PubmedArticle"]:
                try:
                    title    = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
                    abstract = str(article["MedlineCitation"]["Article"].get(
                                   "Abstract", {}).get("AbstractText", [""])[0])
                    year     = str(article["MedlineCitation"]["Article"]["Journal"]
                                   ["JournalIssue"]["PubDate"].get("Year", "unknown"))
                    pmid     = str(article["MedlineCitation"]["PMID"])
                    if len(abstract) > 100:
                        papers.append({"pmid": pmid, "title": title,
                                       "abstract": abstract, "year": year,
                                       "search_term": term})
                except Exception:
                    continue
            return papers
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
            else:
                return []

def main():
    all_papers, seen_pmids = load_existing()
    print(f"Starting with {len(all_papers)} papers")

    for term in EXTRA_TERMS:
        print(f"Fetching: '{term}'")
        papers = fetch(term)
        new = 0
        for p in papers:
            if p["pmid"] not in seen_pmids:
                all_papers.append(p)
                seen_pmids.add(p["pmid"])
                new += 1
        save(all_papers)
        print(f"  Added {new} new papers. Total: {len(all_papers)}")
        time.sleep(2)

    print(f"Done. Now run: python build_rag.py")

if __name__ == "__main__":
    main()
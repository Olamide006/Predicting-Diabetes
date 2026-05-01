from Bio import Entrez
import json
import time

Entrez.email = "your_email@example.com"

EXTRA_TERMS = [
    "physical inactivity sedentary type 2 diabetes risk",
    "exercise intervention diabetes prevention randomized",
    "hypertension insulin resistance type 2 diabetes",
    "blood pressure diabetes risk metabolic syndrome",
]

def fetch(term, max_results=30):
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
            title    = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
            abstract = str(article["MedlineCitation"]["Article"].get(
                           "Abstract", {}).get("AbstractText", [""])[0])
            year     = str(article["MedlineCitation"]["Article"]["Journal"]
                           ["JournalIssue"]["PubDate"].get("Year", "unknown"))
            pmid     = str(article["MedlineCitation"]["PMID"])
            if len(abstract) > 100:
                papers.append({"pmid": pmid, "title": title,
                                "abstract": abstract, "year": year, "search_term": term})
        except Exception:
            continue
    return papers

# Load existing papers
with open("papers.json", "r", encoding="utf-8") as f:
    all_papers = json.load(f)
seen_pmids = {p["pmid"] for p in all_papers}

for term in EXTRA_TERMS:
    print(f"Fetching: {term}")
    papers = fetch(term)
    new = 0
    for p in papers:
        if p["pmid"] not in seen_pmids:
            all_papers.append(p)
            seen_pmids.add(p["pmid"])
            new += 1
    print(f"  Added {new} new papers. Total: {len(all_papers)}")
    time.sleep(2)

with open("papers.json", "w", encoding="utf-8") as f:
    json.dump(all_papers, f, indent=2, ensure_ascii=False)

print("Done. Now run: python build_rag.py")
# clean_papers.py
import json
import re

def clean_text(text):
    if not text:
        return ""
    # Decode HTML entities first (&lt; → < and &gt; → >)
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    text = text.replace('&quot;', '"').replace('&#39;', "'")
    # Now strip actual HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove section labels
    text = re.sub(
        r'\b(Background|Objective|Objectives|Methods|Method|Results|Result|'
        r'Conclusion|Conclusions|Purpose|Aims|Aim|Introduction|Discussion|'
        r'Findings|Context|Setting|Design|Participants|Interventions)'
        r'(\s+and\s+\w+)?\s*[:\-]\s*',
        '',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'^\s*[:\-]\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    with open('papers.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    print(f"Cleaning {len(papers)} papers...")
    for p in papers:
        p['abstract'] = clean_text(p.get('abstract', ''))
        p['title']    = clean_text(p.get('title', ''))
    with open('papers.json', 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print("Done. Now run: python build_rag.py")

if __name__ == "__main__":
    main()
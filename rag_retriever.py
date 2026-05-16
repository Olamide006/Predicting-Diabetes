# rag_retriever.py
# RAG retrieval engine for the Diabetes Risk Prediction System

import pickle
import numpy as np
import faiss
import re

INDEX_FILE = "rag_index.faiss"
META_FILE  = "rag_metadata.pkl"

RELEVANCE_THRESHOLD = 0.10  # lowered because we now use full PDF text

FEATURE_QUERIES = {
    "age":                  "aging older adults diabetes incidence risk Africa age-related",
    "bmi":                  "BMI body mass index obesity overweight diabetes Africa adiposity",
    "sex":                  "sex gender women men diabetes differences hormones",
    "family_history":       "family history genetic hereditary diabetes offspring parental",
    "gestational_diabetes": "gestational diabetes GDM pregnancy postpartum future diabetes",
    "physical_activity":    "physical activity exercise sedentary inactivity diabetes prevention",
    "hypertension":         "hypertension blood pressure diabetes comorbidity risk",
}

DIABETES_KEYWORDS = [
    'diabetes', 'diabetic', 'insulin', 'glucose', 'glycemic',
    'prediabetes', 'hyperglycemia', 'hba1c', 'blood sugar', 'glycaemic'
]


def clean_text(text):
    """Clean extracted PDF text."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_evidence_sentences(full_text, feature_name, max_sentences=2):
    """
    Extract the most relevant sentences from a paper's full text
    based on the feature being explained.
    """
    feature_keywords = {
        "age":                  ["age", "older", "aging", "elderly", "middle-aged", "years old"],
        "bmi":                  ["bmi", "obesity", "obese", "overweight", "body mass", "adipos"],
        "sex":                  ["sex", "gender", "women", "female", "male", "men"],
        "family_history":       ["family history", "genetic", "hereditary", "offspring", "parental", "first-degree"],
        "gestational_diabetes": ["gestational", "gdm", "pregnancy", "postpartum", "obstetric"],
        "physical_activity":    ["physical activity", "exercise", "sedentary", "inactivity", "walking"],
        "hypertension":         ["hypertension", "blood pressure", "systolic", "diastolic"],
    }

    key = feature_name.lower().replace(" ", "_")
    keywords = feature_keywords.get(key, [feature_name])

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    relevant = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 40 or len(sent) > 400:
            continue
        sent_lower = sent.lower()
        # Must contain a diabetes keyword AND a feature keyword
        has_diabetes = any(kw in sent_lower for kw in DIABETES_KEYWORDS)
        has_feature  = any(kw in sent_lower for kw in keywords)
        if has_diabetes and has_feature:
            relevant.append(sent)

    return relevant[:max_sentences]


def build_explanation(feature_name, feature_value, shap_value, prediction_label, papers):
    """
    Build a paper-grounded explanation for a feature's contribution.
    SHAP value determines strength/direction.
    Papers provide the actual evidence sentences.
    """
    direction = "increases" if shap_value > 0 else "decreases"
    strength  = (
        "strongly"   if abs(shap_value) > 0.1  else
        "moderately" if abs(shap_value) > 0.05 else
        "slightly"
    )

    key = feature_name.lower().replace(" ", "_")

    # ── Patient-facing opening based on feature + value ──────────────
    if key == "age":
        age = int(feature_value)
        if age >= 45:
            opening = (
                f"At {age} years old, age is a significant factor in this prediction. "
                f"Research consistently shows that diabetes risk rises substantially after age 45 "
                f"as insulin sensitivity declines and pancreatic beta-cell function weakens."
            )
        elif age >= 35:
            opening = (
                f"At {age} years old, you are approaching the age range where diabetes risk "
                f"begins to climb more sharply. Insulin sensitivity gradually declines through "
                f"the mid-30s, making lifestyle habits particularly important at this stage."
            )
        else:
            opening = (
                f"At {age} years old, age alone is a lower risk factor. However, diabetes in "
                f"younger adults is rising, particularly when combined with other risk factors "
                f"such as obesity, inactivity, or family history."
            )

    elif key == "bmi":
        bmi = float(feature_value)
        if bmi >= 35:
            opening = (
                f"Your BMI of {bmi:.1f} falls in the severely obese range. "
                f"At this level, insulin resistance is highly likely and even modest weight "
                f"loss of 5-10% can significantly improve blood sugar regulation."
            )
        elif bmi >= 30:
            opening = (
                f"Your BMI of {bmi:.1f} falls in the obese range. Excess body fat, "
                f"particularly around the abdomen, directly impairs the body's ability "
                f"to use insulin effectively, raising diabetes risk substantially."
            )
        elif bmi >= 25:
            opening = (
                f"Your BMI of {bmi:.1f} places you in the overweight range. "
                f"This puts additional strain on insulin-producing cells and raises "
                f"your risk of developing type 2 diabetes over time."
            )
        else:
            opening = (
                f"Your BMI of {bmi:.1f} is within the healthy range, which is a "
                f"positive protective factor against type 2 diabetes."
            )

    elif key == "sex":
        if feature_value == 0:
            opening = (
                "Being female introduces specific hormonal risk pathways including PCOS, "
                "menopausal oestrogen decline, and the long-term effects of gestational diabetes. "
                "These factors can reduce insulin sensitivity and raise diabetes risk over time."
            )
        else:
            if shap_value < 0:
                opening = (
                    "Being male is associated with a relatively lower diabetes risk in this "
                    "profile compared to other factors present. Maintaining a healthy weight "
                    "and staying active remains the best protection."
                )
            else:
                opening = (
                    "Men tend to develop insulin resistance at lower BMI levels than women, "
                    "often due to visceral fat accumulation even without significant overall "
                    "weight gain, raising diabetes risk."
                )

    elif key == "family_history":
        if feature_value == 1:
            opening = (
                "Having a first-degree relative with diabetes roughly doubles your inherited "
                "genetic risk. Genetic factors influence both insulin secretion and sensitivity, "
                "making lifestyle modifications especially important for you."
            )
        else:
            opening = (
                "No family history of diabetes means your inherited genetic risk is lower. "
                "However, lifestyle factors such as poor diet, physical inactivity, and excess "
                "weight remain important contributors regardless of family history."
            )

    elif key == "gestational_diabetes":
        if feature_value == 1:
            opening = (
                "A history of gestational diabetes (GDM) is one of the strongest predictors "
                "of future type 2 diabetes. Women who had GDM face up to a 10-fold increased "
                "lifetime risk. The underlying insulin resistance often persists after pregnancy."
            )
        else:
            opening = (
                "No history of gestational diabetes removes one significant risk pathway. "
                "However, other risk factors can still contribute to diabetes development, "
                "so maintaining a healthy lifestyle remains important."
            )

    elif key == "physical_activity":
        if feature_value == 0:
            opening = (
                "Physical inactivity is a major modifiable risk factor for type 2 diabetes. "
                "Without regular exercise, the body gradually loses insulin sensitivity, "
                "leading to higher blood glucose levels over time."
            )
        else:
            opening = (
                "Being physically active is one of the most effective protections against "
                "type 2 diabetes. Exercise directly improves insulin sensitivity and helps "
                "regulate blood glucose levels."
            )

    elif key == "hypertension":
        if feature_value == 1:
            opening = (
                "Hypertension and type 2 diabetes share common underlying mechanisms including "
                "insulin resistance, inflammation, and vascular dysfunction. People with high "
                "blood pressure are roughly twice as likely to develop type 2 diabetes."
            )
        else:
            opening = (
                "Normal blood pressure is a positive metabolic indicator. The absence of "
                "hypertension reduces one significant pathway toward diabetes development. "
                "Continue maintaining healthy blood pressure through diet and exercise."
            )

    else:
        opening = (
            f"This factor {strength} {direction} your diabetes risk "
            f"based on the model's analysis of your profile."
        )

    # ── SHAP context line ─────────────────────────────────────────────
    if prediction_label == "Normal":
        outcome_direction = "decreases" if shap_value > 0 else "increases"
        shap_context = (
            f"The model's analysis shows this factor {strength} {outcome_direction} "
            f"your diabetes risk."
    )
    elif prediction_label == "Prediabetes":
        shap_context = (
            f"The model's analysis shows this factor {strength} {direction} "
            f"your risk of Prediabetes."
    )
    else:
        shap_context = (
            f"The model's analysis shows this factor {strength} {direction} "
            f"your risk of Diabetic."
    )
    # ── Extract evidence from actual paper text ───────────────────────
    evidence_parts = []
    for paper in papers:
        full_text = paper.get("full_text", paper.get("abstract", ""))
        full_text = clean_text(full_text)
        sentences = extract_evidence_sentences(full_text, key)
        title     = paper.get("title", "").replace(".pdf", "").strip()
        year      = paper.get("year", "")

        if sentences:
            evidence_parts.append(
                f'Research evidence: "{sentences[0]}" '
                f'(Source: {title[:80]}, {year})'
            )

    # ── Combine into final explanation ────────────────────────────────
    if evidence_parts:
        evidence_block = " | ".join(evidence_parts[:2])  # max 2 paper citations
        detail = f"{opening} {shap_context} {evidence_block}."
    else:
        detail = f"{opening} {shap_context}"

    return detail, direction, strength


class RAGRetriever:

    def __init__(self):
        self.index      = None
        self.vectorizer = None
        self.metadata   = None
        self.loaded     = False

    def load(self):
        try:
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.metadata   = data["metadata"]
            self.loaded     = True
        except FileNotFoundError:
            print("RAG index not found. Run build_rag.py first.")
            self.loaded = False

    def retrieve(self, query, top_k=3):
        if not self.loaded:
            return []

        query_vec = self.vectorizer.transform([query]).toarray().astype(np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores, indices = self.index.search(query_vec, top_k * 3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < RELEVANCE_THRESHOLD:
                continue

            paper          = self.metadata[idx].copy()
            full_text      = paper.get("full_text", paper.get("abstract", "")).lower()
            title_lower    = paper.get("title", "").lower()

            is_diabetes_relevant = any(
                kw in full_text or kw in title_lower
                for kw in DIABETES_KEYWORDS
            )
            if not is_diabetes_relevant:
                continue

            paper["score"] = float(score)
            results.append(paper)

            if len(results) == top_k:
                break

        return results

    def retrieve_for_feature(self, feature_name, top_k=3):
        key   = feature_name.lower().replace(" ", "_")
        query = FEATURE_QUERIES.get(key, key + " type 2 diabetes risk")
        return self.retrieve(query, top_k)


if __name__ == "__main__":
    retriever = RAGRetriever()
    retriever.load()

    test_features = ["bmi", "physical_activity", "hypertension", "age", "family_history"]

    for feature in test_features:
        print(f"\n--- {feature.upper()} ---")
        papers = retriever.retrieve_for_feature(feature, top_k=3)
        if papers:
            for p in papers:
                print(f"  [{p['score']:.3f}] {p['title']} ({p['year']})")
                sentences = p.get("full_text", "")[:500]
                print(f"  Preview: {sentences[:200]}")
        else:
            print("  No relevant papers found above threshold.")
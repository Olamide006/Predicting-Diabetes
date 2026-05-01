# rag_retriever.py
# RAG retrieval engine for the Diabetes Risk Prediction System

import pickle
import numpy as np
import faiss
import re

INDEX_FILE = "rag_index.faiss"
META_FILE  = "rag_metadata.pkl"

RELEVANCE_THRESHOLD = 0.18

def clean_abstract(text):
    """Strip HTML tags, section labels, and clean up PubMed abstract text."""
    # Strip HTML tags first
    text = re.sub(r'<[^>]+>', '', text)
    # Remove section labels at the start or after punctuation
    text = re.sub(
        r'(^|\.\s*)(Background|Objective|Objectives|Methods|Method|Results|Result|'
        r'Conclusion|Conclusions|Purpose|Aims|Aim|Introduction|Discussion|Findings)'
        r'\s*:\s*',
        r'\1',
        text,
        flags=re.IGNORECASE
    )
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

FEATURE_QUERIES = {
    "age":                 "age older adults type 2 diabetes risk insulin resistance",
    "bmi":                 "BMI obesity overweight type 2 diabetes risk insulin resistance",
    "sex":                 "sex gender differences type 2 diabetes risk women men",
    "family_history":      "family history genetic hereditary type 2 diabetes risk",
    "gestational_diabetes":"gestational diabetes mellitus future type 2 diabetes risk women",
    "physical_activity":   "physical activity exercise inactivity type 2 diabetes prevention",
    "hypertension":        "hypertension high blood pressure type 2 diabetes risk",
}

DIABETES_KEYWORDS = [
    'diabetes', 'diabetic', 'insulin', 'glucose', 'glycemic',
    'prediabetes', 'hyperglycemia', 'hba1c', 'blood sugar', 'glycaemic'
]



def build_explanation(feature_name, feature_value, shap_value, prediction_label, papers):
    direction = "increases" if shap_value > 0 else "decreases"
    strength = (
        "strongly"   if abs(shap_value) > 0.1  else
        "moderately" if abs(shap_value) > 0.05 else
        "slightly"
    )

    key = feature_name.lower().replace(" ", "_")

    if key == "age":
        age = int(feature_value)
        if age >= 45:
            detail = (
                f"At {age} years old, age is one of the most significant factors in your result. "
                f"The risk of type 2 diabetes rises substantially after age 45, as the body "
                f"gradually loses insulin sensitivity and pancreatic beta-cell function declines. "
                f"Regular blood glucose screening is strongly recommended at this age."
            )
        elif age >= 35:
            detail = (
                f"At {age} years old, you are approaching the age range where diabetes risk "
                f"begins to increase more sharply. The body's insulin sensitivity starts declining "
                f"in the mid-30s, making lifestyle habits particularly important at this stage."
            )
        else:
            detail = (
                f"At {age} years old, age is a less dominant risk factor on its own. "
                f"However, diabetes in younger adults is increasingly common, especially when "
                f"combined with obesity, inactivity, or a family history of diabetes."
            )

    elif key == "bmi":
        bmi = float(feature_value)
        if bmi >= 35:
            category = "severely obese"
            advice = (
                "At this level, the risk of insulin resistance is very high. "
                "Even a modest weight reduction of 5-10% can significantly improve "
                "blood sugar regulation and reduce diabetes risk."
            )
        elif bmi >= 30:
            category = "obese"
            advice = (
                "Obesity is one of the strongest modifiable risk factors for type 2 diabetes. "
                "Excess body fat, particularly around the abdomen, interferes with the body's "
                "ability to use insulin effectively."
            )
        elif bmi >= 25:
            category = "overweight"
            advice = (
                "Being overweight increases strain on the body's insulin-producing cells. "
                "Maintaining a healthy weight through diet and exercise significantly reduces risk."
            )
        else:
            category = "healthy weight range"
            advice = (
                "Your weight is within a healthy range, which is a positive factor "
                "for diabetes prevention."
            )
        detail = f"Your BMI of {bmi:.1f} places you in the {category}. {advice}"

    elif key == "sex":
        if feature_value == 0:
            detail = (
                "Being female introduces specific diabetes risk factors including hormonal "
                "changes during menopause, polycystic ovary syndrome (PCOS), and the long-term "
                "effects of gestational diabetes. Oestrogen decline after menopause can reduce "
                "insulin sensitivity. Women should monitor blood glucose regularly, particularly "
                "after age 40 or following a pregnancy with gestational diabetes."
            )
        else:
            if shap_value < 0:
                detail = (
                    "Being male is associated with a relatively lower diabetes risk in this "
                    "profile compared to other factors. While men can develop visceral fat at "
                    "lower BMI levels, your overall risk profile suggests other factors are "
                    "more dominant here. Maintaining a healthy weight and staying active "
                    "remains the best protection."
                )
            else:
                detail = (
                    "Men tend to develop type 2 diabetes at lower BMI levels than women, "
                    "suggesting higher susceptibility to insulin resistance even without "
                    "significant weight gain. Visceral fat accumulation, common in men, "
                    "is particularly associated with metabolic dysfunction and elevated "
                    "diabetes risk."
                )

    elif key == "family_history":
        if feature_value == 1:
            detail = (
                "Having a first-degree relative (parent or sibling) with diabetes roughly doubles "
                "your risk of developing the condition. Genetic factors influence insulin secretion "
                "and sensitivity. This makes lifestyle modifications such as healthy diet, regular "
                "exercise, and weight management especially important for you."
            )
        else:
            detail = (
                "No family history of diabetes reduces your inherited genetic risk. "
                "However, lifestyle factors such as poor diet, physical inactivity, and excess "
                "weight remain important contributors regardless of family history."
            )

    elif key == "gestational_diabetes":
        if feature_value == 1:
            detail = (
                "A history of gestational diabetes (GDM) is one of the strongest predictors of "
                "future type 2 diabetes. Women who had GDM have up to a 10-fold increased lifetime "
                "risk compared to those without. The underlying insulin resistance that caused GDM "
                "often persists after pregnancy. Regular blood sugar monitoring at least every "
                "1-3 years is strongly recommended."
            )
        else:
            detail = (
                "No history of gestational diabetes removes one significant risk pathway. "
                "However, other risk factors can still contribute to diabetes development, "
                "so maintaining a healthy weight and active lifestyle remains important."
            )

    elif key == "physical_activity":
        if feature_value == 0:
            detail = (
                "Physical inactivity is a major modifiable risk factor for type 2 diabetes. "
                "Without regular exercise, the body becomes less sensitive to insulin over time, "
                "leading to higher blood glucose levels. The WHO recommends at least 150 minutes "
                "of moderate-intensity activity per week such as brisk walking, cycling, or "
                "swimming. Even small increases in daily movement can make a meaningful difference."
            )
        else:
            detail = (
                "Being physically active is one of the most effective ways to reduce diabetes risk. "
                "Exercise improves insulin sensitivity, helps maintain a healthy weight, and "
                "directly lowers blood glucose levels. Continue aiming for at least 150 minutes "
                "of moderate activity per week to maintain this protective effect."
            )

    elif key == "hypertension":
        if feature_value == 1:
            detail = (
                "Having hypertension significantly increases your diabetes risk. High blood "
                "pressure and diabetes share common underlying mechanisms including insulin "
                "resistance, inflammation, and vascular dysfunction. People with hypertension "
                "are roughly twice as likely to develop type 2 diabetes. Managing blood pressure "
                "through diet, exercise, and medication where prescribed also helps protect "
                "against diabetes."
            )
        else:
            detail = (
                "Normal blood pressure is a positive indicator for metabolic health. "
                "The absence of hypertension reduces one significant risk pathway for diabetes. "
                "Continue maintaining healthy blood pressure through a balanced diet, regular "
                "physical activity, and limited salt and alcohol intake."
            )

    else:
        detail = (
            f"This factor {strength} {direction} your diabetes risk based on the model's analysis."
        )

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
            abstract_lower = paper['abstract'].lower()
            title_lower    = paper['title'].lower()

            is_diabetes_relevant = any(
                kw in abstract_lower or kw in title_lower
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
        else:
            print("  No relevant papers found above threshold.")
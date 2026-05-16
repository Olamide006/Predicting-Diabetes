# app.py
# Diabetes Risk Prediction System — Streamlit Web Interface

import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd
import re
from rag_retriever import RAGRetriever, build_explanation


def clean_abstract(text):
    """Strip HTML tags and section labels from abstract text."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
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


def get_risk_explanation(prediction_label):
    if prediction_label == "Diabetic":
        return (
            "Based on your risk profile, the model has classified you as HIGH RISK for diabetes. "
            "This does not constitute a medical diagnosis. You are strongly advised to consult a "
            "healthcare provider for proper clinical evaluation, including fasting blood glucose "
            "and HbA1c testing. Early intervention can prevent or delay serious complications "
            "including cardiovascular disease, kidney failure, and vision loss (IDF, 2024)."
        )
    elif prediction_label == "Prediabetes":
        return (
            "Based on your risk profile, the model has classified you as MODERATE RISK, suggesting "
            "possible prediabetes. Prediabetes is a critical window where lifestyle changes can "
            "prevent full diabetes from developing (ADA, 2024). You are advised to consult a "
            "healthcare provider and consider adopting healthier dietary habits and increasing "
            "physical activity."
        )
    else:
        return (
            "Based on your risk profile, the model has classified you as LOW RISK for diabetes. "
            "Continue maintaining a healthy lifestyle including regular physical activity, balanced "
            "diet, and routine health check-ups. Even at low risk, annual screening is recommended "
            "especially as you age (WHO, 2023)."
        )


@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le


@st.cache_resource
def load_rag():
    retriever = RAGRetriever()
    retriever.load()
    return retriever


model, scaler, le = load_model()
rag = load_rag()

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# ── HEADER ─────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("""
This system predicts diabetes risk using seven non-clinical,
self-reported features. No laboratory tests are required.
*For Sub-Saharan African populations.*
""")
st.divider()

# ── INPUT FORM ─────────────────────────────────────────────────
st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)", min_value=18, max_value=100,
        value=None, placeholder="Enter age"
    )
    sex = st.selectbox("Sex", ["Female", "Male"])
    bmi = st.number_input(
        "BMI", min_value=10.0, max_value=60.0,
        value=None, placeholder="Enter BMI", step=0.1
    )
    family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])

with col2:
    if sex == "Male":
        st.selectbox(
            "History of Gestational Diabetes (GDM)", ["No"],
            disabled=True, help="Not applicable for male patients"
        )
        previous_gdm = "No"
    else:
        previous_gdm = st.selectbox(
            "History of Gestational Diabetes (GDM)", ["No", "Yes"]
        )
    physically_active = st.selectbox("Physically Active", ["No", "Yes"])
    has_hypertension  = st.selectbox("Has Hypertension", ["No", "Yes"])

st.divider()

# ── PREDICT ────────────────────────────────────────────────────
if st.button("Predict Diabetes Risk", use_container_width=True):

    if age is None or bmi is None:
        st.error("Please fill in all fields before predicting.")

    else:
        # Encode inputs
        sex_enc    = 0 if sex == "Female" else 1
        family_enc = 1 if family_history == "Yes" else 0
        gdm_enc    = 1 if previous_gdm == "Yes" else 0
        active_enc = 1 if physically_active == "Yes" else 0
        hyper_enc  = 1 if has_hypertension == "Yes" else 0

        input_data = pd.DataFrame(
            [[age, sex_enc, bmi, family_enc, gdm_enc, active_enc, hyper_enc]],
            columns=[
                'age', 'sex', 'bmi', 'family_history_diabetes',
                'previous_gdm', 'physically_active', 'has_hypertension'
            ]
        )

        input_scaled     = scaler.transform(input_data)
        prediction_enc   = model.predict(input_scaled)[0]
        prediction_label = le.inverse_transform([prediction_enc])[0]

        # ── PREDICTION RESULT ──────────────────────────────────
        st.subheader("Prediction Result")
        if prediction_label == "Diabetic":
            st.error(f"High Risk: {prediction_label}")
        elif prediction_label == "Prediabetes":
            st.warning(f"Moderate Risk: {prediction_label}")
        else:
            st.success(f"Low Risk: {prediction_label}")

        st.divider()

        # ── SHAP CHART ─────────────────────────────────────────
        st.subheader("Why This Prediction?")
        st.markdown(
            "The chart below shows which factors contributed most to this prediction. "
            "Red bars increase risk, green bars decrease risk. "
            "Longer bars indicate stronger influence on the prediction."
        )

        feature_names = [
            'Age', 'Sex', 'BMI', 'Family History',
            'Previous GDM', 'Physically Active', 'Hypertension'
        ]

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        class_idx   = list(le.classes_).index(prediction_label)

        if isinstance(shap_values, list):
            shap_vals = shap_values[class_idx][0]
        else:
            shap_vals = shap_values[0, :, class_idx]

        shap_vals = np.array(shap_vals).flatten()[:len(feature_names)]

        # Sort by absolute SHAP value for better visualisation
        sorted_idx    = np.argsort(np.abs(shap_vals))
        sorted_names  = [feature_names[i] for i in sorted_idx]
        sorted_vals   = shap_vals[sorted_idx]
        sorted_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in sorted_vals]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(sorted_names, sorted_vals, color=sorted_colors)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP Value (Impact on Prediction)")
        ax.set_title(f"Feature Contributions for {prediction_label} Prediction")
        plt.tight_layout()
        st.pyplot(fig)

        st.divider()

        # ── RAG RESEARCH EVIDENCE ──────────────────────────────
        st.subheader("📄 Research Evidence (RAG-Enhanced)")
        st.markdown(
            "Explanations below are generated for your specific risk profile "
            "and supported by peer-reviewed research manually selected from "
            "PubMed (2015–2025). Features are ranked by their SHAP influence "
            "on the prediction — strongest contributors appear first."
        )

        feature_keys = [
            'age', 'sex', 'bmi', 'family_history_diabetes',
            'previous_gdm', 'physically_active', 'has_hypertension'
        ]
        input_values = [
            age, sex_enc, bmi, family_enc, gdm_enc, active_enc, hyper_enc
        ]

        feature_name_map = {
            'age':                     'age',
            'sex':                     'sex',
            'bmi':                     'bmi',
            'family_history_diabetes': 'family_history',
            'previous_gdm':            'gestational_diabetes',
            'physically_active':       'physical_activity',
            'has_hypertension':        'hypertension',
        }

        feature_display = {
            'age':                     'Age',
            'sex':                     'Sex',
            'bmi':                     'BMI',
            'family_history_diabetes': 'Family History of Diabetes',
            'previous_gdm':            'Gestational Diabetes History',
            'physically_active':       'Physical Activity Level',
            'has_hypertension':        'Hypertension',
        }

        # Sort features by SHAP magnitude — strongest first (SHAP triage)
        top_features = sorted(
            zip(feature_keys, input_values, shap_vals),
            key=lambda x: abs(x[2]),
            reverse=True
        )

        if rag.loaded:
            for feat, val, shap_val in top_features:

                # Skip GDM for male patients
                if feat == 'previous_gdm' and sex == 'Male':
                    continue

                # Skip negligible SHAP contributions
                if abs(shap_val) < 0.01:
                    continue

                retriever_key = feature_name_map.get(feat, feat)

                # ── SHAP TRIAGE — mathematically driven, no if/else ──
                # Higher SHAP magnitude = more papers retrieved
                top_k  = max(1, min(5, round(abs(shap_val) * 50)))
                papers = rag.retrieve_for_feature(retriever_key, top_k=top_k)

                label = feature_display.get(
                    feat, feat.replace('_', ' ').title()
                )

                summary, direction, strength = build_explanation(
                    retriever_key, val, shap_val, prediction_label, papers
                )

                # ── Icon driven entirely by SHAP direction ────
                is_high_risk_prediction = prediction_label == "Diabetic"
                if is_high_risk_prediction:
                     icon             = "🔴" if shap_val > 0 else "🟢"
                     header_direction = "increases" if shap_val > 0 else "decreases"
                else:
                    icon             = "🟢" if shap_val > 0 else "🔴"
                    header_direction = "decreases" if shap_val > 0 else "increases"

                # ── Expander content ──────────────────────────
                with st.expander(
                    f"{icon} {label} — {strength} {header_direction} your risk "
                    f"(SHAP: {shap_val:+.3f})"
                ):
                    st.markdown("**What this means for you:**")
                    st.write(summary)

                    if papers:
                        st.write(f"**Supporting research ({len(papers)} papers):**")
                        for i, paper in enumerate(papers, 1):
                            clean_title   = clean_abstract(paper['title'])
                            clean_snippet = clean_abstract(paper['abstract'])
                            snippet       = clean_snippet[:200].rsplit(' ', 1)[0] + "..."
                            st.write(f"**{i}.** {clean_title} ({paper['year']})")
                            st.write(snippet)
                            if i < len(papers):
                                st.write("---")
                    else:
                        st.write(
                            "No closely matching papers found in the index for this feature."
                        )
        else:
            st.warning(
                "RAG index not found. Run build_rag.py to enable research evidence."
            )

        st.divider()

        # ── OVERALL RISK ASSESSMENT ────────────────────────────
        st.subheader("Overall Risk Assessment")
        st.info(get_risk_explanation(prediction_label))

        st.caption("""
        Disclaimer: This tool is designed to support early screening and
        is not a substitute for professional medical diagnosis.
        Please consult a qualified healthcare provider for clinical evaluation.
        """)
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────
print("Loading dataset...")
train = load_dataset("electricsheepafrica/african-diabetes-dataset", split="train").to_pandas()
validation = load_dataset("electricsheepafrica/african-diabetes-dataset", split="validation").to_pandas()
test = load_dataset("electricsheepafrica/african-diabetes-dataset", split="test").to_pandas()
df = pd.concat([train, validation, test], ignore_index=True)

# ── 2. SELECT FEATURES ────────────────────────────────────
features = ['age', 'sex', 'bmi', 'family_history_diabetes',
            'previous_gdm', 'physically_active', 'has_hypertension']
target = 'diabetes_status'
df = df[features + [target]]

# ── 3. ENCODE CATEGORICAL FEATURES ───────────────────────
print("Encoding features...")
df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
df['family_history_diabetes'] = df['family_history_diabetes'].astype(int)
df['previous_gdm'] = df['previous_gdm'].astype(int)
df['physically_active'] = df['physically_active'].astype(int)
df['has_hypertension'] = df['has_hypertension'].astype(int)

# Encode target variable
le = LabelEncoder()
df[target] = le.fit_transform(df[target])
print(f"Target classes: {le.classes_}")

# ── 4. SPLIT DATA ─────────────────────────────────────────
print("Splitting data...")
X = df[features]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 5. SCALE NUMERICAL FEATURES ──────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ── 6. HANDLE CLASS IMBALANCE WITH SMOTE ─────────────────
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Train size: {len(X_train)}")

# ── 7. DEFINE MODELS ──────────────────────────────────────
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# ── 8. TRAIN AND EVALUATE WITH STRATIFIED K-FOLD ─────────
print("\nTraining and evaluating models...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    cv_results = cross_validate(model, X_train, y_train, cv=skf,
                                 scoring=['f1_weighted', 'f1_macro', 'recall_weighted'],
                                 return_train_score=False)
    
    # Train on full training set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Metrics
    report = classification_report(y_test, y_pred, 
                                    target_names=le.classes_, output_dict=True)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"AUC-ROC (weighted): {auc:.4f}")
    print(f"CV Weighted F1: {cv_results['test_f1_weighted'].mean():.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    results[name] = {
        'model': model,
        'weighted_f1': report['weighted avg']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'auc': auc,
        'cv_f1': cv_results['test_f1_weighted'].mean()
    }

# ── 9. SELECT BEST MODEL ──────────────────────────────────
best_name = max(results, key=lambda x: results[x]['weighted_f1'])
best_model = results[best_name]['model']
print(f"\n✅ Best Model: {best_name}")
print(f"   Weighted F1: {results[best_name]['weighted_f1']:.4f}")
print(f"   Macro F1:    {results[best_name]['macro_f1']:.4f}")
print(f"   AUC-ROC:     {results[best_name]['auc']:.4f}")

# ── 10. SAVE MODEL AND SCALER ─────────────────────────────
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\n✅ Model, scaler and label encoder saved successfully.")
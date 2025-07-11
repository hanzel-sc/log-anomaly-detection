
# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import webbrowser
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix, roc_auc_score)

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from lime.lime_tabular import LimeTabularExplainer
import shap

from sklearn.metrics.pairwise import cosine_similarity

# === Step 1: Load Preprocessed Data ===
print("Loading preprocessed log file...")
df = pd.read_csv('preprocessed_hdfs_labeled.csv').dropna(subset=['Label'])
df = df.apply(pd.to_numeric, axis=1)

# Robust label mapping
label_mapping = {
    'Normal': 0, 'normal': 0, '0': 0, 0: 0,
    'Anomaly': 1, 'anomaly': 1, '1': 1, 1: 1
}
df['Label'] = df['Label'].map(label_mapping)

# Drop unmapped rows (NaNs)
df = df.dropna(subset=['Label'])

# Safe cast to int
df['Label'] = df['Label'].astype(int)

X = df.drop(columns=['Label'])
y = df['Label']

# === Step 2: Handle Class Imbalance with SMOTE ===
print("Applying SMOTE...")
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# === Step 3: Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42
)

# === Step 4: Initialize Models ===
models = {
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, 
                              use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, iterations=100, depth=7, learning_rate=0.1, random_state=42),
     "MLP": MLPClassifier(hidden_layer_sizes=(75,), max_iter=50, early_stopping=True, random_state=42, verbose=True)
}

# === Step 5: Train & Evaluate Models ===
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }
    print(f"\n=== {name} Report ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

# === Step 6: Voting Ensemble ===
ensemble = VotingClassifier(
    estimators=[('xgb', models['XGBoost']), ('lgbm', models['LightGBM']), ('cat', models['CatBoost'])],
    voting='soft'
)
ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)
y_prob_ens = ensemble.predict_proba(X_test)[:, 1]

results["Voting Ensemble"] = {
    "Accuracy": accuracy_score(y_test, y_pred_ens),
    "Precision": precision_score(y_test, y_pred_ens),
    "Recall": recall_score(y_test, y_pred_ens),
    "F1 Score": f1_score(y_test, y_pred_ens),
    "ROC AUC": roc_auc_score(y_test, y_prob_ens),
    "Confusion Matrix": confusion_matrix(y_test, y_pred_ens)
}

# === Step 7: Summary Table & Visualization ===
summary_df = pd.DataFrame(results).T.round(4)
summary_df.to_csv("enhanced_model_summary.csv")
print("\n=== Model Summary ===")
print(summary_df)

# Plot comparison
summary_df[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]].plot(
    kind="bar", figsize=(14, 6), colormap="tab10")
plt.title("Enhanced Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("enhanced_model_comparison.png")
plt.show()

# Confusion Matrix
for name, result in results.items():
    plt.figure(figsize=(5,4))
    sns.heatmap(result['Confusion Matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()


# === Step 8: SHAP Explainability (XGBoost) ===
print("\nComputing SHAP values (XGBoost)...")
shap_explainer = shap.Explainer(models['XGBoost'], X_train)
shap_values = shap_explainer(X_test[:100])

# Global Feature Importance
shap.summary_plot(shap_values, X_test[:100], plot_type='bar', show=False)
plt.tight_layout()
plt.savefig("shap_global_importance.png", dpi=300)
plt.close()

# === Step 9: LIME Explainability (Random Forest as proxy) ===
print("\nRunning LIME on 100 samples...")
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=["Normal", "Anomaly"],
    mode='classification'
)

lime_results = []
for i in range(100):
    exp = lime_explainer.explain_instance(
        data_row=X_test.iloc[i],
        predict_fn=models['XGBoost'].predict_proba,
        num_features=10
    )
    lime_results.append(exp.as_list())

agg_contrib = defaultdict(float)
for instance in lime_results:
    for feat, weight in instance:
        agg_contrib[feat] += weight

lime_df = pd.DataFrame(agg_contrib.items(), columns=["Feature", "Aggregate Contribution"])
lime_df = lime_df.sort_values(by="Aggregate Contribution", ascending=False)

# Plot LIME feature importance
plt.figure(figsize=(12, 6))
sns.barplot(data=lime_df.head(15), x="Aggregate Contribution", y="Feature", palette="mako")
plt.title("LIME Aggregate Feature Contributions")
plt.tight_layout()
plt.savefig("lime_feature_contributions.png")
plt.show()

# === Step 10: Save Single LIME Explanation ===
exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[5],
    predict_fn=models['XGBoost'].predict_proba,
    num_features=10
)
html_path = os.path.abspath("lime_instance_5.html")
exp.save_to_file(html_path)
webbrowser.open(f"file://{html_path}")

print("\n=== Step 11: SHAP + LIME Explanation Agreement ===")

# Step 11a: Extract SHAP values for 100 samples (already done)
shap_features = X_test.iloc[:100]
shap_vals_matrix = shap_values.values  # Already shape: (100, num_features)

# Step 11b: Convert LIME results to a matrix
lime_matrix = np.zeros_like(shap_vals_matrix)

feature_names = X.columns.tolist()
feature_index = {feat: i for i, feat in enumerate(feature_names)}

for i, instance_expl in enumerate(lime_results):
    for feat, val in instance_expl:
        idx = feature_index.get(feat)
        if idx is not None:
            lime_matrix[i, idx] = val

# Step 11c: Normalize both matrices for cosine similarity
def normalize_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms

shap_norm = normalize_rows(shap_vals_matrix)
lime_norm = normalize_rows(lime_matrix)

# Step 11d: Compute cosine similarity
similarities = np.sum(shap_norm * lime_norm, axis=1)
mean_agreement = np.mean(similarities)

# Step 11e: Plot distribution
plt.figure(figsize=(8, 5))
sns.histplot(similarities, kde=True, bins=15, color="teal")
plt.title("SHAP-LIME Agreement per Instance (Cosine Similarity)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.axvline(mean_agreement, color='red', linestyle='--', label=f"Mean = {mean_agreement:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig("shap_lime_agreement.png", dpi=300)
plt.show()

print(f"Average SHAP-LIME Cosine Similarity: {mean_agreement:.4f}")


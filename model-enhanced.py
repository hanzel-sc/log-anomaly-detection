
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


# -------------------------
# Parameters / Safeguards
# -------------------------
MODEL = models['XGBoost']                # XGBoost model from your training pipeline
FEATURE_NAMES = X.columns.tolist()       # feature names (must match order of X_train/X_test)
MAX_SAMPLES = 500                        # how many test samples to analyze
SHAP_BACKGROUND_SIZE = 200               # background size for TreeExplainer (safe default)
TOP_K = 10                               # for top-k overlap
OUTPUT_DIR = os.getcwd()

# Ensure X_train, X_test are DataFrames
assert isinstance(X_train, pd.DataFrame), "X_train must be a pandas DataFrame"
assert isinstance(X_test, pd.DataFrame), "X_test must be a pandas DataFrame"

n_test = min(MAX_SAMPLES, len(X_test))
if n_test == 0:
    raise ValueError("X_test is empty. Provide test samples to analyze.")

X_train_np = np.array(X_train)
X_test_np = np.array(X_test.iloc[:n_test])

# -------------------------
# Determine positive class index (for LIME/SHAP alignment)
# -------------------------
# We want the class index corresponding to label value 1 (Anomaly).
# Models from scikit-learn/XGBoost usually have attribute classes_
if hasattr(MODEL, 'classes_'):
    classes_arr = np.array(MODEL.classes_)
    # try to find class label == 1, else default to index 1 if binary, else use last class
    pos_idx_arr = np.where(classes_arr == 1)[0]
    if pos_idx_arr.size > 0:
        CLASS_INDEX = int(pos_idx_arr[0])
    else:
        # fallbacks
        if len(classes_arr) == 2:
            CLASS_INDEX = 1
        else:
            CLASS_INDEX = len(classes_arr) - 1
else:
    # fallback guess (binary)
    CLASS_INDEX = 1

print(f"[INFO] Using class index = {CLASS_INDEX} for 'Anomaly' explanations.")

# -------------------------
# Step 8: Compute SHAP values (TreeExplainer optimized)
# -------------------------
print("\n[STEP 8] Computing SHAP values...")

# Build safe background for TreeExplainer: sample from X_train (or use X_train if small)
bg_size = min(SHAP_BACKGROUND_SIZE, len(X_train))
shap_background = X_train.sample(bg_size, random_state=42)

# Create SHAP explainer (TreeExplainer for tree models is efficient)
shap_explainer = shap.Explainer(MODEL, shap_background, feature_names=FEATURE_NAMES)
shap_explanation = shap_explainer(X_test.iloc[:n_test])  # returns an Explanation object

# shap_explanation.values shape handling:
# - For binary/multi: could be (n_samples, n_features) or (n_samples, n_classes, n_features)
shap_vals = shap_explanation.values
if shap_vals.ndim == 3:
    # shape: (n_samples, n_classes, n_features) -> select class axis
    if CLASS_INDEX < shap_vals.shape[1]:
        shap_matrix = shap_vals[:, CLASS_INDEX, :]
    else:
        # fallback: choose last class
        shap_matrix = shap_vals[:, -1, :]
else:
    # shape: (n_samples, n_features)
    shap_matrix = shap_vals

# Ensure shape is (n_samples, n_features)
assert shap_matrix.shape[0] == n_test
n_features = shap_matrix.shape[1]
print(f"[INFO] SHAP matrix shape: {shap_matrix.shape}")

# Save global SHAP importance plot
print("[STEP 8] Saving SHAP summary plot...")
shap.summary_plot(shap_explanation, X_test.iloc[:n_test], plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_global_importance.png"), dpi=300)
plt.close()


# -------------------------
# Step 9: Compute LIME explanations (store Explanation objects)
# -------------------------
print("\n[STEP 9] Generating LIME explanations...")

lime_explainer = LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=FEATURE_NAMES,
    class_names=[str(c) for c in MODEL.classes_],
    mode='classification',
    discretize_continuous=False
)

lime_explanations = []  # store actual Explanation objects
for i in range(n_test):
    # LIME expects a 1-D numpy row for data_row
    data_row = X_test.iloc[i].values
    exp = lime_explainer.explain_instance(
        data_row=data_row,
        predict_fn=MODEL.predict_proba,
        num_features=min(n_features, max(20, TOP_K * 3))  # allow LIME to consider more features locally
    )
    lime_explanations.append(exp)

print(f"[INFO] Collected {len(lime_explanations)} LIME explanation objects.")

# Build LIME aggregate contributions (human-readable)
agg_contrib = defaultdict(float)
for exp in lime_explanations:
    # Use the class index mapping in as_map; choose CLASS_INDEX if present, else fallback to predicted class
    as_map = exp.as_map()
    if CLASS_INDEX in as_map:
        entries = as_map[CLASS_INDEX]
    else:
        # fallback: pick the first key present
        entries = list(as_map.values())[0]
    # entries : list of (feature_idx, weight)
    for feat_idx, weight in entries:
        # Map feature index from LIME to feature name robustly (domain_mapper)
        try:
            feat_name = exp.domain_mapper.feature_names[feat_idx]
        except Exception:
            # fallback: use feature_names list
            feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"f{feat_idx}"
        agg_contrib[feat_name] += weight

lime_df = pd.DataFrame(list(agg_contrib.items()), columns=["Feature", "Aggregate Contribution"])
lime_df = lime_df.sort_values(by="Aggregate Contribution", ascending=False)

# Save LIME aggregate plot
plt.figure(figsize=(12, 6))
sns.barplot(data=lime_df.head(20), x="Aggregate Contribution", y="Feature")
plt.title("LIME Aggregate Feature Contributions (Top 20)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lime_feature_contributions.png"), dpi=300)
plt.close()


# Save a single LIME HTML for inspection (sample index 5 if exists)
sample_idx_for_html = 5 if n_test > 5 else 0
exp_html = lime_explanations[sample_idx_for_html]
html_path = os.path.join(OUTPUT_DIR, "lime_instance_sample.html")
exp_html.save_to_file(html_path)
try:
    webbrowser.open(f"file://{html_path}")
except Exception:
    pass


# -------------------------
# Step 11: Build aligned dense matrices for SHAP and LIME
# -------------------------
print("\n[STEP 11] Building aligned matrices and computing agreement metrics...")

# 11a) SHAP matrix already: shap_matrix (n_test x n_features)
# Take care of NaNs/infs: replace with 0
shap_matrix = np.nan_to_num(shap_matrix, nan=0.0, posinf=0.0, neginf=0.0)

# 11b) Convert LIME explanations -> dense matrix (n_test x n_features)
lime_matrix = np.zeros((n_test, n_features), dtype=float)

for i, exp in enumerate(lime_explanations):
    as_map = exp.as_map()
    if CLASS_INDEX in as_map:
        entries = as_map[CLASS_INDEX]
    else:
        entries = list(as_map.values())[0]
    for feat_idx, weight in entries:
        # map feat_idx -> feature name then to index in FEATURE_NAMES
        try:
            feat_name = exp.domain_mapper.feature_names[feat_idx]
        except Exception:
            feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else None
        if feat_name is None:
            continue
        try:
            col_idx = FEATURE_NAMES.index(feat_name)
        except ValueError:
            # feature name not found in global features -> skip
            continue
        lime_matrix[i, col_idx] = weight

# Replace NaNs
lime_matrix = np.nan_to_num(lime_matrix, nan=0.0, posinf=0.0, neginf=0.0)

# 11c) Normalize rows safely for cosine similarity
def safe_normalize_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # avoid zeros -> set zero-norm rows to 1 to keep them zero after division
    norms_safe = np.where(norms == 0, 1.0, norms)
    return mat / norms_safe

shap_norm = safe_normalize_rows(shap_matrix)
lime_norm = safe_normalize_rows(lime_matrix)

# 11d) Per-instance cosine similarity (dot product of normalized vectors)
cosine_sims = np.sum(shap_norm * lime_norm, axis=1)  # length n_test
mean_cosine = float(np.mean(cosine_sims))
median_cosine = float(np.median(cosine_sims))
std_cosine = float(np.std(cosine_sims, ddof=1)) if len(cosine_sims) > 1 else 0.0

print(f"[RESULT] Average SHAP-LIME Cosine Similarity: {mean_cosine:.6f}")
print(f"[RESULT] Median SHAP-LIME Cosine Similarity:  {median_cosine:.6f} (std={std_cosine:.6f})")

# 11e) Top-k overlap (by absolute importance)
topk = min(TOP_K, n_features)
overlaps = []
for i in range(n_test):
    shap_top = np.argsort(np.abs(shap_matrix[i]))[-topk:]
    lime_top = np.argsort(np.abs(lime_matrix[i]))[-topk:]
    intersection = len(set(shap_top).intersection(set(lime_top)))
    overlaps.append(intersection / topk)
mean_topk_overlap = float(np.mean(overlaps))
print(f"[RESULT] Top-{topk} Feature Agreement (mean over {n_test} samples): {mean_topk_overlap:.4f}")

# ============================================================
# === STEP 12: SHAP–LIME AGREEMENT — NORMAL vs ANOMALY ANALYSIS
# ============================================================
print("\n[STEP 12] Running SHAP–LIME agreement analysis by class subsets...")

save_dir = os.path.abspath("shap_lime_classwise")
os.makedirs(save_dir, exist_ok=True)

# ------------------------------------------------------------
# Helper: compute classwise SHAP-LIME agreement
# ------------------------------------------------------------
def compute_agreement(shap_matrix, lime_matrix, idxs):
    """Returns cosine similarity, top-k overlap for a subset of indices."""
    if len(idxs) == 0:
        return None

    s = shap_matrix[idxs]
    l = lime_matrix[idxs]

    # normalize
    s_norm = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)
    l_norm = l / (np.linalg.norm(l, axis=1, keepdims=True) + 1e-9)

    # cosine sim
    cos_sim = np.sum(s_norm * l_norm, axis=1)

    # top-k overlap
    k = 10
    overlap_scores = []
    for i in range(len(idxs)):
        s_top = np.argsort(s[i])[-k:]
        l_top = np.argsort(l[i])[-k:]
        inter = len(set(s_top).intersection(set(l_top)))
        overlap_scores.append(inter / k)

    return {
        "cos_sim": cos_sim,
        "overlap": overlap_scores,
        "n": len(idxs)
    }

# ------------------------------------------------------------
# Build index sets
# ------------------------------------------------------------
# Only these samples have SHAP + LIME explanations
shap_available = np.arange(shap_matrix.shape[0])   # normally 0..99

# Raw index sets
true_normal_raw = np.where(y_test.values == 0)[0]
true_anomaly_raw = np.where(y_test.values == 1)[0]

pred_labels = models['XGBoost'].predict(X_test)
pred_normal_raw = np.where(pred_labels == 0)[0]
pred_anomaly_raw = np.where(pred_labels == 1)[0]

# Restrict to samples for which we have SHAP+LIME explanations
true_normal = np.intersect1d(true_normal_raw, shap_available)
true_anomaly = np.intersect1d(true_anomaly_raw, shap_available)
pred_normal = np.intersect1d(pred_normal_raw, shap_available)
pred_anomaly = np.intersect1d(pred_anomaly_raw, shap_available)

# Intersection breakdown
TP = np.intersect1d(true_anomaly, pred_anomaly)
FP = np.intersect1d(true_normal, pred_anomaly)
TN = np.intersect1d(true_normal, pred_normal)
FN = np.intersect1d(true_anomaly, pred_normal)


index_groups = {
    "True-Normal": true_normal,
    "True-Anomaly": true_anomaly,
    "Pred-Normal": pred_normal,
    "Pred-Anomaly": pred_anomaly,
    "TN (True Negatives)": TN,
    "TP (True Positives)": TP,
    "FP (False Positives)": FP,
    "FN (False Negatives)": FN,
}

# ------------------------------------------------------------
# Compute & Save All Group Results
# ------------------------------------------------------------
all_results = {}

for name, idxs in index_groups.items():
    print(f"[INFO] Computing agreement for: {name} (n={len(idxs)})")

    res = compute_agreement(shap_matrix, lime_matrix, idxs)
    if res is None:
        print(f"[WARN] No samples for group: {name}")
        continue

    all_results[name] = res

    # Plot histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(res["cos_sim"], bins=15, kde=True)
    plt.axvline(np.mean(res["cos_sim"]), color='red', linestyle='--',
                label=f"Mean = {np.mean(res['cos_sim']):.3f}")
    plt.title(f"Cosine Similarity Distribution — {name}")
    plt.legend()
    plt.xlabel("Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_cosine_similarity.png"))
    plt.close()

    # Save overlap distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(res["overlap"], bins=10, kde=True)
    plt.title(f"Top-10 Feature Overlap — {name}")
    plt.xlabel("Overlap Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_topk_overlap.png"))
    plt.close()

# ------------------------------------------------------------
# Statistical Comparison (Core Research Value)
# ------------------------------------------------------------
from scipy.stats import ks_2samp, ttest_ind

def compare_groups(a, b, nameA, nameB):
    """Run statistical comparison of cosine similarity distributions."""
    ks = ks_2samp(a, b)
    ttest = ttest_ind(a, b, equal_var=False)
    return {
        "groupA": nameA,
        "groupB": nameB,
        "KS_stat": ks.statistic,
        "KS_p": ks.pvalue,
        "t_stat": ttest.statistic,
        "t_p": ttest.pvalue
    }

comparisons = []

# Compare: True-Normal vs True-Anomaly
if "True-Normal" in all_results and "True-Anomaly" in all_results:
    comparisons.append(
        compare_groups(all_results["True-Normal"]["cos_sim"],
                       all_results["True-Anomaly"]["cos_sim"],
                       "True-Normal", "True-Anomaly")
    )

# Compare: Pred-Normal vs Pred-Anomaly
if "Pred-Normal" in all_results and "Pred-Anomaly" in all_results:
    comparisons.append(
        compare_groups(all_results["Pred-Normal"]["cos_sim"],
                       all_results["Pred-Anomaly"]["cos_sim"],
                       "Pred-Normal", "Pred-Anomaly")
    )

# Compare: TP vs FP (VERY INTERESTING FOR RESEARCH)
if "TP (True Positives)" in all_results and "FP (False Positives)" in all_results:
    comparisons.append(
        compare_groups(all_results["TP (True Positives)"]["cos_sim"],
                       all_results["FP (False Positives)"]["cos_sim"],
                       "TP", "FP")
    )

# Save comparison results
comp_df = pd.DataFrame(comparisons)
comp_df.to_csv(os.path.join(save_dir, "classwise_statistical_comparison.csv"), index=False)

print("\n[STEP 12 SUMMARY]")
for row in comparisons:
    print(f"{row['groupA']} vs {row['groupB']}: "
          f"KS_p={row['KS_p']:.4f}, t_p={row['t_p']:.4f}")

print(f"\nAll plots and results saved in: {save_dir}")
print("✓ Classwise SHAP–LIME agreement analysis completed.")

# -------------------------
# Visualizations (stable, no NaNs)
# -------------------------
# Cosine similarity distribution
plt.figure(figsize=(8,5))
sns.histplot(cosine_sims, kde=True, bins=15)
plt.title("SHAP–LIME Cosine Similarity Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.axvline(mean_cosine, color='red', linestyle='--', label=f"Mean = {mean_cosine:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_lime_cosine_distribution.png"), dpi=300)
plt.close()

# Heatmap: pairwise cosine similarity matrix between SHAP instances and LIME instances
# This is shap_norm @ lime_norm.T which yields a stable numeric matrix
pairwise_sim = np.dot(shap_norm, lime_norm.T)  # shape (n_test, n_test)

plt.figure(figsize=(10,8))
sns.heatmap(pairwise_sim, cmap='coolwarm', center=0, square=False)
plt.title("Pairwise SHAP–LIME Cosine Similarity (SHAP instances vs LIME instances)")
plt.xlabel("LIME Instance Index")
plt.ylabel("SHAP Instance Index")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_lime_pairwise_similarity_heatmap.png"), dpi=300)
plt.close()

# Save numeric results to CSV for reproducibility
results_df = pd.DataFrame({
    "sample_index": np.arange(n_test),
    "cosine_similarity": cosine_sims,
    "topk_overlap": overlaps
})
results_df.to_csv(os.path.join(OUTPUT_DIR, "shap_lime_agreement_by_instance.csv"), index=False)

# Summary print
print("\nSummary:")
print(f" - Samples analyzed: {n_test}")
print(f" - Features: {n_features}")
print(f" - Mean cosine similarity: {mean_cosine:.6f}")
print(f" - Mean top-{topk} overlap: {mean_topk_overlap:.4f}")
print(f" - Plots saved in: {OUTPUT_DIR}")
print("✓ SHAP–LIME agreement analysis finished successfully.")

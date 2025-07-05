# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from lime.lime_tabular import LimeTabularExplainer
import webbrowser
from collections import defaultdict
from imblearn.over_sampling import SMOTE

# === Step 1: Load Preprocessed Data ===
print("Loading pre-processed log file...")
df = pd.read_csv('C:\\Users\\chris\\OneDrive\\Desktop\\College\\ARDC-Research\\preprocessed_hdfs_labeled.csv')
df = df.dropna(subset=['Label'])

if df['Label'].dtype == object:
    df['Label'] = df['Label'].map({'Normal': 0, 'Anomaly': 1})

X = df.drop(columns=['Label'])
y = df['Label'].astype(int)

print("Balancing the dataset using SMOTE (oversampling anomalies)...")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print class distribution
print(f"Original dataset shape: {df.shape}")
print("Original class distribution:")
print(df['Label'].value_counts())

print(f"\nBalanced dataset shape: {X_resampled.shape}")
print("Balanced class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# === Step 2: Train-Test Split ===
print("Splitting the dataset 70-30...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42
)

# === Step 3: Initialize Models ===
dt_model = DecisionTreeClassifier(random_state=10)
rf_model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=10)
gb_model = GradientBoostingClassifier(n_estimators=10, random_state=10)

models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

# === Step 4: Train and Evaluate Models ===
print("Training and evaluating models...")
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Normal', 'Actual Anomaly'],
                         columns=['Predicted Normal', 'Predicted Anomaly'])

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=1),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": cm_df
    }

    print(f"\n=== {name} Report ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=1))
    print(cm_df)

# === Step 5: Voting Ensemble ===
print("Training Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=[('dt', dt_model), ('rf', rf_model), ('gb', gb_model)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
y_pred_vote = voting_clf.predict(X_test)

vote_cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred_vote),
                          index=['Actual Normal', 'Actual Anomaly'],
                          columns=['Predicted Normal', 'Predicted Anomaly'])

print("\n=== Voting Ensemble Report ===")
print(classification_report(y_test, y_pred_vote, target_names=["Normal", "Anomaly"], zero_division=1))
print(vote_cm_df)

# === Step 6: Summary Table ===
summary_df = pd.DataFrame({
    model: {
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1 Score": metrics["F1 Score"]
    } for model, metrics in results.items()
}).T

summary_df.loc["Voting Ensemble"] = [
    accuracy_score(y_test, y_pred_vote),
    precision_score(y_test, y_pred_vote, zero_division=1),
    recall_score(y_test, y_pred_vote),
    f1_score(y_test, y_pred_vote)
]

print("\n=== Model Summary ===")
print(summary_df.round(4))
summary_df.to_csv("model_evaluation_summary.csv")

# === Step 7: Confusion Matrices ===
print("\nVisualizing confusion matrices...")
plt.figure(figsize=(18, 5))
for idx, (model_name, metrics) in enumerate(results.items()):
    plt.subplot(1, 4, idx+1)
    sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name}\nConfusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

plt.subplot(1, 4, 4)
sns.heatmap(vote_cm_df, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("Voting Ensemble\nConfusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300)
plt.show()

# === Step 8: Metric Comparison Bar Plot ===
print("Plotting model comparison metrics...")
summary_df[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
    kind="bar", figsize=(12, 6), colormap="tab10"
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("model_performance_comparison.png", dpi=300)
plt.show()

# === Step 9: Prediction Pie Chart ===
print("Pie chart of predictions (Voting Ensemble)...")
vote_counts = pd.Series(y_pred_vote).value_counts().sort_index()
plt.figure(figsize=(5, 5))
plt.pie(vote_counts, labels=["Normal", "Anomaly"],
        autopct='%1.1f%%', startangle=90, colors=["skyblue", "lightcoral"])
plt.title("Voting Ensemble Prediction Distribution")
plt.tight_layout()
plt.savefig("voting_ensemble_prediction_distribution.png", dpi=300)
plt.show()

# === Step 10: LIME Explainability ===
print("Running LIME on multiple instances (Random Forest)...")
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=["Normal", "Anomaly"],
    mode='classification'
)

lime_results = []
num_instances = 100

for i in range(num_instances):
    exp = explainer.explain_instance(
        data_row=X_test.iloc[i],
        predict_fn=rf_model.predict_proba,
        num_features=10
    )
    lime_results.append(exp.as_list())

# Aggregate feature weights
total_contrib = defaultdict(float)
for instance in lime_results:
    for feat, weight in instance:
        total_contrib[feat] += weight

lime_df = pd.DataFrame(total_contrib.items(), columns=["Feature", "Aggregate Contribution"])
lime_df = lime_df.sort_values(by="Aggregate Contribution", ascending=False)

# Plot aggregate LIME feature contributions
plt.figure(figsize=(12, 6))
sns.barplot(data=lime_df.head(15), x="Aggregate Contribution", y="Feature", palette="viridis")
plt.title("LIME Aggregate Feature Contributions (Top 15 across 100 samples)")
plt.xlabel("Cumulative Contribution")
plt.tight_layout()
plt.savefig("lime_aggregate_feature_contribution.png", dpi=300)
plt.show()

# Save individual explanation for report
idx = 5
explanation = explainer.explain_instance(
    data_row=X_test.iloc[idx],
    predict_fn=rf_model.predict_proba,
    num_features=10
)
html_path = os.path.abspath("lime_explanation_instance_5.html")
explanation.save_to_file(html_path)
print(f"LIME explanation for instance 5 saved to: {html_path}")
webbrowser.open(f"file://{html_path}")

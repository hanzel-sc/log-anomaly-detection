import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

# === Step 1: Set File paths ===
input_dir = r'C:\Users\chris\OneDrive\Desktop\College\ARDC-Research'
output_dir = os.path.join(input_dir, 'parsed_train_logfile')

# === Step 2: Load Structured Log File ===
print("Reading structured log file...")
df = pd.read_csv(os.path.join(output_dir, 'train_logfile.log_structured.csv'))

# === Step 3: Extract BlockId (Vectorized) ===
print("Extracting BlockId using vectorized method...")
df['BlockId'] = df['Content'].str.extract(r'(blk_-?\d+)')
df = df.dropna(subset=['BlockId'])

# === Step 4: Create Sessions ===
print("Grouping EventIds by BlockId to create sessions...")
session_df = df.groupby('BlockId')['EventId'].apply(lambda x: ' '.join(x)).reset_index()
block_ids = session_df['BlockId'].tolist()
sessions = session_df['EventId'].tolist()

# === Step 5: TF-IDF + Normalization (Sparse-compatible) ===
print("Vectorizing sessions using TF-IDF (sparse)...")
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(sessions)

print("Normalizing TF-IDF features using MaxAbsScaler...")
scaler = MaxAbsScaler()
X_normalized = scaler.fit_transform(X_tfidf)

# === Step 6: Load Anomaly Labels and Map to BlockId ===
print("Loading and mapping labels...")
labels_path = os.path.join(input_dir, 'preprocessed', 'anomaly_label.csv')
label_df = pd.read_csv(labels_path)
label_df['Label'] = label_df['Label'].map({'Anomaly': 1, 'Normal': 0})
label_map = dict(zip(label_df['BlockId'], label_df['Label']))

# === Step 7: Combine Features with Labels ===
print("Combining normalized features with labels...")
final_data = []
for i, block_id in enumerate(block_ids):
    label = label_map.get(block_id)
    if label is not None:
        row = X_normalized[i].toarray()[0].tolist()
        row.append(label)
        final_data.append(row)

# === Step 8: Save to CSV ===
print("Saving final preprocessed dataset to CSV...")
columns = vectorizer.get_feature_names_out().tolist()
columns.append('Label')

final_df = pd.DataFrame(final_data, columns=columns)
output_csv = os.path.join(input_dir, 'preprocessed_hdfs_labeled.csv')
final_df.to_csv(output_csv, index=False)

# === Done ===
print(f"\nFinal dataset saved to: {output_csv}")
print(f"Final shape: {final_df.shape}")
print(final_df.head())

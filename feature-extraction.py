import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('C:\\Users\\chris\\OneDrive\\Desktop\\College\\ARDC-Research\\parsed_logfile\\HDFS_2k.log_structured.csv')

# Extract BlockId from Content
df['BlockId'] = df['Content'].apply(lambda x: re.search(r'blk_-?\d+', x).group() if re.search(r'blk_-?\d+', x) else None)

# Output 1: Show the DataFrame with BlockId
print("DataFrame with BlockId:")
print(df.head())

# Drop rows without a Block ID
df = df.dropna(subset=['BlockId'])

# Output 2: Show the DataFrame after dropping rows without BlockId
print("\nDataFrame after dropping rows without BlockId:")
print(df.head())

# Group by Block ID to form sessions
session_dict = df.groupby('BlockId')['EventId'].apply(list).to_dict()

# Output 3: Show the session_dict (sessions grouped by BlockId)
print("\nSession Dictionary (BlockId -> EventId List):")
print(session_dict)

# Join event IDs into space-separated strings for each session
sessions = [' '.join(events) for events in session_dict.values()]

# Output 4: Show the sessions list (space-separated EventId strings)
print("\nSessions List (Space-Separated EventId Strings):")
print(sessions[:5])  # Display only the first 5 sessions for brevity

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(sessions).toarray()

# Output 5: Show the shape and first row of the TF-IDF matrix
print("\nTF-IDF Matrix (Shape and First Row):")
print(X_tfidf.shape)
print(X_tfidf[0])

# Standard Scaling
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_tfidf)

# Output 6: Show the shape and first row of the normalized TF-IDF matrix
print("\nNormalized TF-IDF Matrix (Shape and First Row):")
print(X_normalized.shape)
print(X_normalized[0])

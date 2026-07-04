import pandas as pd
import nltk
from sklearn.datasets import fetch_20newsgroups
from utils import clean_text,preprocess_text_advanced

# --- 1. Data Acquisition and Preprocessing ---

# Download necessary NLTK data (if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Fetch the 20 newsgroups dataset
print("Loading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Create a pandas DataFrame from the dataset
df_full = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})
print(f"Original dataset loaded with {len(df_full)} documents.")

# Initialize lemmatizer and stopwords

# Apply cleaning and preprocessing
print("Cleaning and preprocessing text...")
df_full['cleaned_text'] = df_full['text'].apply(clean_text)
df_full['preprocessed_text'] = df_full['cleaned_text'].apply(preprocess_text_advanced)

# Filter out empty preprocessed texts
df_processed = df_full[df_full['preprocessed_text'].notna() & (df_full['preprocessed_text'].str.strip() != '')].copy()
print(f"Processed dataset contains {len(df_processed)} non-empty documents.")

# --- 2. Full Lingo Pipeline Implementation ---
from model import full_lingo_pipeline

# --- 3. Execute the pipeline with (default or optimized) hyperparameters ---
# Using default hyperparameters for this full implementation
# In a real scenario, optimal_n_components_svd and optimal_n_clusters_kmeans would come from prior optimization steps.
optimal_n_components_svd_val = 100
optimal_n_clusters_kmeans_val = 20

final_cluster_labels, final_cluster_top_phrases = full_lingo_pipeline(
    df_processed['preprocessed_text'],
    n_components_svd=optimal_n_components_svd_val,
    n_clusters_kmeans=optimal_n_clusters_kmeans_val
)

# Add the final cluster labels back to the original full DataFrame
df_full['final_cluster_label'] = pd.NA
df_full.loc[df_processed.index, 'final_cluster_label'] = final_cluster_labels.astype(int)

print("\n--- Final Results Summary ---")
print("First 5 documents with original text and assigned cluster labels:")
print(df_full[['text', 'final_cluster_label']].head())

print("\nTop phrases for the first 5 clusters:")
for i in range(min(5, len(final_cluster_top_phrases))):
    print(f"Cluster {i}: {', '.join(final_cluster_top_phrases.get(i, []))}")
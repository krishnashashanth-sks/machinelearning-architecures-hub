from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def full_lingo_pipeline(
    preprocessed_text_series, 
    n_components_svd=100, 
    n_clusters_kmeans=20,
    ngram_range=(1, 3),
    max_df=0.95,
    min_df=2
    ):
    
    print("\n--- Starting Lingo Pipeline ---")

    # 1. Document Representation (TF-IDF with N-grams)
    print("Step 1: TF-IDF Vectorization...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_text_series)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # 2. Topic Discovery (Dimensionality Reduction with TruncatedSVD)
    print("Step 2: TruncatedSVD for topic discovery...")
    actual_n_components_svd = n_components_svd
    if tfidf_matrix.shape[1] == 0:
        print("Warning: TF-IDF matrix has no features. Skipping SVD and KMeans.")
        return pd.Series(index=preprocessed_text_series.index), {}
    if tfidf_matrix.shape[1] < actual_n_components_svd:
        actual_n_components_svd = tfidf_matrix.shape[1] - 1 if tfidf_matrix.shape[1] > 1 else 1
    if actual_n_components_svd <= 0:
        actual_n_components_svd = 1 # Use at least 1 component if features exist

    svd_model = TruncatedSVD(n_components=actual_n_components_svd, random_state=42)
    document_topics = svd_model.fit_transform(tfidf_matrix)
    print(f"Document topics matrix shape (after SVD): {document_topics.shape}")

    # 3. Clustering Algorithms (K-Means)
    print("Step 3: K-Means Clustering...")
    actual_n_clusters_kmeans = n_clusters_kmeans
    if len(preprocessed_text_series) == 0: # Ensure there are documents to cluster
        print("Warning: No documents to cluster. Returning empty cluster labels.")
        return pd.Series(index=preprocessed_text_series.index), {}
    if len(preprocessed_text_series) < actual_n_clusters_kmeans:
        actual_n_clusters_kmeans = len(preprocessed_text_series)
    if document_topics.shape[1] < actual_n_clusters_kmeans:
        actual_n_clusters_kmeans = document_topics.shape[1]
    if actual_n_clusters_kmeans < 2 and len(preprocessed_text_series) > 0:
        actual_n_clusters_kmeans = 2 # At least two clusters for meaningful clustering

    kmeans_model = KMeans(n_clusters=actual_n_clusters_kmeans, init='k-means++', max_iter=100, n_init=1, random_state=42)
    kmeans_model.fit(document_topics)
    cluster_labels = pd.Series(kmeans_model.labels_, index=preprocessed_text_series.index)
    print(f"K-Means clustering complete with {actual_n_clusters_kmeans} clusters.")

    # 4. Phrase Extraction for Clusters
    print("Step 4: Phrase Extraction for Clusters...")
    cluster_top_phrases = {}
    for i in range(actual_n_clusters_kmeans):
        cluster_docs_indices = cluster_labels[cluster_labels == i].index
        if len(cluster_docs_indices) == 0:
            cluster_top_phrases[i] = []
            continue

        original_df_processed_indices_map = {idx: i for i, idx in enumerate(preprocessed_text_series.index)}
        tfidf_row_indices_for_cluster = [original_df_processed_indices_map[idx] for idx in cluster_docs_indices if idx in original_df_processed_indices_map]

        if not tfidf_row_indices_for_cluster:
            cluster_top_phrases[i] = []
            continue

        cluster_tfidf_matrix = tfidf_matrix[tfidf_row_indices_for_cluster]
        cluster_tfidf_means = np.array(cluster_tfidf_matrix.mean(axis=0)).flatten()
        num_phrases_to_get = min(19, len(feature_names))
        if num_phrases_to_get == 0:
            cluster_top_phrases[i] = []
            continue

        top_features_indices = cluster_tfidf_means.argsort()[:-num_phrases_to_get-1:-1]
        top_phrases = [feature_names[idx] for idx in top_features_indices]
        cluster_top_phrases[i] = top_phrases
        print(f"Cluster {i}: {', '.join(top_phrases)}")
    
    print("--- Lingo Pipeline Complete ---")
    return cluster_labels, cluster_top_phrases

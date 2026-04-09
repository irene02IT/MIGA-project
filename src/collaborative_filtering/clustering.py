import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run_clustering(rating_matrix: pd.DataFrame, n_clusters=5, plot=True):
    """
    Esegue il clustering degli utenti usando cosine similarity e K-Means.
    Se plot=True, mostra una proiezione 2D dei cluster.
    """

    # Sostituisce eventuali NaN con 0
    rating_matrix_filled = rating_matrix.fillna(0)

    # Calcolo cosine similarity tra utenti
    similarity = cosine_similarity(rating_matrix_filled)

    # Applicazione K-Means sulla matrice di similarità
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(similarity)

    # DataFrame con user_id e cluster assegnato
    clustered_users = pd.DataFrame({
        "user_id": rating_matrix_filled.index,
        "cluster": cluster_labels
    })

    print(f"✅ Clustering completato con {n_clusters} cluster.")
    print(clustered_users['cluster'].value_counts())

    if plot:
        # PCA per ridurre la similarità a 2 dimensioni
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(similarity)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1],
                              c=cluster_labels, cmap='tab10', alpha=0.7)
        plt.title("Visualizzazione dei cluster utenti (PCA su cosine similarity)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(scatter, label="Cluster")
        plt.show()
    return clustered_users


def evaluate_clustering(matrix, max_k=10):
    inertia = []
    silhouette_scores = []
    K = range(2, max_k+1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix.fillna(0))  # gestisce eventuali NaN
        inertia.append(kmeans.inertia_)
        sil_score = silhouette_score(matrix.fillna(0), labels)
        silhouette_scores.append(sil_score)

    # Plot Inertia
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(K, inertia, marker='o')
    plt.title("Metodo del gomito")
    plt.xlabel("Numero di cluster (k)")
    plt.ylabel("Inertia")

    # Plot Silhouette
    plt.subplot(1,2,2)
    plt.plot(K, silhouette_scores, marker='o')
    plt.title("Silhouette Score")
    plt.xlabel("Numero di cluster (k)")
    plt.ylabel("Silhouette")

    plt.show()

    return inertia, silhouette_scores

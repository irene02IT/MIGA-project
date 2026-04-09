from src.data_loading import load_data
from src.collaborative_filtering.collaborative_knn import run_knn, fill_rating_matrix, get_top_k_recommendations
from src.collaborative_filtering.clustering import run_clustering
from src.collaborative_filtering.matrix_factorization import fill_rating_matrix_with_method, compare_matrix_filling_methods

if __name__ == "__main__":
    df = load_data("data/Books.jsonl", 100000)
    filter_user_reviews = 3
    filter_item_reviews = 3
    # Step 1: esplorazione
    #run_exploration(df)

    # Step 2: tuning ottimizzato per predizione del rating con algoritmo knn
    # Trova la configurazione ottimale ù+prima su un sottoinsieme (fornito come parametro della funzione), poi sull'intero dataset
    model, best_config, df_filtered = run_knn(
        df,
        min_user_reviews=filter_user_reviews,
        min_item_reviews=filter_item_reviews
    )

    # Ora model è addestrato sull'intero dataset filtrato
    print("\nModello pronto per il filling della matrice!")

    # Step 3: filling matrice ottimizzato
    rating_matrix = fill_rating_matrix(df, best_config, min_user_reviews=filter_user_reviews, min_item_reviews=filter_item_reviews)
    print(rating_matrix.head())

    # Salvataggio matrice
    rating_matrix.to_csv("outputs/rating_matrix_filled.csv")

    # Step 4: Clustering
    #evaluate_clustering(rating_matrix, max_k=10)

    clustered_users = run_clustering(rating_matrix, n_clusters=3)
    clustered_users.to_csv("outputs/user_clusters.csv", index=False)

    # Genera e salva le raccomandazioni
    recommendations = get_top_k_recommendations(rating_matrix, k=5)
    recommendations.to_csv("outputs/user_recommendations.csv", index=False)

    # Step 3.1: Confronto metodi di filling
    svd_model, knn_model, performance_comparison = compare_matrix_filling_methods(
        df_filtered,
        best_config,
        n_factors=100,
        n_epochs=20
    )

    # Step 3.2: Filling con entrambi i metodi
    rating_matrix_svd = fill_rating_matrix_with_method(df_filtered, svd_model, "SVD")
    rating_matrix_knn = fill_rating_matrix_with_method(df_filtered, knn_model, "KNN")

    # Salvataggio delle matrici
    rating_matrix_svd.to_csv("outputs/rating_matrix_svd.csv")
    rating_matrix_knn.to_csv("outputs/rating_matrix_knn.csv")
    performance_comparison.to_csv("outputs/performance_comparison.csv")
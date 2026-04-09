from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_matrix_filling_methods(df: pd.DataFrame, best_knn_config, n_factors=100, n_epochs=20):
    """
    Confronta KNN e Matrix Factorization per il filling della matrice di rating.

    Args:
        df: DataFrame con colonne ['user_id', 'parent_asin', 'rating']
        best_knn_config: tuple (similarity, user_based, k) per KNN
        n_factors: numero di fattori latenti per SVD
        n_epochs: numero di iterazioni per SVD
    """
    # Preparazione dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "parent_asin", "rating"]], reader)

    # Split train-test
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # 1. Matrix Factorization (SVD)
    print("\n🔄 Training Matrix Factorization...")
    svd = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
    svd.fit(trainset)

    # 2. KNN
    print("\n🔄 Training KNN...")
    sim, user_based, k = best_knn_config
    knn = KNNBasic(k=k, sim_options={"name": sim, "user_based": user_based})
    knn.fit(trainset)

    # Valutazione su test set
    svd_predictions = svd.test(testset)
    knn_predictions = knn.test(testset)

    # Calcolo metriche
    results = {
        "SVD": {
            "RMSE": accuracy.rmse(svd_predictions),
            "MSE": accuracy.mse(svd_predictions),
        },
        "KNN": {
            "RMSE": accuracy.rmse(knn_predictions),
            "MSE": accuracy.mse(knn_predictions),
        }
    }

    # Visualizzazione risultati
    print("\n📊 Confronto delle performance:")
    results_df = pd.DataFrame(results).round(4)
    print(results_df)

    # Visualizzazione grafica
    plt.figure(figsize=(10, 6))
    metrics = ["RMSE", "MSE"]
    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width / 2, [results["SVD"][m] for m in metrics], width, label='SVD')
    plt.bar(x + width / 2, [results["KNN"][m] for m in metrics], width, label='KNN')

    plt.xlabel('Metriche')
    plt.ylabel('Valore')
    plt.title('Confronto Performance SVD vs KNN')
    plt.xticks(x, metrics)
    plt.legend()
    plt.show()

    return svd, knn, results_df


def fill_rating_matrix_with_method(df: pd.DataFrame, method, method_name: str):
    """
    Riempie la matrice di rating usando il metodo specificato (SVD o KNN).

    Args:
        df: DataFrame originale
        method: modello addestrato (SVD o KNN)
        method_name: nome del metodo ('SVD' o 'KNN')
    """
    # Creazione lista per predizioni
    all_users = df["user_id"].unique()
    all_items = df["parent_asin"].unique()

    # Dizionario dei rating noti
    known_ratings = {
        (u, a): r for u, a, r in df[["user_id", "parent_asin", "rating"]].itertuples(index=False)
    }

    filled_data = []
    total = len(all_users) * len(all_items)

    print(f"\n🔄 Filling matrice con {method_name}...")
    for i, user in enumerate(all_users):

        for item in all_items:
            if (user, item) in known_ratings:
                rating = known_ratings[(user, item)]
            else:
                pred = method.predict(user, item).est
                rating = pred

            filled_data.append({
                "user_id": user,
                "parent_asin": item,
                "rating": rating
            })

    # Creazione matrice finale
    filled_df = pd.DataFrame(filled_data)
    rating_matrix = filled_df.pivot(index="user_id", columns="parent_asin", values="rating")

    print(f"✅ Matrice completata con {method_name}!")
    return rating_matrix
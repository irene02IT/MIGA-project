import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate

def run_knn(df: pd.DataFrame,
            min_user_reviews,
            min_item_reviews):
    """
    Esegue il tuning di KNN su un sottoinsieme e riaddestra il modello ottimale
    sull'intero dataset filtrato.

    Args:
        df: DataFrame con ['user_id', 'parent_asin', 'rating']
        min_user_reviews: minimo di recensioni per utente da mantenere
        min_item_reviews: minimo di recensioni per prodotto da mantenere
        tuning_sample_size: righe da usare per tuning
    """
    # 1. Filtro utenti e prodotti poco popolari
    user_counts = df["user_id"].value_counts()
    item_counts = df["parent_asin"].value_counts()
    df_filtered = df[
        df["user_id"].isin(user_counts[user_counts >= min_user_reviews].index) &
        df["parent_asin"].isin(item_counts[item_counts >= min_item_reviews].index)
        ]
    print(f"📊 Dopo filtro: {df_filtered['user_id'].nunique()} utenti, "
          f"{df_filtered['parent_asin'].nunique()} prodotti, {len(df_filtered)} righe.")

    # 2. Testa diverse combinazioni di parametri
    # (Metriche di similarità: coseno e Pearson;
    # Approccio: user-based (user=true) o item-based (user=false);
    # Valori di k: 5, 10, 20)
    df_sample = df_filtered.sample(
        n= len(df_filtered),
        random_state=42
    )

    # Preparazione dataset per Surprise (sul campione preso)
    reader = Reader(rating_scale=(1, 5))
    data_sample = Dataset.load_from_df(df_sample[["user_id", "parent_asin", "rating"]], reader)

    # Parametri da testare: solo quelle supportate
    similarities = ["cosine", "pearson", "msd"]
    user_based_options = [True, False]
    k_values = [5, 10, 20]

    best_score = float("inf")
    best_config = None

    print("\n🔍 Inizio tuning su campione...")

    # per ogni metrica, per ogni aproccio e per ogni valore di k
    for sim in similarities:
        for user_based in user_based_options:
            for k in k_values:
                algo = KNNBasic(k=k, sim_options={"name": sim, "user_based": user_based}) # esegue l'algoritmo knn
                results = cross_validate(algo, data_sample, measures=["RMSE"], cv=3, verbose=False)  #testa l'accuratezza dei risultati con RMSE
                mean_rmse = results["test_rmse"].mean()

                print(f"Sim={sim}, user_based={user_based}, k={k} → RMSE={mean_rmse:.4f}")

                if mean_rmse < best_score:  # se ho trovato una configurazione migliore delle precedenti, lo memorizzo
                    best_score = mean_rmse
                    best_config = (sim, user_based, k)

    print(f"\n✅ Configurazione ottimale trovata: {best_config}, RMSE={best_score:.4f}")

    # 3. Riaddestra il modello con i parametri ottimali sull'intero dataset filtrato.
    print("\n🏋️‍♂️ Riaddestramento modello su dataset filtrato...")
    data_full = Dataset.load_from_df(df_filtered[["user_id", "parent_asin", "rating"]], reader)
    trainset_full = data_full.build_full_trainset()

    final_algo = KNNBasic(k=best_config[2],
                          sim_options={"name": best_config[0], "user_based": best_config[1]})
    final_algo.fit(trainset_full)

    return final_algo, best_config, df_filtered

def _filter_with_min_counts(df: pd.DataFrame, min_user_reviews: int, min_item_reviews: int) -> pd.DataFrame:
    """
    Applica il filtro su utenti e item in base ai conteggi minimi.
    """
    user_counts = df["user_id"].value_counts()
    item_counts = df["parent_asin"].value_counts()
    filtered = df[df["user_id"].isin(user_counts[user_counts >= min_user_reviews].index)]
    filtered = filtered[filtered["parent_asin"].isin(item_counts[item_counts >= min_item_reviews].index)]
    return filtered


def fill_rating_matrix(df: pd.DataFrame, best_config, min_user_reviews, min_item_reviews):
    """
    Riempie in modo ottimizzato la matrice user-item con KNN.
    Riduce il carico filtrando utenti e item poco popolari.

    Args:
        df: DataFrame con colonne ['user_id', 'asin', 'rating']
        best_config: tuple (similarity, user_based, k)
        min_user_reviews: minimo di recensioni per utente
        min_item_reviews: minimo di recensioni per item
    """
    sim, user_based, k = best_config

    # 1) Deduplica eventuali (user, item) multipli usando la media del rating
    df = df.groupby(["user_id", "parent_asin"], as_index=False)["rating"].mean()

    # 2) Applica filtro min user reviews e min item reviews
    cur_user_min = int(min_user_reviews)
    cur_item_min = int(min_item_reviews)

    filtered = _filter_with_min_counts(df, cur_user_min, cur_item_min)

    # se dopo il filtro il risultato è vuoto, rilassa progressivamente
    while (filtered.empty or filtered["user_id"].nunique() == 0 or filtered["parent_asin"].nunique() == 0) and (
            cur_user_min > 1 or cur_item_min > 1):
        # Rilassa gradualmente entrambe le soglie fino ad arrivare a 1
        cur_user_min = max(1, cur_user_min - 1)
        cur_item_min = max(1, cur_item_min - 1)
        filtered = _filter_with_min_counts(df, cur_user_min, cur_item_min)

    # Se ancora vuoto, usa il df originale (senza filtro) per evitare pivot su DF vuoto
    if filtered.empty or filtered["user_id"].nunique() == 0 or filtered["parent_asin"].nunique() == 0:
        print("⚠️ Nessun dato dopo il filtro. Si procede senza filtro per evitare una matrice vuota.")
        filtered = df.copy()
        cur_user_min, cur_item_min = 1, 1

    print(f"Filtri applicati: min_user_reviews={cur_user_min}, min_item_reviews={cur_item_min} "
          f"(utenti={filtered['user_id'].nunique()}, item={filtered['parent_asin'].nunique()}, interazioni={len(filtered)})")

    # 3) Setup per Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(filtered[["user_id", "parent_asin", "rating"]], reader)
    trainset = data.build_full_trainset()

    # 4) Modello KNN con la configurazione ottimale
    algo = KNNBasic(k=k, sim_options={"name": sim, "user_based": user_based})
    algo.fit(trainset)

    # 5) Spazio utenti/item e rating noti
    all_users = filtered["user_id"].unique()
    all_items = filtered["parent_asin"].unique()

    # Dizionario dei rating noti per lookup (più efficiente, tempo O(1))
    known_ratings = {
        (u, a): r for u, a, r in filtered[["user_id", "parent_asin", "rating"]].itertuples(index=False, name=None)
    }

    # 6) Costruzione della matrice completa
    filled_data = []
    for user in all_users:
        for item in all_items:
            if (user, item) in known_ratings:
                filled_data.append({"user_id": user, "parent_asin": item, "rating": known_ratings[(user, item)]})
            else:
                pred = algo.predict(user, item).est
                filled_data.append({"user_id": user, "parent_asin": item, "rating": pred})

    # 7) DataFrame finale e pivot
    filled_df = pd.DataFrame(filled_data)

    # A questo punto filled_df non è vuoto, ma per sicurezza controlliamo
    if filled_df.empty or "user_id" not in filled_df.columns or "parent_asin" not in filled_df.columns:
        raise ValueError("Impossibile creare la matrice di rating perché i dati risultano vuoti dopo il riempimento.")

    rating_matrix = filled_df.pivot(index="user_id", columns="parent_asin", values="rating")
    print(f"✅ Matrice completata con {len(all_users)} utenti e {len(all_items)} prodotti.")
    return rating_matrix


def get_top_k_recommendations(rating_matrix: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Genera i top-k item raccomandati per ogni utente basandosi sui rating predetti.

    Args:
        rating_matrix: Matrice di rating user-item completata
        k: Numero di item da raccomandare per ogni utente

    Returns:
        DataFrame con le raccomandazioni per ogni utente
    """
    # Creiamo una lista per salvare tutte le raccomandazioni
    recommendations = []

    # Per ogni utente nella matrice
    for user_id in rating_matrix.index:
        # Prendiamo i rating dell'utente e li ordiniamo in modo decrescente
        user_ratings = rating_matrix.loc[user_id].sort_values(ascending=False)

        # Prendiamo i primi k item
        top_k_items = user_ratings.head(k)

        # Aggiungiamo le raccomandazioni alla lista
        recommendations.append({
            'user_id': user_id,
            'recommended_items': top_k_items.index.tolist(),
            'predicted_ratings': top_k_items.values.tolist()
        })

    # Convertiamo in DataFrame
    recommendations_df = pd.DataFrame(recommendations)

    return recommendations_df

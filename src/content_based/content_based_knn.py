from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def recommend_content_based(user_profile, item_embeddings, item_ids, top_k=10):
    """
    Raccomandazioni content-based con predizione rating.

    Args:
        user_profile: indici degli item apprezzati dall'utente
        item_embeddings: matrice embeddings (TF-IDF o Transformer)
        item_ids: lista degli asin
        top_k: numero di raccomandazioni

    Returns:
        recommendations: lista di asin raccomandati
        predicted_ratings: lista di rating previsti (scala 1-5)
    """
    if not user_profile:
        return [], []

    if not isinstance(item_embeddings, np.ndarray):
        item_embeddings = item_embeddings.toarray()

    # Profilo utente = media embedding degli item apprezzati
    liked_embeddings = item_embeddings[user_profile]
    user_vector = np.mean(liked_embeddings, axis=0).reshape(1, -1)

    # Similarità coseno
    similarities = cosine_similarity(user_vector, item_embeddings).flatten()

    # Ordina gli indici per similarità
    ranked_idx = similarities.argsort()[::-1]

    # Escludi gli item già apprezzati
    ranked_idx = [i for i in ranked_idx if i not in user_profile]

    # Normalizza similarity in scala [1, 5] per simulare rating previsti
    sims = similarities[ranked_idx[:top_k]]
    sims_norm = 1 + 4 * (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    recommendations = [item_ids[i] for i in ranked_idx[:top_k]]

    return recommendations, sims_norm.tolist()

def generate_recommendations(
    reviews,
    metadata,
    X_tfidf,
    X_transformer,
    build_user_profile,
    recommend_content_based,
    evaluate_content_based,
    k: int = 20,
    output_path: str = "outputs/recommendations_with_evaluation.csv",
):
    """
    Genera raccomandazioni per ogni utente e salva i risultati su CSV.

    Parametri:
    - reviews: DataFrame con le recensioni, deve includere 'user_id'
    - metadata: DataFrame con almeno la colonna 'asin'
    - X_tfidf: matrice feature (TF-IDF) allineata a metadata["asin"]
    - X_transformer: matrice feature (Transformer) allineata a metadata["asin"]
    - build_user_profile: funzione (group, metadata) -> (user_profile_idx, relevant_items)
    - recommend_content_based: funzione (user_profile_idx, X, item_ids, top_k) -> (rec_ids, scores)
    - evaluate_content_based: funzione (rec_ids, relevant_items, k) -> dict metriche
    - k: cut-off per ranking e metriche
    - output_path: percorso file CSV di output

    Ritorna:
    - DataFrame con le raccomandazioni e metriche per utente.
    """
    import pandas as pd

    all_recommendations = []

    for uid, group in reviews.groupby("user_id"):
        user_profile_idx, relevant_items = build_user_profile(group, metadata)

        if not user_profile_idx or not relevant_items:
            continue  # salta utenti senza overlap

        # TF-IDF
        rec_tfidf, _ = recommend_content_based(
            user_profile_idx, X_tfidf, metadata["asin"], top_k=k
        )
        eval_tfidf = evaluate_content_based(rec_tfidf, relevant_items, k=k)

        # Transformer
        rec_transformer, _ = recommend_content_based(
            user_profile_idx, X_transformer, metadata["asin"], top_k=k
        )
        eval_transformer = evaluate_content_based(rec_transformer, relevant_items, k=k)

        all_recommendations.append({
            "user_id": uid,
            "rec_tfidf": ",".join(rec_tfidf),
            "precision_tfidf": eval_tfidf.get("precision@k"),
            "recall_tfidf": eval_tfidf.get("recall@k"),
            "map_tfidf": eval_tfidf.get("MAP@k"),
            "rec_transformer": ",".join(rec_transformer),
            "precision_transformer": eval_transformer.get("precision@k"),
            "recall_transformer": eval_transformer.get("recall@k"),
            "map_transformer": eval_transformer.get("MAP@k"),
            "relevant_items": ",".join(relevant_items)
        })

    df_recs = pd.DataFrame(all_recommendations)
    df_recs.to_csv(output_path, index=False)
    print(f"Raccomandazioni salvate su {output_path}")
    return df_recs


def summarize_recommendation_metrics(
    df_recs,
    model_suffixes=("tfidf", "transformer"),
    metrics_prefixes=("precision", "recall", "map", "ndcg", "hitrate"),
    output_path: str = "outputs/",
):
    """
    Elabora e riassume le metriche, considerando solo gli utenti con metriche non nulle.

    Parametri:
    - df_recs: DataFrame generato da generate_recommendations (colonne come precision_tfidf, ...).
    - model_suffixes: suffissi dei modelli presenti nelle colonne (es. 'tfidf', 'transformer').
    - metrics_prefixes: prefissi delle metriche attese (quelle presenti verranno usate).
    - output_path: se valorizzato, salva un CSV del riepilogo.

    Ritorna:
    - dict con:
        {
          "per_model": {
             "<model>": {
                "<metric>": {"mean": float, "std": float, "median": float, "count": int}
             }, ...
          },
          "comparison": {
             "<metric>": {
                "both_n": int,
                "win_rate_transformer_over_tfidf": float,  # se i due modelli sono presenti
                "mean_diff(transformer - tfidf)": float
             }, ...
          }
        }
    """
    import pandas as pd
    import numpy as np

    df = df_recs.copy()

    # Scopri quali metriche sono effettivamente presenti per ogni modello
    present_by_model = {}
    for model in model_suffixes:
        cols = []
        for mp in metrics_prefixes:
            col = f"{mp}_{model}"
            if col in df.columns:
                cols.append(col)
        present_by_model[model] = cols

    # Riassunto per modello: filtra righe con metriche non nulle (tutte quelle presenti per il modello)
    per_model = {}
    for model, cols in present_by_model.items():
        if not cols:
            continue
        valid = df.dropna(subset=cols)
        model_summary = {}
        for col in cols:
            metric = col.rsplit("_", 1)[0]  # rimuove il suffisso modello
            s = valid[col]
            model_summary[metric] = {
                "mean": float(np.mean(s)) if not s.empty else np.nan,
                "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0 if len(s) == 1 else np.nan,
                "median": float(np.median(s)) if not s.empty else np.nan,
                "count": int(len(s)),
            }
        per_model[model] = model_summary

    # Confronto tra modelli dove le metriche sono in comune
    comparison = {}
    if len(model_suffixes) >= 2:
        base, other = model_suffixes[0], model_suffixes[1]
        common_metrics = set(m.rsplit("_", 1)[0] for m in present_by_model.get(base, [])) & \
                         set(m.rsplit("_", 1)[0] for m in present_by_model.get(other, []))
        for metric in sorted(common_metrics):
            col_b = f"{metric}_{base}"
            col_o = f"{metric}_{other}"
            mask = df[col_b].notna() & df[col_o].notna()
            both = df.loc[mask, [col_b, col_o]]
            if both.empty:
                comparison[metric] = {"both_n": 0, f"win_rate_{other}_over_{base}": np.nan, f"mean_diff({other} - {base})": np.nan}
                continue
            wins = (both[col_o] > both[col_b]).mean()
            mean_diff = float((both[col_o] - both[col_b]).mean())
            comparison[metric] = {
                "both_n": int(len(both)),
                f"win_rate_{other}_over_{base}": float(wins),
                f"mean_diff({other} - {base})": mean_diff,
            }

    result = {"per_model": per_model, "comparison": comparison}

    # Opzionale: salva CSV con tabella leggibile
    if output_path:
        rows = []
        for model, metrics in per_model.items():
            for metric, stats in metrics.items():
                rows.append({
                    "model": model,
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "median": stats["median"],
                    "count": stats["count"],
                })
        comp_rows = []
        for metric, stats in comparison.items():
            row = {"metric": metric}
            row.update(stats)
            comp_rows.append(row)

        df_summary = pd.DataFrame(rows)
        df_comparison = pd.DataFrame(comp_rows)

        # Salva due fogli separati se possibile, altrimenti due CSV
        if output_path.lower().endswith(".xlsx"):
            with pd.ExcelWriter(output_path) as xw:
                df_summary.to_excel(xw, sheet_name="per_model", index=False)
                df_comparison.to_excel(xw, sheet_name="comparison", index=False)
        else:
            # salva come due CSV con suffissi
            base_path = output_path.rsplit(".", 1)[0]
            df_summary.to_csv(f"{base_path}_per_model.csv", index=False)
            df_comparison.to_csv(f"{base_path}_comparison.csv", index=False)

    return result

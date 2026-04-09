import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_metric_distributions(
    df_recs,
    metrics=("precision", "recall", "map"),
    models=("tfidf", "transformer"),
    save_dir: str = "outputs/",
    dpi: int = 150,
    show: bool = False,
    kde: bool = True,
    bins: int = 40,
):
    """
    Grafica le distribuzioni per-utente delle metriche, per ciascun modello.
    Considera solo gli utenti con metriche non nulle (dropna per metrica-modello).
    """

    os.makedirs(save_dir, exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(7, 4))
        plotted_any = False
        for model in models:
            col = f"{metric}_{model}"
            if col not in df_recs.columns:
                continue
            s = df_recs[col].dropna()
            if s.empty:
                continue
            plotted_any = True
            if kde:
                sns.kdeplot(s, label=model, fill=True, alpha=0.3)
            else:
                sns.histplot(s, label=model, bins=bins, element="step", stat="density", fill=False)
        if not plotted_any:
            plt.close()
            continue
        plt.title(f"Distribuzione {metric} (utenti con valori non nulli)")
        plt.xlabel(metric)
        plt.ylabel("Densità")
        plt.legend(title="Modello")
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"dist_{metric}.png")
        plt.savefig(out_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()


def plot_cb_comparisons(
    metadata,
    rec_tfidf,
    pred_ratings_tfidf,
    rec_transformer,
    pred_ratings_transformer,
    relevant_items,
    eval_tfidf: dict | None = None,
    eval_transformer: dict | None = None,
    top_k_plot: int = 10,
    save_dir: str = "outputs/plots",
    show: bool = False,
    dpi: int = 150,
):
    """
    Genera i 4 grafici richiesti per il confronto Content-Based:
      1) Confronto hit su item rilevanti
      2) Distribuzione dei punteggi previsti
      3) Confronto metriche di valutazione
      4) Top‑10 raccomandazioni e punteggi previsti

    Parametri principali:
    - metadata: DataFrame con almeno colonne ['asin','title'] per label nei grafici top‑K.
    - rec_tfidf, rec_transformer: liste di ASIN raccomandati.
    - pred_ratings_tfidf, pred_ratings_transformer: liste di punteggi previsti (stessa lunghezza delle raccomandazioni).
    - relevant_items: lista/set di ASIN realmente rilevanti (ground truth).
    - eval_tfidf, eval_transformer: dict metriche (opzionale ma richiesto per il grafico 3).
    - top_k_plot: quanti elementi visualizzare nel grafico Top‑10 (puoi variare il numero).
    - save_dir: cartella di destinazione dei grafici.
    - show: se True, mostra anche a video.
    - dpi: risoluzione salvataggio.

    Ritorna:
    - dict con i path dei file salvati.
    """


    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="whitegrid")

    def map_asin_to_title(asins, df):
        if df is None or "asin" not in df.columns or "title" not in df.columns:
            return asins
        sub = df[df["asin"].isin(asins)][["asin", "title"]].drop_duplicates()
        title_map = dict(zip(sub["asin"], sub["title"].fillna("").astype(str)))
        labels = []
        for a in asins:
            t = title_map.get(a, "").strip()
            labels.append(t if t else a)
        return labels

    # 1) Confronto hit su item rilevanti
    relevant_set = set(relevant_items) if not isinstance(relevant_items, set) else relevant_items
    hits_tfidf = len(set(rec_tfidf) & relevant_set)
    hits_trans = len(set(rec_transformer) & relevant_set)
    tot_relevant = len(relevant_set)

    plt.figure(figsize=(8, 5))
    labels = ["Rilevanti (tot)", "Hit TF-IDF", "Hit Transformer"]
    values = [tot_relevant, hits_tfidf, hits_trans]
    palette = ["#72B7B2", "#4C78A8", "#F58518"]
    ax = sns.barplot(x=labels, y=values, palette=palette)
    ax.set_title("Confronto hit su item rilevanti")
    ax.set_ylabel("Conteggio")
    ax.set_xlabel("")
    vmax = max(values) if values else 0
    for i, v in enumerate(values):
        ax.text(i, v + (0.02 * vmax if vmax > 0 else 0.05), str(v), ha="center")
    plt.tight_layout()
    path_hits = os.path.join(save_dir, "01_confronto_hits_rilevanti.png")
    plt.savefig(path_hits, dpi=dpi)
    if show:
        plt.show()
    plt.close()

    # 2) Distribuzione dei punteggi previsti
    plt.figure(figsize=(10, 5))
    sns.histplot(pred_ratings_tfidf, color="#4C78A8", kde=True, stat="density", label="TF-IDF", bins=20)
    sns.histplot(pred_ratings_transformer, color="#F58518", kde=True, stat="density", label="Transformer", bins=20, alpha=0.6)
    plt.title("Distribuzione dei punteggi previsti")
    plt.xlabel("Punteggio previsto")
    plt.ylabel("Densità")
    plt.legend()
    plt.tight_layout()
    path_dist = os.path.join(save_dir, "02_distribuzione_punteggi_previsti.png")
    plt.savefig(path_dist, dpi=dpi)
    if show:
        plt.show()
    plt.close()

    # 3) Confronto metriche di valutazione (solo se i dizionari sono forniti)
    path_eval = None
    if isinstance(eval_tfidf, dict) and isinstance(eval_transformer, dict):
        common_keys = [k for k in eval_tfidf.keys()
                       if k in eval_transformer
                       and isinstance(eval_tfidf[k], (int, float))
                       and isinstance(eval_transformer[k], (int, float))]
        if common_keys:
            x = range(len(common_keys))
            width = 0.38
            plt.figure(figsize=(10, 5))
            plt.bar([i - width/2 for i in x], [eval_tfidf[k] for k in common_keys], width, label="TF-IDF", color="#4C78A8")
            plt.bar([i + width/2 for i in x], [eval_transformer[k] for k in common_keys], width, label="Transformer", color="#F58518")
            plt.xticks(list(x), common_keys)
            plt.ylabel("Valore")
            plt.title("Confronto metriche di valutazione")
            plt.legend()
            plt.tight_layout()
            path_eval = os.path.join(save_dir, "03_confronto_metriche_valutazione.png")
            plt.savefig(path_eval, dpi=dpi)
            if show:
                plt.show()
            plt.close()

    # 4) Top‑10 raccomandazioni e punteggi previsti (due subplot affiancati)
    tfidf_top_asins = rec_tfidf[:top_k_plot]
    tfidf_top_scores = pred_ratings_tfidf[:top_k_plot]
    tfidf_top_labels = map_asin_to_title(tfidf_top_asins, metadata)

    trans_top_asins = rec_transformer[:top_k_plot]
    trans_top_scores = pred_ratings_transformer[:top_k_plot]
    trans_top_labels = map_asin_to_title(trans_top_asins, metadata)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    sns.barplot(ax=axes[0], x=tfidf_top_scores, y=tfidf_top_labels, color="#4C78A8")
    axes[0].set_title(f"Top-{top_k_plot} raccomandazioni (TF-IDF)")
    axes[0].set_xlabel("Punteggio previsto")
    axes[0].set_ylabel("Item")

    sns.barplot(ax=axes[1], x=trans_top_scores, y=trans_top_labels, color="#F58518")
    axes[1].set_title(f"Top-{top_k_plot} raccomandazioni (Transformer)")
    axes[1].set_xlabel("Punteggio previsto")
    axes[1].set_ylabel("")
    plt.tight_layout()
    path_topk = os.path.join(save_dir, "04_topk_raccomandazioni_punteggi.png")
    plt.savefig(path_topk, dpi=dpi)
    if show:
        plt.show()
    plt.close()

    return {
        "confronto_hits": path_hits,
        "distribuzione_punteggi": path_dist,
        "confronto_metriche": path_eval,
        "topk_raccomandazioni": path_topk,
    }


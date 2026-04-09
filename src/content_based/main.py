from src.content_based.text_preprocessing import preprocess_text
from src.content_based.embeddings import compute_tfidf_embeddings, compute_transformer_embeddings
from src.content_based.content_based_knn import recommend_content_based, generate_recommendations, summarize_recommendation_metrics
from src.data_loading import load_metadata, load_data
from src.content_based.evaluation_cb import evaluate_content_based, build_user_profile, get_valid_user
from src.content_based.visualisation import plot_cb_comparisons

# Carico un subset più grande per avere più probabilità di overlap
metadata = load_metadata("data/meta_Books.jsonl", 50000)
reviews = load_data("data/Books.jsonl", 50000)

asin_in_reviews = set(reviews["asin"].unique())
metadata = metadata[metadata["asin"].isin(asin_in_reviews) | metadata["parent_asin"].isin(asin_in_reviews)]
metadata = metadata.reset_index(drop=True)

# Preprocessa title + description + features
metadata["processed"] = (metadata["title"].fillna("") + " " + metadata["description"].fillna("")+" " + metadata["features"].fillna(""))
metadata["processed"] = metadata["processed"].apply(preprocess_text)

#trova utenti con overlaps
users_with_overlaps = []
for uid, group in reviews.groupby("user_id"): # per ogni utente
    liked = group[group["rating"] >= 4]["asin"].tolist() # item apprezzati
    ov = set(liked) & set(metadata["asin"]) # overlap tra item apprezzati e item presenti nel metadata
    if ov:
        users_with_overlaps.append((uid, len(ov)))     #se l'overlap è stato identificato, aggiungi l'utente alla lista

# Trova un utente valido (che abbia overlap con metadata): i dati di evaluation si baseranno
# su questo caso specifico e verranno utilizzati per valutare la performance del modello
user_id, liked_asins, overlap = get_valid_user(reviews, metadata)
print("Utente scelto:", user_id)
print("Item apprezzati:", liked_asins)
print("Overlap con metadata:", overlap)

# Costruisci profilo utente sugli item presenti in metadata
user_profile_idx, relevant_items = build_user_profile(
    reviews[reviews["user_id"] == user_id], metadata
)

# ---- embedding TF-IDF ----
X_tfidf, _ = compute_tfidf_embeddings(metadata["processed"])

# ---- Raccomandazioni  ----
rec_tfidf, pred_ratings_tfidf = recommend_content_based(
    user_profile_idx, X_tfidf, metadata["asin"], top_k=100
)

# ----embedding Transformer  ----
texts = metadata["processed"].fillna("").astype(str).tolist()
X_transformer = compute_transformer_embeddings(texts)

# ---- Raccomandazioni  ----
rec_transformer, pred_ratings_transformer = recommend_content_based(
    user_profile_idx, X_transformer, metadata["asin"], top_k=100
)

# ---- Valutazione ----
eval_tfidf = evaluate_content_based(rec_tfidf, relevant_items, k=100)
eval_transformer = evaluate_content_based(rec_transformer, relevant_items, k=100)


#------- per TUTTI gli utenti --------#
# ----genera raccomandazioni per ogni utente e riassume i risultati ----
rec_per_user=generate_recommendations(reviews, metadata, X_tfidf, X_transformer, build_user_profile, recommend_content_based, evaluate_content_based)
summary = summarize_recommendation_metrics(rec_per_user)

paths = plot_cb_comparisons(
    metadata=metadata,
    rec_tfidf=rec_tfidf,
    pred_ratings_tfidf=pred_ratings_tfidf,
    rec_transformer=rec_transformer,
    pred_ratings_transformer=pred_ratings_transformer,
    relevant_items=relevant_items,
    eval_tfidf=eval_tfidf,
    eval_transformer=eval_transformer,
    top_k_plot=10,
    save_dir="outputs/plots",
    show=False
)
print("generati i seguenti file: ", ", ".join(paths))
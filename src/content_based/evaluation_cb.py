from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def precision_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return len(set(rec_k) & set(relevant)) / float(k)


def recall_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / float(len(set(relevant)))

def mean_average_precision(recommended: List[str], relevant: List[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    num_hits = 0.0
    rel_set = set(relevant)
    for i, rec in enumerate(recommended[:k], start=1):
        if rec in rel_set:
            num_hits += 1.0
            score += num_hits / i
    return score / float(min(len(rel_set), k))

def ndcg_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
    """
    nDCG binaria: 1 se l'item è rilevante, 0 altrimenti.
    """
    if not relevant or k <= 0:
        return 0.0
    rel_set = set(relevant)
    rec_k = recommended[:k]
    gains = [1.0 if r in rel_set else 0.0 for r in rec_k]
    if not any(gains):
        return 0.0
    discounts = [1.0 / np.log2(i + 2) for i in range(len(rec_k))]
    dcg = float(np.sum(np.array(gains) * np.array(discounts)))

    ideal_gains = [1.0] * int(min(len(rel_set), k))
    ideal_discounts = discounts[:len(ideal_gains)]
    idcg = float(np.sum(np.array(ideal_gains) * np.array(ideal_discounts)))
    return dcg / (idcg + 1e-12)

def hit_rate_at_k(recommended: List[str], relevant: List[str], k: int = 10) -> float:
    """
    1.0 se almeno un rilevante appare nella top-k, altrimenti 0.0.
    """
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if len(set(recommended[:k]) & set(relevant)) > 0 else 0.0

def evaluate_content_based(recommended: List[str], relevant: List[str], k: int = 10) -> Dict[str, float]:
    k = max(1, min(k, len(recommended)))  # evita k > len(recommended) o k <= 0
    # filtra relevant a duplicati/ripetizioni (difensivo)
    relevant = list(dict.fromkeys(relevant))
    return {
        "precision@k": precision_at_k(recommended, relevant, k),
        "recall@k": recall_at_k(recommended, relevant, k),
        "MAP@k": mean_average_precision(recommended, relevant, k),
        "nDCG@k": ndcg_at_k(recommended, relevant, k),
        "HitRate@k": hit_rate_at_k(recommended, relevant, k),
    }



def _ensure_item_id_column(metadata: pd.DataFrame) -> pd.DataFrame:
    """Crea/garantisce la colonna item_id coerente in metadata."""
    if "item_id" not in metadata.columns:
        metadata = metadata.copy()
        metadata["item_id"] = metadata["asin"].fillna(metadata["parent_asin"])
    return metadata


def get_valid_user(
    reviews: pd.DataFrame,
    metadata: pd.DataFrame,
    min_rating: int = 4,
    min_overlap: int = 3
) -> Tuple[str, list, list]:
    """
    Restituisce:
        user_id
        liked_asins (tutti quelli con rating >= min_rating, come appaiono nelle review)
        overlap (solo quelli presenti in metadata.item_id)
    """
    md = _ensure_item_id_column(metadata)
    md_ids = set(md["item_id"].dropna().astype(str))

    for uid, group in reviews.groupby("user_id"):
        liked_asins = group.loc[group["rating"] >= min_rating, "asin"].dropna().astype(str).tolist()
        overlap = list(set(liked_asins) & md_ids)
        if len(overlap) >= min_overlap:
            return uid, liked_asins, overlap

    return None, [], []


def build_user_profile(
    user_reviews: pd.DataFrame,
    metadata: pd.DataFrame,
    min_rating: int = 4,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[list, list]:
    """
    Costruisce un profilo con split train/test.

    Ritorna:
        train_idx -> indici nel metadata (righe) degli item usati per costruire il profilo
        test_ids  -> lista di item_id (non indici) da usare come relevant per l'evaluation
    """
    md = _ensure_item_id_column(metadata)

    liked_asins = user_reviews.loc[user_reviews["rating"] >= min_rating, "asin"].dropna().astype(str).tolist()
    if not liked_asins:
        return [], []

    # mappa ai soli item presenti in metadata
    md_ids = md["item_id"].astype(str)
    present_mask = md_ids.isin(liked_asins)
    present_idx = md.index[present_mask].to_numpy()

    if len(present_idx) < 2:
        # troppo pochi per uno split robusto
        return present_idx.tolist(), md.loc[present_idx, "item_id"].astype(str).tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(present_idx)

    split = max(1, int(len(present_idx) * (1.0 - test_ratio)))
    train_idx = present_idx[:split]
    test_idx = present_idx[split:]

    test_ids = md.loc[test_idx, "item_id"].astype(str).tolist()

    return train_idx.tolist(), test_ids


import pandas as pd

def load_data(path: str, nrows: int) -> pd.DataFrame:

    df = pd.read_json(path, lines=True, nrows=nrows) #carico solo un sottoinisieme del dataset per evitare SIGKILL


    cols_needed = ["user_id", "parent_asin", "asin", "rating", "helpful_vote", "timestamp"]
    df = df[cols_needed].dropna()
    return df

def load_metadata(path: str, nrows: int) -> pd.DataFrame:
    """
    Carica Item Metadata in modo robusto.
    """
    df = pd.read_json(path, lines=True, nrows=nrows)

    print("Colonne disponibili:", df.columns.tolist())  # DEBUG

    # Se 'asin' non esiste, prova a sostituire con 'parent_asin'
    if "asin" not in df.columns and "parent_asin" in df.columns:
        df["asin"] = df["parent_asin"]

    # Tieni solo le colonne disponibili tra quelle che servono
    keep_cols = [
        "asin", "main_category", "title", "average_rating", "rating_number",
        "features", "description", "price", "store", "categories", "details", "parent_asin"
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]

    df = df[existing_cols]

    # Normalizza description e features
    if "description" in df.columns:
        df["description"] = df["description"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    if "features" in df.columns:
        df["features"] = df["features"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

    # Campo testuale unico
    df["text_content"] = (
        df["title"].fillna("") + " " +
        (df["description"] if "description" in df else "") + " " +
        (df["features"] if "features" in df else "")
    )

    return df
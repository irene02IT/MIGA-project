import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loading import load_data

def run_exploration(df: pd.DataFrame):
    """
    Analisi esplorativa
    - :Distribuzione temporale delle recensioni
    - Statistiche rating
    - Distribuzione rating
    - Distribuzione recensioni per utente e per prodotto
    - Analisi helpful votes
    """

    # Distribuzione temporale delle recensioni
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    reviews_per_month = df.groupby(df['timestamp'].dt.to_period('M')).size()

    reviews_per_month.plot(kind='line', figsize=(10, 4))
    plt.title("Andamento recensioni nel tempo")
    plt.ylabel("Numero recensioni")
    plt.xlabel("Mese")
    plt.show()
    print("\n=== STATISTICHE DI BASE ===")
    print(df.describe())

    # Controllo valori null
    null_counts = df.isnull().sum()
    print("\nValori null per colonna:")
    print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "Nessun valore nullo")

    # Distribuzione rating
    plt.figure(figsize=(6, 4))

    sns.countplot(
        x="rating",
        data=df,
        hue="rating",
        palette="viridis",
        legend=False,  # evita una legenda superflua
    )
    plt.title("Distribuzione dei rating")
    plt.show()

    # Numero utenti e prodotti unici
    num_users = df["user_id"].nunique()
    num_items = df["parent_asin"].nunique()
    print(f"\nNumero utenti unici: {num_users}")
    print(f"Numero prodotti unici (parent_asin): {num_items}")

    # Recensioni per utente
    reviews_per_user = df["user_id"].value_counts()
    plt.figure(figsize=(6, 4))
    sns.histplot(reviews_per_user, bins=50, log_scale=(True, False))
    plt.title("Distribuzione recensioni per utente")
    plt.xlabel("Numero recensioni")
    plt.ylabel("Frequenza")
    plt.show()
    print(f"Media recensioni per utente: {reviews_per_user.mean():.2f}")
    print(f"Utenti con 1 sola recensione: {(reviews_per_user == 1).sum()}")

    # Recensioni per prodotto
    reviews_per_item = df["parent_asin"].value_counts()
    plt.figure(figsize=(6, 4))
    sns.histplot(reviews_per_item, bins=50, log_scale=(True, False))
    plt.title("Distribuzione recensioni per prodotto")
    plt.xlabel("Numero recensioni")
    plt.ylabel("Frequenza")
    plt.show()
    print(f"Prodotti con 1 sola recensione: {(reviews_per_item == 1).sum()}")

    # Top utenti e prodotti
    print("\nTop 10 utenti più attivi:")
    print(reviews_per_user.head(10))
    print("\nTop 10 prodotti più recensiti:")
    print(reviews_per_item.head(10))

    # Analisi helpful votes
    helpful_votes_analysis(df, high_threshold=5)

    sns.boxplot(x='rating', y='helpful_vote', data=df)
    plt.yscale('log')
    plt.title("Distribuzione helpful votes per rating")
    plt.show()

    print("\n=== CORRELAZIONI ===")
    pearson_corr = df['rating'].corr(df['helpful_vote'], method='pearson')
    spearman_corr = df['rating'].corr(df['helpful_vote'], method='spearman')
    print(f"Corr(rating, helpful_vote): Pearson = {pearson_corr:.2f}, Spearman = {spearman_corr:.2f}")

def helpful_votes_analysis(df, high_threshold=5):
    """
    Analizza la distribuzione degli helpful votes e confronta i rating medi
    tra recensioni molto utili e poco utili.
    """
    # Normalizza nome colonna
    if 'helpful_votes' in df.columns:
        col = 'helpful_votes'
    elif 'helpful_vote' in df.columns:
        col = 'helpful_vote'
    else:
        print("⚠️ Nessuna colonna helpful_votes trovata.")
        return

    if 'rating' not in df.columns:
        print("⚠️ Colonna 'rating' non trovata.")
        return

    print("\n=== ANALISI HELPFUL VOTES ===")
    zero_helpful_pct = (df[col] == 0).mean() * 100
    print(f"Percentuale recensioni con 0 helpful votes: {zero_helpful_pct:.2f}%")

    # Distribuzione helpful votes
    plt.figure(figsize=(8, 4))
    plt.hist(df[col], bins=50, color='skyblue', edgecolor='black')
    plt.yscale('log')
    plt.xlabel('Helpful votes per recensione')
    plt.ylabel('Numero recensioni (scala log)')
    plt.title('Distribuzione helpful votes')
    plt.tight_layout()
    plt.show()

    # Confronto rating medio
    avg_high = df.loc[df[col] >= high_threshold, 'rating'].mean()
    avg_low = df.loc[df[col] <= 1, 'rating'].mean()
    print(f"Rating medio recensioni con >= {high_threshold} helpful votes: {avg_high:.2f}")
    print(f"Rating medio recensioni con <= 1 helpful vote: {avg_low:.2f}")
    print(f"Differenza media: {avg_high - avg_low:.2f}")


if __name__ == "__main__":
    df = load_data("data/Books.jsonl", nrows=1000000)
    run_exploration(df)

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

stop_words = set(ENGLISH_STOP_WORDS)
stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    """
    Pulizia del testo:
    - minuscole
    - rimozione caratteri non alfabetici
    - stopwords
    - stemming (al posto della lemmatizzazione per evitare dipendenze da corpora)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]

    return " ".join(tokens)
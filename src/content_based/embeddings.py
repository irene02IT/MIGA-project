from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def compute_tfidf_embeddings(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def compute_transformer_embeddings(texts, model_name="distilbert-base-uncased", batch_size=16, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(texts)-1) // batch_size + 1}")
        encodings = tokenizer(batch,
                              padding=True,
                              truncation=True,
                              max_length=max_length,
                              return_tensors="pt")

        with torch.no_grad():
            outputs = model(**encodings)

        # media dei token embeddings
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


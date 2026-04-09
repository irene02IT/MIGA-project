# 🇬🇧 Project on Computational Methods for Business Management Course (Università di Milano Bicocca)

Development of a recommender system based
on collaborative filtering and content-based filtering using data from the Amazon Reviews 2023 dataset (Books subcategory).

## Project Structure
```
src/
│
├── collaborative_filtering/
│   ├── __init__.py
│   ├── collaborative_knn.py
│   ├── clustering.py
│   ├── data_loading.py
│   ├── exploratory_analysis.py
│   ├── main.py
│   ├── matrix_factorization.py
│
├── content_based/
│   ├── __init__.py
│   ├── text_preprocessing.py
│   ├── embeddings.py
│   ├── content_based_knn.py
│   ├── evaluation_cb.py
│   ├── main.py
│   ├── visualisation.py
│
data/
│   ├── Books.jsonl
│   ├── meta_Books.jsonl
│
outputs/
    ├── *** model outputs ***
```
---

## Requirements

- Required Python version: Python 3.12
  To check:
```
  python --version
```

### Recommended Solution
1. Create a virtualenv with Python 3.12:
```
   python3.12 -m venv .venv
```

2. Activate it:
   - macOS/Linux:
   ```
     source .venv/bin/activate
   ```
   - Windows PowerShell:
   ```
     .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:
```
   pip install -r requirements.txt
```

---

## Running the project

1. Download the Books.jsonl and meta_Books.jsonl datasets from:
   https://amazon-reviews-2023.github.io
   Place them in the data/ folder

2. Run the main modules from the terminal:

   (1) Exploratory Analysis (EA)
   ```
       python -m src.collaborative_filtering.exploratory_analysis
   ```

   (2) Collaborative Filtering (CF) Model
   ```bash
       python -m src.collaborative_filtering.main
   ```
   (3) Content-Based (CB) Model
   ```
       python -m src.content_based.main
   ```
---

# 🇮🇹 Progetto del corso di Metodi informatici per la gestione aziendale (Università di Milano Bicocca)

Sviluppo di un Recommender system basato su Collaborative Filtering e content-based a partire da dati contenuti nel dataset Amazon Reviews 2023 (sottocategoria Books).

## Struttura del progetto
```
src/
│
├── collaborative_filtering/
│   ├── __init__.py
│   ├── collaborative_knn.py
│   ├── clustering.py
│   ├── data_loading.py
│   ├── exploratory_analysis.py
│   ├── main.py
│   ├── matrix_factorization.py
│
├── content_based/
│   ├── __init__.py
│   ├── text_preprocessing.py
│   ├── embeddings.py
│   ├── content_based_knn.py
│   ├── evaluation_cb.py
│   ├── main.py
│   ├── visualisation.py
│
data/
│   ├── Books.jsonl
│   ├── meta_Books.jsonl
│
outputs/
    ├── *** outputs dei modelli ***
```
---

## Requisiti

- Versione Python necessaria: Python 3.12
  Per verificarla:
  python --version

### Soluzione consigliata
1. Creare un virtualenv con Python 3.12:
```
   python3.12 -m venv .venv
```

2. Attivarlo:
   - macOS/Linux:
     source .venv/bin/activate
   - Windows PowerShell:
     .venv\Scripts\Activate.ps1

3. Installare le dipendenze:
```
   pip install -r requirements.txt
```
---

## Esecuzione del progetto

1. Scaricare i dataset Books.jsonl e meta_Books.jsonl da:
   https://amazon-reviews-2023.github.io
   Inserirli nella cartella data/

2. Eseguire i moduli principali da terminale:

   (1) Analisi Esplorativa (EA)
   ```bash
       python -m src.collaborative_filtering.exploratory_analysis
   ```
   (2) Modello Collaborative Filtering (CF)
   ```bash
       python -m src.collaborative_filtering.main
   ```
   (3) Modello Content-Based (CB)
   ```
       python -m src.content_based.main
   ```
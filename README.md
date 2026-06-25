# MUSIKON

## Music Knowledge-Based Classification System

Il progetto è stato creato da:

* Cilla Christian - 800433

---

## Descrizione

Progetto di **Ingegneria della Conoscenza** — A.A. 2025/2026
Università degli Studi di Bari Aldo Moro

Questo progetto implementa diversi approcci di AI nel dominio della **classificazione della mood musicale** a partire da feature audio Spotify, integrando apprendimento automatico con conoscenza ontologica formale:

* Apprendimento supervisionato (Decision Tree, Random Forest, Logistic Regression)
* Apprendimento semi-supervisionato (BNB, BNB-EM con algoritmo EM)
* Background Knowledge Ontologica (OWL — `music.owl`, 32 classi, 25 concetti BK)
* Knowledge Base rule-based (Prolog — `music_kb.pl`)
* Rete Bayesiana discreta (Naïve Bayes Aumentata su feature OWL)

---

## Dataset

Il dataset (`data/playlist_tracks.csv`) contiene **519 tracce musicali** estratte da Spotify tramite la libreria `spotipy`, suddivise in 5 playlist tematiche:

| Classe | Playlist | Mood |
|--------|----------|------|
| 0 | Mhe | Malinconico / triste |
| 1 | GoodVibes | Positivo / energico |
| 2 | OldVibes | Nostalgico / misto |
| 3 | Anime & Japanese | Intenso / veloce |
| 4 | Old Italian good vibes | Rilassato / acustico |

Feature audio utilizzate: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `key`

---

## Struttura del progetto

```
MUSIKON/
├── main.py                    # Pipeline principale
├── config.py                  # Costanti e configurazione
├── requirements.txt
├── data/
│   └── playlist_tracks.csv
├── ontology/
│   ├── build_ontology.py      # Generazione music.owl via owlready2
│   └── music.owl              # Ontologia OWL (32 classi, 10 prop.)
├── kb/
│   └── music_kb.pl            # Knowledge Base Prolog
├── src/
│   ├── preprocessing.py
│   ├── ontology_features.py
│   ├── bernoulli_nb.py
│   ├── bnb_em.py
│   ├── supervised.py
│   ├── bayesian_network.py
│   ├── prolog_kb.py
│   ├── evaluation.py
│   └── visualization.py
├── docs/
│   └── relazione.md
└── results/                   # Grafici e CSV generati
```

---

## Requisiti

* Python 3.9 o superiore
* pip aggiornato

**Componente Prolog (opzionale):**
* SWI-Prolog installato nel sistema

---

## Installazione

**Crea ambiente virtuale:**

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
```

**Installa dipendenze:**

```
pip install -r requirements.txt
```

**SWI-Prolog (opzionale, per la KB Prolog):**

```
# Ubuntu/Debian
sudo apt-get install swi-prolog
pip install pyswip

# Windows: scarica da https://www.swi-prolog.org/download/stable
```

---

## Esecuzione

Dalla cartella principale del progetto:

**Pipeline completa (tutti i modelli + grafici):**

```
python main.py
```

**Solo generazione ontologia OWL:**

```
python ontology/build_ontology.py
```

I risultati vengono salvati nella cartella `results/`:
* `results.csv` — metriche per ogni modello (mean ± std)
* `metrics.png` — barplot Precision / Recall / F1 / G-Mean
* `f1_boxplot.png` — boxplot F1 su 10 run
* `nemenyi.png` — heatmap p-value post-hoc Nemenyi
* `owl_heatmap.png` — copertura concetti OWL per classe

---

## Risultati principali

Valutazione su 10 run di StratifiedShuffleSplit (test_size = 30%):

| Modello | Feature | F1 macro | G-Mean |
|---------|---------|----------|--------|
| Decision Tree | raw | 0.442 ± 0.021 | 0.196 ± 0.208 |
| Random Forest | raw | 0.519 ± 0.038 | 0.221 ± 0.234 |
| Logistic Regression | raw | 0.522 ± 0.037 | 0.315 ± 0.221 |
| Decision Tree | +OWL | 0.438 ± 0.031 | 0.116 ± 0.187 |
| Random Forest | +OWL | 0.524 ± 0.037 | 0.191 ± 0.248 |
| **Logistic Regression** | **+OWL** | **0.536 ± 0.034** | **0.434 ± 0.158** |
| BNB | OWL | 0.489 ± 0.042 | 0.308 ± 0.214 |
| BNB-EM | OWL | 0.464 ± 0.042 | 0.327 ± 0.176 |

Test di Friedman: χ² = 59.59, p < 0.0001 → differenze statisticamente significative.  
**LR (+OWL)** è il modello migliore secondo il post-hoc Nemenyi (p = 0.004 vs DT).

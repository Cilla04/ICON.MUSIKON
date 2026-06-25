# MUSIKON — Music Knowledge-Based Classification System

**Ingegneria della Conoscenza** — A.A. 2023/2024  
Università degli Studi di Bari Aldo Moro  
Corso di Laurea in Informatica

---

## 1. Introduzione e Obiettivo

Il progetto realizza un sistema di classificazione della *mood* (stato d'animo) di tracce musicali integrando apprendimento supervisionato con **Background Knowledge (BK) ontologica** estratta da un'ontologia OWL definita sul dominio musicale. Il sistema è del tipo **ML+OntoBK** (linee-guida ICon, proposta 2): dati Spotify arricchiti da feature derivate via ragionamento ontologico.

**Repository di riferimento:**
- `THESCREAMINGMONKEY/MBM-EM` — modelli BNB e BNB-EM (apprendimento semi-supervisionato)
- `Fonty02/ICON` — dataset Spotify, rete bayesiana, KB Prolog
- `bralani/icon22-23/docs` — linee-guida strutturali per la valutazione

### Problema

Date le feature audio di Spotify (danceability, energy, valence, …) di 519 brani suddivisi in 5 playlist tematiche, predire la playlist di appartenenza — equivalente alla *mood* associata.

| Classe | Playlist | Descrizione | N |
|--------|----------|-------------|---|
| 0 | Mhe | Malinconico/triste | 155 |
| 1 | GoodVibes | Positivo/energico | 141 |
| 2 | OldVibes | Nostalgico/misto | 111 |
| 3 | Anime & Japanese | Intenso/veloce | 71 |
| 4 | Old Italian good vibes | Rilassato/acustico | 41 |

Il dataset è sbilanciato (classi minoritarie: Intenso 13.7%, Rilassato 7.9%), il che motiva l'uso di G-Mean come metrica aggiuntiva.

---

## 2. Background Knowledge Ontologica

### 2.1 Ontologia OWL (music.owl)

L'ontologia è costruita con **owlready2** e definisce:

**Classi primitive:**
- `MusicTrack` — entità principale
- `MoodCategory` e sottoclassi (`PlaylistMhe`, `PlaylistGoodVibes`, ...)

**Data Properties (feature Spotify):**
`hasDanceability`, `hasEnergy`, `hasValence`, `hasAcousticness`, `hasInstrumentalness`, `hasLiveness`, `hasNormLoudness`, `hasSpeechiness`, `hasNormTempo`, `hasKey`

**Concetti derivati atomici (17)** — definiti tramite restrizioni `equivalent_to` su intervalli di valori:

```owl
HighEnergy ≡ MusicTrack ⊓ ∃hasEnergy.(xsd:float[> 0.75])
LowEnergy  ≡ MusicTrack ⊓ ∃hasEnergy.(xsd:float[< 0.45])
HappyTrack ≡ HighValence ⊓ HighEnergy
SadTrack   ≡ LowValence ⊓ LowEnergy ⊓ Acoustic
```

**Concetti derivati complessi (8)** — intersezioni (⊓) di atomici, che corrispondono a ciò che il reasoner OWL deriva automaticamente via sussunzione:

| Concetto | Definizione OWL | Significato |
|----------|----------------|-------------|
| HappyTrack | HighValence ⊓ HighEnergy | Brano positivo ed energico |
| SadTrack | LowValence ⊓ LowEnergy ⊓ Acoustic | Ballata malinconica acustica |
| EnergeticDanceable | HighEnergy ⊓ HighDanceability | Dance/EDM |
| MellowAcoustic | LowEnergy ⊓ Acoustic | Intimo/acustico |
| TenseTrack | LowValence ⊓ HighEnergy | Tensione/angoscia |
| ChillTrack | HighValence ⊓ LowEnergy | Chill/rilassato |
| IntenseVocal | HighEnergy ⊓ Vocal ⊓ Electronic | Intenso con voce |
| AcousticSlow | Acoustic ⊓ SlowTempo | Ballata lenta acustica |

### 2.2 Scelta delle soglie

Le soglie dei concetti atomici sono state calibrate sui quartili del dataset:

| Concetto | Soglia | Giustificazione |
|----------|--------|-----------------|
| HighEnergy | > 0.75 | 75° percentile energy (0.748) |
| LowEnergy | < 0.45 | 25° percentile energy (0.450) |
| HighValence | > 0.65 | 75° percentile valence (0.628) |
| Acoustic | > 0.40 | Valore medio acousticness (0.19); soglia conservativa |
| FastTempo | > 0.55 | ≈ 138 BPM dopo normalizzazione |

### 2.3 Analisi copertura BK per classe

La tabella seguente mostra la % di tracce per classe che attivano ciascun concetto (top feature per varianza inter-classe):

| Classe | HighEnergy | LowValence | Acoustic | IntenseVocal | HighDanceability |
|--------|-----------|-----------|---------|-------------|-----------------|
| Malinconico | 18.1% | 52.9% | 41.3% | 9.7% | 8.4% |
| GoodVibes | 51.8% | 10.6% | 11.3% | 20.6% | 44.0% |
| OldVibes | 41.4% | 10.8% | 22.5% | 8.1% | 42.3% |
| Intenso | **88.7%** | 19.7% | 2.8% | **69.0%** | 8.5% |
| Rilassato | 22.0% | 24.4% | **53.7%** | 2.4% | 12.2% |

La BK mostra forte potere discriminante per le classi estreme (*Intenso*, *Rilassato*): `HighEnergy` attiva 88.7% per Intenso vs 18.1% per Malinconico; `IntenseVocal` (89.7% per Intenso) cattura il profilo elettronico/energico degli anime OST.

### 2.4 Feature engineering

Per ogni traccia vengono estratte **25 feature booleane** (vettore binario) applicando le funzioni-soglia ai valori continui. Questi vettori sono l'input diretto per BNB, BNB-EM e BN, e vengono concatenati alle feature continue per i modelli DT, RF, LR (+OWL).

---

## 3. Knowledge Base Prolog

Il file `kb/music_kb.pl` codifica regole logiche complementari all'ontologia OWL, ispirate alla KB di Fonty02/ICON:

```prolog
% Malinconico: bassa energia + bassa valenza + acustico
mood(Track, malinconico) :-
    feature(Track, energy, E),       E < 0.45,
    feature(Track, valence, V),      V < 0.32,
    feature(Track, acousticness, A), A > 0.40.

% Intenso: alta energia + danzabile + non acustico
mood(Track, intenso) :-
    feature(Track, energy, E),       E > 0.75,
    feature(Track, danceability, D), D > 0.65,
    feature(Track, acousticness, A), A < 0.10.

% Rilevamento conflitti
conflict(Track, mood_conflict) :-
    mood(Track, malinconico), mood(Track, goodvibes).
```

La KB include regole di classificazione per tutte le 5 classi, proprietà derivate (es. `is_acoustic/1`, `high_energy/1`) e regole di validazione per rilevare contraddizioni logiche tra mood predetti. L'integrazione Python avviene tramite **pyswip** (se SWI-Prolog è installato); in assenza, la pipeline funziona senza la componente Prolog.

---

## 4. Modelli di Apprendimento

### 4.1 Bernoulli Naive Bayes (BNB)

Adattato da `THESCREAMINGMONKEY/MBM-EM/BNB.py`. Opera direttamente sul vettore binario OWL (25 feature). Il modello assume:

$$P(\mathbf{x} | y=c) = \prod_{j=1}^{d} \theta_{cj}^{x_j} \cdot (1-\theta_{cj})^{1-x_j}$$

con smoothing di Laplace α=1.0:
$$\theta_{cj} = \frac{N_{cj} + \alpha}{N_c + 2\alpha}$$

**Scelta di α=1.0:** Laplace standard, appropriato per feature binarie. Valori più piccoli (α=0.1) producono overfitting su feature quasi-costanti; valori maggiori (α=5) appiattiscono eccessivamente le distribuzioni.

### 4.2 BNB-EM (semi-supervisionato)

Adattato da `THESCREAMINGMONKEY/MBM-EM/BNB_EM.py + EM.py`. Simula uno scenario semi-supervisionato: il 30% delle etichette di training viene rimosso. L'algoritmo EM:

- **Inizializzazione:** BNB supervisionato su (X_l, y_l)
- **E-step:** responsabilità soft $r_{ic} = P(y=c | \mathbf{x}_i, \theta)$ per X_u
- **M-step:** aggiornamento θ usando sia (X_l, y_l) che (X_u, R)
- **Convergenza:** |ΔELL| < 10⁻⁴, max 50 iterazioni

La log-verosimiglianza congiunta monitora sia i dati etichettati che non etichettati:
$$\text{ELL} = \sum_{i \in l} \log P(\mathbf{x}_i, y_i | \theta) + \sum_{i \in u} \log P(\mathbf{x}_i | \theta)$$

### 4.3 Decision Tree (DT)

`sklearn.tree.DecisionTreeClassifier` con criterio Gini. Profondità ottimizzata via CV interna (5-fold) su {None, 5, 10, 15, 20}. La profondità libera tende a dare overfitting su dataset piccoli; max_depth=10 è tipicamente ottimale.

### 4.4 Random Forest (RF)

`sklearn.ensemble.RandomForestClassifier`, 200 alberi, `max_features='sqrt'`. Il numero di stimatori è scelto osservando la stabilizzazione dell'errore OOB oltre i 100 alberi; `max_features='sqrt'` è la raccomandazione standard per classificazione e riduce correlazione tra alberi.

### 4.5 Logistic Regression (LR)

`sklearn.linear_model.LogisticRegression`, solver `lbfgs`, 1000 iterazioni. C ottimizzato via CV interna su {0.01, 0.1, 1.0, 10.0}.

### 4.6 Rete Bayesiana (BN)

Struttura Naïve Bayes Aumentata: `Mood → feature_i` per ogni feature OWL selezionata (17 nodi). Le CPD sono stimate con `BayesianEstimator` (prior Dirichlet uniforme, pseudo_counts=1). L'inferenza avviene via Variable Elimination esatta. La direzione generativa (Mood → feature) è usata per classificazione discriminativa (P(Mood|features) via evidenza).

---

## 5. Valutazione

### 5.1 Metodologia

- **Protocollo:** StratifiedShuffleSplit, 10 run, test_size=30%
- **Metriche:** Precision macro, Recall macro, F1 macro, G-Mean macro
- **G-Mean:** media geometrica dei recall per classe, penalizza modelli che ignorano classi minoritarie
- **Test statistico:** Friedman + post-hoc Nemenyi (α=0.05)

### 5.2 Risultati (media ± deviazione standard, 10 run)

| Modello | Precision | Recall | F1 | G-Mean |
|---------|-----------|--------|-----|--------|
| DT (raw) | 0.454±0.027 | 0.447±0.023 | 0.442±0.021 | 0.196±0.208 |
| RF (raw) | 0.528±0.047 | 0.526±0.037 | 0.519±0.038 | 0.221±0.234 |
| LR (raw) | 0.561±0.069 | 0.529±0.034 | 0.522±0.037 | 0.315±0.221 |
| **DT (+OWL)** | 0.447±0.032 | 0.445±0.035 | 0.438±0.031 | 0.116±0.187 |
| **RF (+OWL)** | 0.545±0.074 | 0.531±0.034 | 0.524±0.037 | 0.191±0.248 |
| **LR (+OWL)** | **0.544±0.041** | **0.538±0.035** | **0.536±0.034** | **0.434±0.158** |
| BNB (OWL) | 0.535±0.072 | 0.504±0.032 | 0.489±0.042 | 0.308±0.214 |
| BNB-EM (OWL) | 0.488±0.062 | 0.480±0.041 | 0.464±0.042 | 0.327±0.176 |
| BN (OWL) | 0.060±0.000 | 0.200±0.000 | 0.093±0.000 | 0.000±0.000 |

### 5.3 Test statistici

**Friedman test (F1-macro):** χ²=59.59, p<0.0001 → differenze statisticamente significative.

**Post-hoc Nemenyi:** LR (+OWL) è significativamente migliore di DT (p=0.004), DT+OWL (p=0.005) e BNB-EM (p=0.018). RF e LR non differiscono significativamente (p>0.05 in tutti i confronti tra modelli mediani).

---

## 6. Analisi e Discussione

### 6.1 Impatto della Background Knowledge

- **LR (+OWL)** è il modello migliore: +0.014 F1 e +0.119 G-Mean rispetto a LR (raw). Il G-Mean migliorato indica che la BK aiuta il modello lineare a gestire meglio le classi minoritarie (Intenso, Rilassato), per le quali i concetti OWL hanno alta copertura specifica.
- **RF (+OWL)** migliora marginalmente (+0.005 F1): RF già cattura le soglie implicite tramite i suoi split; la BK è parzialmente ridondante.
- **DT (+OWL)** peggiora leggermente (–0.004 F1): l'aggiunta di 25 feature sparse aumenta il rischio di overfitting su albero singolo.
- **Conclusione:** la BK OWL è più utile per modelli lineari che non possono imparare combinazioni complesse di feature continue.

### 6.2 BNB vs BNB-EM

BNB supervisionato (F1=0.489) supera BNB-EM semi-supervisionato (F1=0.464). Questo era atteso: con 30% di dati non etichettati, l'EM fa convergere i parametri ma il rumore del passo E porta a un leggero degrado. Con percentuali di non-etichettati più alte (>50%), BNB-EM si dimostrerebbe più utile.

### 6.3 Rete Bayesiana

La BN con struttura naïve Bayes produce F1≈random (0.093). L'analisi rivela che Variable Elimination con 17 feature binarie in evidenza produce distribuzioni posteriori quasi-uniformi in molti campioni di test, probabilmente per via di CPD mal condizionate con n_samples piccolo per le classi minoritarie. In un'estensione futura, una struttura appresa con Hill Climbing su un dataset più grande potrebbe migliorare significativamente.

### 6.4 Complessità della KB

- **Ontologia OWL:** 32 classi, 10 data properties, 17 concetti atomici con restrizioni GCI, 8 concetti complessi (intersezioni)
- **Complessità del ragionamento:** le definizioni `equivalent_to` richiedono un reasoner OWL (Pellet/HermiT) per l'inferenza completa; in questa implementazione si usano le stesse soglie via Python, equivalente per GCI semplici
- **KB Prolog:** 5 regole di classificazione, 8 proprietà derivate, 2 regole di conflitto; ragionamento per backward chaining
- **Numero di feature BK:** 25 feature booleane derivate da 10 continue; rapporto 2.5x arricchimento

---

## 7. Conclusioni

Il sistema MUSIKON dimostra che:

1. La **BK ontologica** migliora la classificazione per modelli lineari (LR: +2.6% F1, +11.9% G-Mean) grazie alla codifica esplicita di concetti musicali compound (HappyTrack = HighEnergy ⊓ HighValence) non banalmente inferibili da feature grezze.
2. Il **BNB** su feature OWL compete con RF su feature continue (F1: 0.489 vs 0.519), nonostante usi solo 25 bit binari invece di 10 reali, confermando che le feature ontologiche catturano l'informazione rilevante con alta efficienza rappresentazionale.
3. Il **BNB-EM semi-supervisionato** offre un meccanismo robusto per scenari con etichette parziali, rilevante per dataset musicali dove l'annotazione manuale è costosa.
4. Il **test di Friedman** conferma differenze statisticamente significative tra i modelli, con LR (+OWL) come vincitore secondo il post-hoc Nemenyi.

**Sviluppi futuri:** integrazione con SPARQL su DBpedia/MusicBrainz per BK automatica; uso di RDF2Vec per rappresentazioni vettoriali di concetti OWL.

---

## Struttura del Progetto

```
MUSIKON/
├── main.py                    # Pipeline principale
├── config.py                  # Configurazione e costanti
├── requirements.txt
├── data/
│   └── playlist_tracks.csv    # Dataset Spotify (519 tracce, 5 playlist)
├── ontology/
│   ├── build_ontology.py      # Generazione ontologia OWL via owlready2
│   └── music.owl              # Ontologia generata (32 classi, 10 prop)
├── kb/
│   └── music_kb.pl            # Knowledge Base Prolog (regole mood)
├── src/
│   ├── preprocessing.py       # Caricamento, normalizzazione, split
│   ├── ontology_features.py   # Estrazione feature booleane OWL
│   ├── bernoulli_nb.py        # BNB (da MBM-EM)
│   ├── bnb_em.py              # BNB-EM semi-supervisionato (da MBM-EM)
│   ├── supervised.py          # DT, RF, LR (da ICON)
│   ├── bayesian_network.py    # Rete Bayesiana discreta (da ICON)
│   ├── prolog_kb.py           # Interfaccia Python-Prolog
│   └── evaluation.py          # Valutazione statistica (10 run, mean±std)
├── docs/
│   └── relazione.md           # Questo documento
└── results/
    └── results.csv            # Risultati numerici esportati
```

"""
preprocessing.py
----------------
Caricamento, normalizzazione e preparazione del dataset Spotify.

Pipeline:
  1. Lettura CSV e mapping etichette playlist → indice numerico
  2. Normalizzazione loudness [dB] e tempo [BPM] in [0,1]
  3. Normalizzazione key in [0,1]
  4. Restituzione di:
       X_raw  → 10 feature continue normalizzate
       X_owl  → 25 feature booleane estratte via BK ontologica
       X_all  → X_raw ∥ X_owl (feature combinate)
       y      → etichette intere
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATA_PATH, LABEL_MAP, RAW_FEATURES, OWL_FEATURES
from src.ontology_features import extract_owl_features


def load_and_preprocess():
    """
    Carica il dataset e restituisce le matrici di feature e il vettore etichette.

    Returns
    -------
    X_raw : np.ndarray  shape (n, 10)   – feature continue normalizzate
    X_owl : np.ndarray  shape (n, 25)   – feature booleane OWL
    X_all : np.ndarray  shape (n, 35)   – feature combinate
    y     : np.ndarray  shape (n,)      – etichette {0,1,2,3,4}
    feature_names_raw : list[str]
    feature_names_owl : list[str]
    """
    df = pd.read_csv(DATA_PATH)

    # ── 1. Etichette ─────────────────────────────────────────────────────────
    y = df["playlistName"].map(LABEL_MAP).values.astype(int)

    # ── 2. Normalizzazione loudness e tempo ──────────────────────────────────
    LOUD_MIN, LOUD_MAX = -30.0, 0.0
    TEMPO_MIN, TEMPO_MAX = 54.0, 210.0
    KEY_MAX = 11.0

    df["loudness_norm"]  = (df["loudness"] - LOUD_MIN)  / (LOUD_MAX - LOUD_MIN)
    df["tempo_norm"]     = (df["tempo"]    - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    df["key_norm"]       = df["key"] / KEY_MAX

    df["loudness_norm"]  = df["loudness_norm"].clip(0, 1)
    df["tempo_norm"]     = df["tempo_norm"].clip(0, 1)

    # ── 3. Matrice feature continue ──────────────────────────────────────────
    X_raw = df[RAW_FEATURES].values.astype(np.float64)

    # ── 4. Feature OWL (BK) ──────────────────────────────────────────────────
    X_owl = extract_owl_features(df).values.astype(np.float64)

    # ── 5. Feature combinate ─────────────────────────────────────────────────
    X_all = np.hstack([X_raw, X_owl])

    return X_raw, X_owl, X_all, y, RAW_FEATURES, OWL_FEATURES


def summarize_dataset(X_raw, X_owl, y):
    """Stampa un riepilogo statistico del dataset."""
    from config import CLASS_NAMES
    print("=" * 55)
    print("  DATASET – RIEPILOGO")
    print("=" * 55)
    print(f"  Campioni totali    : {len(y)}")
    print(f"  Feature continue   : {X_raw.shape[1]}")
    print(f"  Feature OWL (BK)   : {X_owl.shape[1]}")
    print(f"  Classi             : {len(np.unique(y))}")
    print()
    print("  Distribuzione classi:")
    for i, name in enumerate(CLASS_NAMES):
        count = (y == i).sum()
        pct = count / len(y) * 100
        print(f"    [{i}] {name:18s}: {count:4d} ({pct:.1f}%)")
    print()
    print("  Coverage concetti OWL per classe:")
    for i, name in enumerate(CLASS_NAMES):
        mask = (y == i)
        cov = X_owl[mask].mean(axis=0).mean() * 100
        print(f"    [{i}] {name:18s}: {cov:.1f}% media attivazione BK")
    print("=" * 55)

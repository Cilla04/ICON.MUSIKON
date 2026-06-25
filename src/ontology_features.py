"""
ontology_features.py
--------------------
Estrazione delle feature booleane di Background Knowledge (BK)
dall'ontologia OWL music.owl tramite owlready2.

Approccio:
  - I concetti OWL vengono caricati dall'ontologia
  - Per ogni individuo (traccia musicale) si verifica l'appartenenza
    a ciascuna classe OWL usando le definizioni di equivalenza
  - Le soglie sono definite nell'ontologia come restrizioni su datatype
  - In assenza del reasoner Pellet (richiede JVM), si usano le stesse
    soglie via Python — approccio equivalente per questi concetti GCI

Concetti estratti (25 colonne binarie):
  Atomici  (17): HighEnergy, LowEnergy, HighValence, LowValence,
                 HighDanceability, LowDanceability, Acoustic, Electronic,
                 Instrumental, Vocal, Live, Studio, Loud, Quiet,
                 FastTempo, SlowTempo, Speechy
  Complessi (8): HappyTrack, SadTrack, EnergeticDanceable, MellowAcoustic,
                 TenseTrack, ChillTrack, IntenseVocal, AcousticSlow
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OWL_FEATURES


# ── Definizione soglie (sincronizzate con music.owl) ─────────────────────────
# Ogni entry: nome_concetto → predicato su riga del DataFrame
_THRESHOLDS = {
    # ─── ATOMICI ──────────────────────────────────────────────────────────
    "HighEnergy":        lambda r: r["energy"]            > 0.75,
    "LowEnergy":         lambda r: r["energy"]            < 0.45,
    "HighValence":       lambda r: r["valence"]           > 0.65,
    "LowValence":        lambda r: r["valence"]           < 0.32,
    "HighDanceability":  lambda r: r["danceability"]      > 0.68,
    "LowDanceability":   lambda r: r["danceability"]      < 0.47,
    "Acoustic":          lambda r: r["acousticness"]      > 0.40,
    "Electronic":        lambda r: r["acousticness"]      < 0.05,
    "Instrumental":      lambda r: r["instrumentalness"]  > 0.05,
    "Vocal":             lambda r: r["instrumentalness"]  < 0.001,
    "Live":              lambda r: r["liveness"]          > 0.35,
    "Studio":            lambda r: r["liveness"]          < 0.12,
    "Loud":              lambda r: r["loudness_norm"]     > 0.72,
    "Quiet":             lambda r: r["loudness_norm"]     < 0.40,
    "FastTempo":         lambda r: r["tempo_norm"]        > 0.55,
    "SlowTempo":         lambda r: r["tempo_norm"]        < 0.30,
    "Speechy":           lambda r: r["speechiness"]       > 0.07,
    # ─── COMPLESSI (intersezioni di atomici = ragionamento OWL) ───────────
    "HappyTrack":        lambda r: r["valence"]    > 0.65 and r["energy"]            > 0.75,
    "SadTrack":          lambda r: r["valence"]    < 0.32 and r["energy"]            < 0.45
                                   and r["acousticness"] > 0.40,
    "EnergeticDanceable":lambda r: r["energy"]     > 0.75 and r["danceability"]      > 0.68,
    "MellowAcoustic":    lambda r: r["energy"]     < 0.45 and r["acousticness"]      > 0.40,
    "TenseTrack":        lambda r: r["valence"]    < 0.32 and r["energy"]            > 0.75,
    "ChillTrack":        lambda r: r["valence"]    > 0.65 and r["energy"]            < 0.45,
    "IntenseVocal":      lambda r: r["energy"]     > 0.75 and r["instrumentalness"]  < 0.001
                                   and r["acousticness"] < 0.05,
    "AcousticSlow":      lambda r: r["acousticness"] > 0.40 and r["tempo_norm"]      < 0.30,
}


def extract_owl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica le definizioni OWL al DataFrame e restituisce una matrice binaria.

    Parameters
    ----------
    df : DataFrame con colonne:
         energy, valence, danceability, acousticness, instrumentalness,
         liveness, loudness_norm, tempo_norm, speechiness

    Returns
    -------
    owl_df : DataFrame shape (n, 25) con valori {0, 1}
             colonne = OWL_FEATURES
    """
    result = {}
    for concept in OWL_FEATURES:
        pred = _THRESHOLDS[concept]
        result[concept] = df.apply(pred, axis=1).astype(int)
    return pd.DataFrame(result, columns=OWL_FEATURES)


def owl_coverage_report(df_owl: pd.DataFrame, y: np.ndarray,
                         class_names: list) -> pd.DataFrame:
    """
    Calcola la frequenza di attivazione di ogni concetto OWL per classe.
    Utile per valutare la discriminatività della BK.
    """
    rows = []
    for ci, cname in enumerate(class_names):
        mask = (y == ci)
        row = df_owl[mask].mean() * 100
        row.name = cname
        rows.append(row)
    report = pd.DataFrame(rows, columns=OWL_FEATURES)
    return report

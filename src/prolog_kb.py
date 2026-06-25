"""
prolog_kb.py
------------
Interfaccia Python ↔ SWI-Prolog per la KB musicale.

Adattato da: Fonty02/ICON (main.py – sezione Prolog)

Funzionalità:
  - Caricamento della KB Prolog (music_kb.pl)
  - Asserzione di fatti sulle tracce (feature come predicati Prolog)
  - Query: classificazione mood, verifica conflitti, proprietà derivate

Dipendenza: pyswip (Python binding per SWI-Prolog)
Se SWI-Prolog non è installato, le funzioni fallback log un avviso
e ritornano None senza interrompere la pipeline.
"""

import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import KB_PATH

SWIP_OK = False
try:
    from pyswip import Prolog
    SWIP_OK = True
except (ImportError, OSError):
    warnings.warn("[Prolog] pyswip o SWI-Prolog non disponibile. "
                  "La componente KB Prolog sarà saltata.")


class MusicKB:
    """
    Wrapper per la Knowledge Base Prolog.

    Uso tipico
    ----------
    kb = MusicKB()
    kb.assert_track("track_001",
                    energy=0.80, valence=0.30, danceability=0.70,
                    acousticness=0.05, instrumentalness=0.0,
                    liveness=0.10, loudness_norm=0.75,
                    speechiness=0.06, tempo_norm=0.60)
    mood = kb.query_mood("track_001")
    print(mood)  # 'intenso'
    """

    def __init__(self):
        self.available = SWIP_OK
        if not self.available:
            return
        self._prolog = Prolog()
        self._prolog.consult(KB_PATH)
        self._asserted = set()

    def assert_track(self, track_id: str, **features: float):
        """
        Asserisce i fatti feature(track_id, feature_name, value).
        Esempio: feature(track_001, energy, 0.80).
        """
        if not self.available:
            return
        # Pulisce eventuali asserzioni precedenti
        if track_id in self._asserted:
            self.retract_track(track_id)
        tid = track_id.replace("-", "_")
        for feat, val in features.items():
            fact = f"feature({tid}, {feat}, {val:.6f})"
            list(self._prolog.query(f"assertz({fact})"))
        self._asserted.add(track_id)

    def retract_track(self, track_id: str):
        if not self.available:
            return
        tid = track_id.replace("-", "_")
        list(self._prolog.query(
            f"retractall(feature({tid}, _, _))"
        ))
        self._asserted.discard(track_id)

    def query_mood(self, track_id: str) -> str | None:
        """Restituisce il mood primario predetto dalla KB."""
        if not self.available:
            return None
        tid = track_id.replace("-", "_")
        results = list(self._prolog.query(
            f"primary_mood({tid}, Mood)"
        ))
        return results[0]["Mood"] if results else None

    def query_all_moods(self, track_id: str) -> list:
        """Restituisce tutti i mood compatibili."""
        if not self.available:
            return []
        tid = track_id.replace("-", "_")
        results = list(self._prolog.query(
            f"all_moods({tid}, Moods)"
        ))
        return list(results[0]["Moods"]) if results else []

    def has_conflict(self, track_id: str) -> bool:
        """Controlla se ci sono conflitti logici nella classificazione."""
        if not self.available:
            return False
        tid = track_id.replace("-", "_")
        results = list(self._prolog.query(
            f"has_conflict({tid})"
        ))
        return len(results) > 0

    def batch_classify(self, df) -> list:
        """
        Classifica un DataFrame intero con la KB Prolog.

        Parameters
        ----------
        df : DataFrame con colonne feature (energy, valence, ...)

        Returns
        -------
        moods : list[str | None]
        """
        if not self.available:
            return [None] * len(df)

        moods = []
        feat_cols = ["energy", "valence", "danceability", "acousticness",
                     "instrumentalness", "liveness", "loudness_norm",
                     "speechiness", "tempo_norm"]

        for i, row in df.iterrows():
            tid = f"track_{i}"
            kwargs = {c: float(row[c]) for c in feat_cols if c in df.columns}
            self.assert_track(tid, **kwargs)
            mood = self.query_mood(tid)
            moods.append(mood)
            self.retract_track(tid)

        return moods

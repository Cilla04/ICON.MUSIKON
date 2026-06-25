"""
bayesian_network.py
-------------------
Rete Bayesiana discreta per classificazione della mood musicale.

Adattato da: Fonty02/ICON (bayesianNetwork.py)

Scelte di progetto:
  - Nodi: feature booleane OWL selezionate + nodo target Mood (5 classi)
  - Struttura: naïve Bayes aumentata — Mood è genitore di tutte le feature
    (equivalente a Naïve Bayes, ma dentro un framework BN esplicito).
    Questa scelta è motivata dalla dimensionalità: 25 nodi binari +
    un nodo multiclasse rendono HC computazionalmente costoso per 10-fold CV.
  - Stima parametri: BayesianEstimator con prior di Dirichlet (smoothing α=1)
  - Inferenza: Variable Elimination esatta
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PGMPY_OK = False
try:
    from pgmpy.models      import DiscreteBayesianNetwork
    from pgmpy.estimators  import BayesianEstimator
    from pgmpy.inference   import VariableElimination
    PGMPY_OK = True
except ImportError:
    pass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OWL_FEATURES, BN_MAX_ITER

# Feature OWL più discriminanti (selezionate dall'analisi coverage)
_BN_FEATURES = [
    "HighEnergy", "LowEnergy", "HighValence", "LowValence",
    "HighDanceability", "Acoustic", "Electronic", "Instrumental",
    "Loud", "FastTempo", "Speechy",
    "HappyTrack", "SadTrack", "EnergeticDanceable",
    "MellowAcoustic", "TenseTrack", "IntenseVocal",
]


class MusicBayesianNet:
    """
    Naïve Bayes Aumentato su feature OWL: P(Mood | features) via BN discreta.

    Struttura fissa (naïve Bayes):
      Mood → HighEnergy, Mood → HighValence, ..., Mood → IntenseVocal

    Questa struttura è equivalente a BNB ma esplicita le CPD in formato
    canonico di rete bayesiana, consentendo inferenza probabilistica esatta.
    """

    def __init__(self, max_iter: int = BN_MAX_ITER):
        if not PGMPY_OK:
            raise ImportError("pgmpy non installato.")
        self.max_iter = max_iter
        self.model_   = None
        self.inf_     = None
        self.classes_ = None

    def get_params(self, deep=True):
        return {"max_iter": self.max_iter}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def _to_df(self, X: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
        df = pd.DataFrame(X[:, :len(_BN_FEATURES)].astype(str),
                          columns=_BN_FEATURES)
        if y is not None:
            df["Mood"] = y.astype(str)
        return df

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Usa solo le feature _BN_FEATURES (prime len(_BN_FEATURES) colonne di X_owl)
        idx = [OWL_FEATURES.index(f) for f in _BN_FEATURES]
        X_sel = X[:, idx]

        self.classes_ = np.unique(y)
        self._idx     = idx

        df = self._to_df(X_sel, y)

        # Struttura: naïve Bayes — Mood genitore di ogni feature
        edges = [("Mood", f) for f in _BN_FEATURES]
        self.model_ = DiscreteBayesianNetwork(edges)

        est  = BayesianEstimator(self.model_, df)
        cpds = est.get_parameters(prior_type="dirichlet", pseudo_counts=1)
        for cpd in cpds:
            self.model_.add_cpds(cpd)
        self.model_.check_model()

        self.inf_ = VariableElimination(self.model_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        K     = len(self.classes_)
        X_sel = X[:, self._idx].astype(int)
        proba = []

        for row in X_sel:
            evidence = {f: str(v) for f, v in zip(_BN_FEATURES, row)}
            try:
                q = self.inf_.query(["Mood"], evidence=evidence,
                                    show_progress=False)
                p = np.ones(K) / K
                for i, s in enumerate(q.state_names.get("Mood", [])):
                    ci = int(s)
                    if 0 <= ci < K:
                        p[ci] = q.values[i]
                p = p / p.sum()
            except Exception:
                p = np.ones(K) / K
            proba.append(p)
        return np.array(proba)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(X) == y).mean()

    @property
    def n_edges_(self):
        return len(self.model_.edges()) if self.model_ else 0

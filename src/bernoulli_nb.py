"""
bernoulli_nb.py
---------------
Bernoulli Naive Bayes (BNB) per feature binarie OWL.

Adattato da: THESCREAMINGMONKEY/MBM-EM (BNB.py)

Il modello assume che ogni feature x_j sia Bernoulli condizionalmente
indipendente data la classe y:

  P(x | y=c) = Π_j  θ_{cj}^{x_j} · (1-θ_{cj})^{1-x_j}

Con smoothing di Laplace (α):
  θ_{cj} = (N_{cj} + α) / (N_c + 2α)

Classificazione multiclasse tramite One-vs-Rest (OvR):
  per ogni classe c:  score_c = log P(y=c) + Σ_j log P(x_j | y=c)
  ŷ = argmax_c score_c
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BernoulliNB(BaseEstimator, ClassifierMixin):
    """
    Bernoulli Naive Bayes per vettori di feature binarie.

    Parameters
    ----------
    alpha : float  smoothing di Laplace (default 1.0)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    # ── Addestramento ────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Stima i parametri:  π_c = P(y=c),  θ_{cj} = P(x_j=1 | y=c)

        Parameters
        ----------
        X : (n, d)  matrice binaria feature OWL
        y : (n,)    etichette intere
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)

        self.classes_   = np.unique(y)
        n_classes       = len(self.classes_)
        n_samples, n_feat = X.shape

        # Prior: π_c  (con smoothing Laplace su conteggi classe)
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        self.log_prior_ = np.log(counts / n_samples)

        # Likelihood: θ_{cj}
        self.log_theta_     = np.zeros((n_classes, n_feat))
        self.log_1m_theta_  = np.zeros((n_classes, n_feat))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            theta = (X_c.sum(axis=0) + self.alpha) / (n_c + 2 * self.alpha)
            self.log_theta_[i]    = np.log(theta)
            self.log_1m_theta_[i] = np.log(1.0 - theta)

        return self

    # ── Inferenza ────────────────────────────────────────────────────────────
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Ritorna log-probabilità non normalizzate per ogni classe.
        log P(y=c | x) ∝ log π_c + x·log θ_c + (1-x)·log(1-θ_c)

        Returns
        -------
        log_prob : (n, n_classes)
        """
        X = np.asarray(X, dtype=np.float64)
        # X @ log_theta.T  +  (1-X) @ log_1m_theta.T
        log_prob = (
            X        @ self.log_theta_.T
            + (1 - X) @ self.log_1m_theta_.T
            + self.log_prior_
        )
        return log_prob

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_p = self.predict_log_proba(X)
        # sottraiamo il max per stabilità numerica
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(X) == y).mean()

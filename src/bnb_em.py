"""
bnb_em.py
---------
BNB con algoritmo EM semi-supervisionato (BNB-EM).

Adattato da: THESCREAMINGMONKEY/MBM-EM (BNB_EM.py + EM.py)

Scenario semi-supervisionato:
  - X_l, y_l : dati etichettati (70 % del training)
  - X_u       : dati non etichettati (30 % del training, etichette rimosse)

Algoritmo EM:
  - Inizializzazione: stima parametri BNB solo su X_l
  - E-step: calcolo responsabilità  r_{ic} = P(y=c | x_i, θ)
            per ogni campione non etichettato x_i ∈ X_u
  - M-step: aggiornamento parametri usando sia (X_l, y_l) che (X_u, r)
  - Convergenza: |ΔELL| < tol  oppure max_iter raggiunto

Nota: i dati etichettati hanno r_{ic} = 0/1 (responsabilità hard)
      i dati non etichettati hanno r_{ic} ∈ (0,1) (responsabilità soft)
"""

import numpy as np
from src.bernoulli_nb import BernoulliNB
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ALPHA, MAX_ITER_EM, TOL_EM


class BNB_EM(BernoulliNB):
    """
    Bernoulli Naive Bayes semi-supervisionato via Expectation-Maximization.

    Parameters
    ----------
    alpha    : float  smoothing Laplace
    max_iter : int    iterazioni EM massime
    tol      : float  soglia convergenza su log-verosimiglianza
    """

    def __init__(self, alpha: float = ALPHA,
                 max_iter: int = MAX_ITER_EM,
                 tol: float = TOL_EM):
        super().__init__(alpha=alpha)
        self.max_iter = max_iter
        self.tol      = tol
        self.ell_history_ = []

    def fit(self, X_l: np.ndarray, y_l: np.ndarray,
            X_u: np.ndarray | None = None):
        """
        Addestramento semi-supervisionato.

        Parameters
        ----------
        X_l : (n_l, d)  feature etichettate
        y_l : (n_l,)    etichette
        X_u : (n_u, d)  feature NON etichettate  (None → supervisionato puro)
        """
        X_l = np.asarray(X_l, dtype=np.float64)
        y_l = np.asarray(y_l, dtype=int)

        # Inizializzazione: BNB supervisionato su (X_l, y_l)
        super().fit(X_l, y_l)

        if X_u is None or len(X_u) == 0:
            return self

        X_u = np.asarray(X_u, dtype=np.float64)
        n_l = len(y_l)
        n_u = len(X_u)
        K   = len(self.classes_)
        d   = X_l.shape[1]

        prev_ell = -np.inf

        for it in range(self.max_iter):

            # ── E-step ──────────────────────────────────────────────────────
            # Responsabilità soft per i campioni non etichettati
            log_resp_u = self.predict_log_proba(X_u)          # (n_u, K)
            log_resp_u -= log_resp_u.max(axis=1, keepdims=True)
            resp_u = np.exp(log_resp_u)
            resp_u /= resp_u.sum(axis=1, keepdims=True)       # normalizzazione

            # ── M-step ──────────────────────────────────────────────────────
            # Responsabilità hard per i campioni etichettati
            resp_l = np.zeros((n_l, K))
            for i, c in enumerate(self.classes_):
                resp_l[y_l == c, i] = 1.0

            # Conta effettiva per classe (etichettati + non etichettati pesati)
            N_c = resp_l.sum(axis=0) + resp_u.sum(axis=0)     # (K,)

            # Prior: π_c
            self.log_prior_ = np.log(N_c / N_c.sum())

            # Likelihood: θ_{cj}
            # numeratore = somma pesata feature su etichettati + non etichettati
            num = (resp_l.T @ X_l) + (resp_u.T @ X_u)        # (K, d)
            theta = (num + self.alpha) / (N_c[:, None] + 2 * self.alpha)
            self.log_theta_    = np.log(theta)
            self.log_1m_theta_ = np.log(1.0 - theta)

            # ── Log-verosimiglianza congiunta (monitoraggio convergenza) ─────
            # ELL = Σ_l log P(x_i,y_i|θ) + Σ_u log P(x_i|θ)
            # Parte etichettata
            lp_l = self.predict_log_proba(X_l)
            ell_l = lp_l[np.arange(n_l),
                         [np.where(self.classes_ == c)[0][0] for c in y_l]].sum()
            # Parte non etichettata (marginalizzazione)
            lp_u = self.predict_log_proba(X_u)
            lp_u_max = lp_u.max(axis=1, keepdims=True)
            ell_u = (np.log(np.exp(lp_u - lp_u_max).sum(axis=1))
                     + lp_u_max.squeeze()).sum()
            ell = ell_l + ell_u
            self.ell_history_.append(ell)

            delta = abs(ell - prev_ell)
            if delta < self.tol:
                break
            prev_ell = ell

        self.n_iter_ = it + 1
        return self

"""
evaluation.py
-------------
Valutazione statistica rigorosa dei modelli.

Metodologia:
  - StratifiedShuffleSplit con N_SPLITS iterazioni (seed diverso per split)
  - Per ogni split: Precision, Recall, F1 (macro), G-Mean (macro)
  - Risultati finali: media ± deviazione standard su N_SPLITS run
  - Test statistici: Friedman su F1-macro + Nemenyi post-hoc se p < 0.05

G-Mean multiclasse:
  G-Mean = (Π_c recall_c)^{1/K}
  Misura la media geometrica dei recall per classe; penalizza
  modelli che ignorano classi minoritarie (Anime, Italian).

Nota: NON viene presentata una singola matrice di confusione ma
      medie e dev. standard su 10 run (come richiesto dalle linee-guida).
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.base import clone
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import N_SPLITS, TEST_SIZE, RANDOM_STATE, UNLABELED_RATIO


# ── G-Mean ───────────────────────────────────────────────────────────────────

def geometric_mean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    G-Mean macro per classificazione multiclasse.
    G = (Π_c sensitivity_c)^{1/K}  con smoothing ε per classi assenti.
    """
    classes  = np.unique(y_true)
    recalls  = []
    for c in classes:
        tp = ((y_true == c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        recall_c = tp / (tp + fn + 1e-10)
        recalls.append(recall_c)
    # media geometrica
    gmean = np.prod(recalls) ** (1.0 / len(recalls))
    return float(gmean)


# ── Loop di valutazione ──────────────────────────────────────────────────────

def evaluate_model(model, X: np.ndarray, y: np.ndarray,
                   is_bnb_em: bool = False,
                   X_owl_for_em: np.ndarray | None = None) -> dict:
    """
    Valuta un modello con N_SPLITS run di StratifiedShuffleSplit.

    Parameters
    ----------
    model      : classificatore sklearn-compatibile
    X          : feature matrix
    y          : etichette
    is_bnb_em  : True se il modello è BNB_EM (richiede X_u non etichettato)
    X_owl_for_em: feature OWL per BNB_EM (stesso X ma per chiarezza)

    Returns
    -------
    dict con chiavi: precision, recall, f1, gmean
         ognuna = {'scores': list, 'mean': float, 'std': float}
    """
    sss = StratifiedShuffleSplit(
        n_splits   = N_SPLITS,
        test_size  = TEST_SIZE,
        random_state = RANDOM_STATE,
    )

    scores = {"precision": [], "recall": [], "f1": [], "gmean": []}

    for split_i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = clone(model)

        if is_bnb_em:
            # Simula scenario semi-supervisionato:
            # rimuove UNLABELED_RATIO etichette dal training
            n_unlabeled = int(len(y_tr) * UNLABELED_RATIO)
            rng = np.random.RandomState(split_i)
            ul_idx = rng.choice(len(y_tr), n_unlabeled, replace=False)
            l_mask = np.ones(len(y_tr), dtype=bool)
            l_mask[ul_idx] = False

            X_l, y_l = X_tr[l_mask],   y_tr[l_mask]
            X_u       = X_tr[~l_mask]

            m.fit(X_l, y_l, X_u)
        else:
            m.fit(X_tr, y_tr)

        y_pred = m.predict(X_te)

        scores["precision"].append(
            precision_score(y_te, y_pred, average="macro",
                            zero_division=0)
        )
        scores["recall"].append(
            recall_score(y_te, y_pred, average="macro",
                         zero_division=0)
        )
        scores["f1"].append(
            f1_score(y_te, y_pred, average="macro",
                     zero_division=0)
        )
        scores["gmean"].append(
            geometric_mean_score(y_te, y_pred)
        )

    result = {}
    for metric, vals in scores.items():
        arr = np.array(vals)
        result[metric] = {
            "scores": vals,
            "mean":   float(arr.mean()),
            "std":    float(arr.std(ddof=1)),
        }
    return result


# ── Formattazione risultati ───────────────────────────────────────────────────

def print_results_table(all_results: dict):
    """
    Stampa una tabella con media ± std per ogni modello e metrica.
    Formato richiesto dalle linee-guida del progetto.
    """
    header = f"{'Modello':<22} {'Precision':>14} {'Recall':>14} {'F1':>14} {'G-Mean':>14}"
    sep    = "-" * len(header)
    print()
    print("=" * len(header))
    print("  RISULTATI VALUTAZIONE  (media ± std su 10 run)")
    print("=" * len(header))
    print(header)
    print(sep)
    for model_name, res in all_results.items():
        row = f"{model_name:<22}"
        for metric in ["precision", "recall", "f1", "gmean"]:
            m = res[metric]["mean"]
            s = res[metric]["std"]
            row += f"  {m:.3f}±{s:.3f}"
        print(row)
    print("=" * len(header))
    print()


# ── Test statistici ───────────────────────────────────────────────────────────

def statistical_tests(all_results: dict):
    """
    Friedman test su F1-macro + Nemenyi post-hoc se p < 0.05.
    Ritorna il DataFrame dei p-value Nemenyi (o None).
    """
    from scipy.stats import friedmanchisquare
    ph_df = None
    try:
        import scikit_posthocs as sp
        HAS_SP = True
    except ImportError:
        HAS_SP = False
        print("  [WARNING] scikit-posthocs non disponibile; test Nemenyi saltato.")

    names  = list(all_results.keys())
    scores = [all_results[n]["f1"]["scores"] for n in names]

    stat, p = friedmanchisquare(*scores)
    print(f"  Friedman test (F1-macro)  χ²={stat:.3f}  p={p:.4f}")

    if p < 0.05:
        print("  → differenze statisticamente significative (α=0.05)")
        if HAS_SP:
            mat  = np.array(scores).T
            ph   = sp.posthoc_nemenyi_friedman(mat)
            ph.columns = names
            ph.index   = names
            ph_df = ph
            print()
            print("  Matrice p-value Nemenyi (prime righe):")
            print(ph.round(3).to_string())
    else:
        print("  → nessuna differenza significativa (α=0.05)")
    print()
    return ph_df

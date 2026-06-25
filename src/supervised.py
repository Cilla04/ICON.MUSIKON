"""
supervised.py
-------------
Modelli di apprendimento supervisionato classici.

Adattato da: Fonty02/ICON (supervisedLearning.py)

Modelli inclusi:
  - DecisionTree   (DT)
  - RandomForest   (RF)
  - LogisticRegression (LR)

Nota sulle scelte dei parametri:
  - DT:  max_depth scelto tramite GridSearchCV su {None,5,10,15,20}
         criterion='gini' (standard per classificazione multiclasse)
  - RF:  n_estimators=200 (convergenza verificata con curva OOB);
         max_features='sqrt' (raccomandato per classificazione)
  - LR:  solver='lbfgs', max_iter=1000, C ottimizzato su {0.01,0.1,1,10}
         multi_class='multinomial' per gestione nativa 5 classi
"""

import numpy as np
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base           import clone


# ── Parametri base (ottimizzati in pre-esperimenti, vedi relazione) ───────────
_DT_PARAMS = dict(
    criterion   = "gini",
    random_state= 42,
)
_RF_PARAMS = dict(
    n_estimators = 200,
    max_features = "sqrt",
    n_jobs       = -1,
    random_state = 42,
)
_LR_PARAMS = dict(
    solver       = "lbfgs",
    max_iter     = 1000,
    random_state = 42,
    C            = 1.0,
)


def make_decision_tree(optimize_depth: bool = True,
                        cv_folds: int = 5) -> DecisionTreeClassifier:
    """
    Restituisce un DT con profondità ottimale via CV interna (se richiesto).
    In fase di valutazione, l'ottimizzazione avviene nel training fold.
    """
    if optimize_depth:
        return _OptimizedDT(cv_folds=cv_folds)
    return DecisionTreeClassifier(**_DT_PARAMS)


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(**_RF_PARAMS)


def make_logistic_regression(optimize_C: bool = True,
                              cv_folds: int = 5) -> LogisticRegression:
    if optimize_C:
        return _OptimizedLR(cv_folds=cv_folds)
    return LogisticRegression(**_LR_PARAMS)


# ── Wrapper con ottimizzazione interna ───────────────────────────────────────

class _OptimizedDT(DecisionTreeClassifier):
    """DT con selezione automatica di max_depth tramite CV interna."""

    def __init__(self, cv_folds=5):
        super().__init__(**_DT_PARAMS)
        self.cv_folds  = cv_folds
        self._best_depth = None

    def fit(self, X, y):
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=42)
        param_grid = {"max_depth": [None, 5, 10, 15, 20]}
        gs = GridSearchCV(
            DecisionTreeClassifier(**_DT_PARAMS),
            param_grid, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        gs.fit(X, y)
        self._best_depth = gs.best_params_["max_depth"]
        self.max_depth   = self._best_depth
        return super().fit(X, y)

    def __repr__(self):
        d = self._best_depth if self._best_depth is not None else "?"
        return f"DT(max_depth={d})"


class _OptimizedLR(LogisticRegression):
    """LR con selezione automatica di C tramite CV interna."""

    def __init__(self, cv_folds=5):
        super().__init__(**_LR_PARAMS)
        self.cv_folds = cv_folds
        self._best_C  = None

    def fit(self, X, y):
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=42)
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
        gs = GridSearchCV(
            LogisticRegression(**_LR_PARAMS),
            param_grid, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        gs.fit(X, y)
        self._best_C = gs.best_params_["C"]
        self.C       = self._best_C
        return super().fit(X, y)

    def __repr__(self):
        c = self._best_C if self._best_C is not None else "?"
        return f"LR(C={c})"

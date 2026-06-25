"""
main.py  –  Pipeline principale di MUSIKON
==========================================
Music Knowledge-Based Classification System

Università degli Studi di Bari Aldo Moro  –  ICon 2023/2024

Repository di riferimento:
  - MBM-EM : THESCREAMINGMONKEY/MBM-EM   → BNB, BNB-EM
  - ICON   : Fonty02/ICON                → dataset Spotify, BN, KB Prolog
  - Docs   : bralani/icon22-23/docs      → linee-guida valutazione

Flusso:
  1. Preprocessing       → X_raw (10), X_owl (25 BK), X_all (35), y
  2. Analisi BK          → coverage concetti OWL per classe
  3. Prolog KB           → classificazione logica (opzionale, richiede SWI-Prolog)
  4. Valutazione modelli → tabella mean ± std su 10 run
  5. Test statistici     → Friedman + Nemenyi post-hoc
  6. Grafici             → barplot, heatmap Nemenyi, distribuzione OWL, boxplot
  7. Export              → results/results.csv
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import RESULTS_DIR, CLASS_NAMES, OWL_FEATURES
from src.preprocessing      import load_and_preprocess, summarize_dataset
from src.ontology_features  import owl_coverage_report
from src.bernoulli_nb       import BernoulliNB
from src.bnb_em             import BNB_EM
from src.supervised         import make_decision_tree, make_random_forest, make_logistic_regression
from src.evaluation         import evaluate_model, print_results_table, statistical_tests
from src.visualization      import plot_metrics, plot_nemenyi, plot_owl_heatmap, plot_f1_boxplot

# ── Bayesian Network (richiede pgmpy) ────────────────────────────────────────
try:
    from src.bayesian_network import MusicBayesianNet, PGMPY_OK
except ImportError:
    PGMPY_OK = False

# ── Prolog KB (richiede pyswip + SWI-Prolog) ─────────────────────────────────
try:
    from src.prolog_kb import MusicKB, SWIP_OK
except Exception:
    SWIP_OK = False


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  MUSIKON – Music Knowledge-Based Classifier      ║")
    print("║  ICon 2023/2024  –  Uniba                        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # ── 1. PREPROCESSING ─────────────────────────────────────────────────────
    print("[1/6] Caricamento e preprocessing...")
    X_raw, X_owl, X_all, y, feat_raw, feat_owl = load_and_preprocess()
    summarize_dataset(X_raw, X_owl, y)

    # ── 2. ANALISI BACKGROUND KNOWLEDGE ──────────────────────────────────────
    print("[2/6] Analisi copertura concetti OWL (Background Knowledge)...")
    df_owl = pd.DataFrame(X_owl, columns=feat_owl)
    cov    = owl_coverage_report(df_owl, y, CLASS_NAMES)

    variances = cov.var(axis=0).sort_values(ascending=False)
    top10     = variances.head(10).index.tolist()
    print()
    print("  Top-10 concetti OWL per varianza inter-classe (%):")
    print(cov[top10].round(1).to_string())
    print()

    # Salva heatmap completa BK
    print("  Generazione heatmap BK...")
    plot_owl_heatmap(X_owl, y, output_prefix="owl_heatmap")

    # ── 3. KB PROLOG (opzionale) ──────────────────────────────────────────────
    print(f"\n[3/6] Knowledge Base Prolog... ", end="")
    if SWIP_OK:
        from src.prolog_kb import MusicKB
        kb = MusicKB()
        if kb.available:
            sample = pd.read_csv("data/playlist_tracks.csv").head(20)
            sample["loudness_norm"] = ((sample["loudness"] + 30) / 30).clip(0, 1)
            sample["tempo_norm"]    = ((sample["tempo"] - 54) / 156).clip(0, 1)
            preds = kb.batch_classify(sample)
            n_ok  = sum(1 for p in preds if p is not None)
            print(f"OK ({n_ok}/20 tracce classificate)")
        else:
            print("SWI-Prolog non trovato → SKIP")
    else:
        print("pyswip non installato → SKIP")

    # ── 4. VALUTAZIONE MODELLI ────────────────────────────────────────────────
    print(f"\n[4/6] Valutazione modelli (10×StratifiedShuffleSplit, test=30%)...")
    print()

    models = {
        # ── Baseline: feature continue raw ───────────────────────────────
        "DT  (raw)":   (make_decision_tree(),       X_raw, False),
        "RF  (raw)":   (make_random_forest(),        X_raw, False),
        "LR  (raw)":   (make_logistic_regression(),  X_raw, False),
        # ── Con Background Knowledge OWL ─────────────────────────────────
        "DT  (+OWL)":  (make_decision_tree(),       X_all, False),
        "RF  (+OWL)":  (make_random_forest(),        X_all, False),
        "LR  (+OWL)":  (make_logistic_regression(),  X_all, False),
        # ── Su feature OWL binarie (BK pura) ─────────────────────────────
        "BNB (OWL)":   (BernoulliNB(),               X_owl, False),
        "BNB-EM(OWL)": (BNB_EM(),                    X_owl, True),
    }

    all_results = {}
    for name, (mdl, X_feat, is_em) in models.items():
        sys.stdout.write(f"  → {name:20s} ")
        sys.stdout.flush()
        res = evaluate_model(mdl, X_feat, y, is_bnb_em=is_em)
        all_results[name] = res
        print(f"  F1={res['f1']['mean']:.3f}±{res['f1']['std']:.3f}  "
              f"G-Mean={res['gmean']['mean']:.3f}±{res['gmean']['std']:.3f}")

    # Bayesian Network (lenta, opzionale)
    if PGMPY_OK:
        sys.stdout.write(f"  → {'BN  (OWL)':20s} ")
        sys.stdout.flush()
        try:
            res = evaluate_model(MusicBayesianNet(), X_owl, y)
            all_results["BN  (OWL)"] = res
            print(f"  F1={res['f1']['mean']:.3f}±{res['f1']['std']:.3f}  "
                  f"G-Mean={res['gmean']['mean']:.3f}±{res['gmean']['std']:.3f}")
        except Exception as e:
            print(f"  ERRORE: {e}")

    # ── 5. REPORT + TEST STATISTICI ──────────────────────────────────────────
    print(f"\n[5/6] Report statistico...")
    print_results_table(all_results)

    print("  Test di significatività statistica (Friedman + Nemenyi):")
    ph_df = statistical_tests(all_results)

    # ── 6. GRAFICI ────────────────────────────────────────────────────────────
    print(f"[6/6] Generazione grafici...")
    plot_metrics(all_results,   output_prefix="metrics")
    plot_f1_boxplot(all_results, output_prefix="f1_boxplot")
    if ph_df is not None:
        plot_nemenyi(ph_df, output_prefix="nemenyi")

    # ── Export CSV ────────────────────────────────────────────────────────────
    rows = []
    for model_name, res in all_results.items():
        row = {"modello": model_name}
        for metric in ["precision", "recall", "f1", "gmean"]:
            row[f"{metric}_mean"] = round(res[metric]["mean"], 4)
            row[f"{metric}_std"]  = round(res[metric]["std"],  4)
        rows.append(row)
    out_csv = os.path.join(RESULTS_DIR, "results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  CSV  → {out_csv}")
    print(f"  Plot → {RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()

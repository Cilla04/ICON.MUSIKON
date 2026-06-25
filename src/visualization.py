"""
visualization.py
----------------
Grafici per l'analisi dei risultati MUSIKON:
  1. Barplot mean ± std per ogni metrica (confronto modelli)
  2. Heatmap post-hoc Nemenyi
  3. Heatmap distribuzione feature OWL per classe (Background Knowledge)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, CLASS_NAMES, OWL_FEATURES

PALETTE_RAW  = "#2196F3"   # blu  → modelli su feature raw
PALETTE_OWL  = "#FF9800"   # arancio → modelli su feature OWL / combinate
PALETTE_BK   = "#4CAF50"   # verde → modelli puri su BK


def _bar_color(name: str) -> str:
    if "raw" in name.lower():
        return PALETTE_RAW
    if any(t in name for t in ["BNB", "BN "]):
        return PALETTE_BK
    return PALETTE_OWL


# ── 1. Barplot metriche ──────────────────────────────────────────────────────

def plot_metrics(all_results: dict, output_prefix: str = "metrics") -> None:
    """
    Griglia 2×2 con barplot (mean ± std) per P / R / F1 / G-Mean.
    Salva in RESULTS_DIR/<output_prefix>.png
    """
    metrics = ["precision", "recall", "f1", "gmean"]
    labels  = ["Precision (macro)", "Recall (macro)", "F1 (macro)", "G-Mean (macro)"]
    names   = list(all_results.keys())
    colors  = [_bar_color(n) for n in names]
    x       = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("MUSIKON — Confronto modelli (media ± std, 10 run)",
                 fontsize=14, fontweight="bold")

    for ax, metric, label in zip(axes.flat, metrics, labels):
        means = [all_results[n][metric]["mean"] for n in names]
        stds  = [all_results[n][metric]["std"]  for n in names]
        bars  = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                       capsize=5, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Score")
        # Annotazione del valore medio sopra ogni barra
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.012,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)

    # Legenda colori
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=PALETTE_RAW, label="Feature continue raw"),
        mpatches.Patch(color=PALETTE_OWL, label="Feature raw + OWL BK"),
        mpatches.Patch(color=PALETTE_BK,  label="Solo feature OWL (BK)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    out = os.path.join(RESULTS_DIR, f"{output_prefix}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Metriche salvato → {out}")


# ── 2. Heatmap Nemenyi ──────────────────────────────────────────────────────

def plot_nemenyi(ph_df: pd.DataFrame,
                 output_prefix: str = "nemenyi") -> None:
    """
    Heatmap colorata dei p-value del test post-hoc Nemenyi.
    Verde scuro = non significativo (p>0.05), rosso = significativo (p<0.05).
    """
    if ph_df is None or ph_df.empty:
        return

    n   = len(ph_df)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(6, n * 0.75)))
    data = ph_df.values.astype(float)

    # Colormap personalizzata: verde=non-sig, rosso=sig
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "nemenyi", ["#e53935", "#ffee58", "#43a047"], N=256
    )
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=0.2, aspect="auto")
    plt.colorbar(im, ax=ax, label="p-value", shrink=0.8)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ph_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ph_df.index, fontsize=8)

    for i in range(n):
        for j in range(n):
            p = data[i, j]
            sig  = "✓" if p >= 0.05 else "✗"
            col  = "black" if p > 0.08 else "white"
            ax.text(j, i, f"{p:.3f}\n{sig}",
                    ha="center", va="center", fontsize=7.5, color=col)

    ax.set_title("Nemenyi Post-Hoc — F1 macro\n"
                 "✓ p≥0.05 (non sign.)  ✗ p<0.05 (sign.)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(RESULTS_DIR, f"{output_prefix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Nemenyi   salvato → {out}")


# ── 3. Heatmap distribuzione OWL per classe ──────────────────────────────────

def plot_owl_heatmap(X_owl: np.ndarray, y: np.ndarray,
                     output_prefix: str = "owl_heatmap") -> None:
    """
    P(concetto OWL = 1 | classe): heatmap (classi × concetti).
    Evidenzia quali concetti BK sono discriminanti per ogni playlist.
    """
    classes    = sorted(np.unique(y))
    n_classes  = len(classes)
    n_concepts = len(OWL_FEATURES)

    prob = np.zeros((n_classes, n_concepts))
    for i, c in enumerate(classes):
        prob[i] = X_owl[y == c].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(14, n_concepts * 0.55), 4.5))
    im = ax.imshow(prob, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="P(concept=1 | class)", shrink=0.85)

    ax.set_xticks(range(n_concepts))
    ax.set_xticklabels(OWL_FEATURES, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    # Annotazione valori
    for i in range(n_classes):
        for j in range(n_concepts):
            col = "white" if prob[i, j] > 0.65 else "black"
            ax.text(j, i, f"{prob[i,j]:.2f}",
                    ha="center", va="center", fontsize=6.5, color=col)

    ax.set_title("Coverage concetti OWL (Background Knowledge) per classe\n"
                 "P(concept=1 | class) — valori alti = alta discriminabilità",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(RESULTS_DIR, f"{output_prefix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] OWL BK    salvato → {out}")


# ── 4. Boxplot distribuzioni F1 per modello ──────────────────────────────────

def plot_f1_boxplot(all_results: dict, output_prefix: str = "f1_boxplot") -> None:
    """
    Boxplot delle distribuzioni F1 su 10 run per ogni modello.
    Mostra variabilità e outlier.
    """
    names  = list(all_results.keys())
    data   = [all_results[n]["f1"]["scores"] for n in names]
    colors = [_bar_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("F1 macro", fontsize=11)
    ax.set_title("Distribuzione F1 macro su 10 run (StratifiedShuffleSplit)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Legenda
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=PALETTE_RAW, label="Feature raw"),
        mpatches.Patch(color=PALETTE_OWL, label="Raw + OWL BK"),
        mpatches.Patch(color=PALETTE_BK,  label="Solo OWL BK"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{output_prefix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Boxplot   salvato → {out}")

# Confusion matrices for all models (gender + age) on LiLaH EN.
# Outputs saved to analysis/figures/confusion_matrices/

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

plt.switch_backend("Agg")

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
LLM_DIR = os.path.join(os.path.dirname(__file__), "..", "LLM_code")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "confusion_matrices")
os.makedirs(OUT_DIR, exist_ok=True)

GENDER_LABELS = ["F", "M"]
AGE_LABELS = ["0-25", "26-35", "36-65", "66-"]

# (display name, gender_file, gender_pred_col, age_file, age_pred_col)
MODELS = [
    (
        "SVM",
        "lilah_gender_predictions_svm.csv",
        "predicted_label",
        "lilah_age_predictions_svm.csv",
        "predicted_label",
    ),
    (
        "BERT",
        "BERT-gender_predictions.csv",
        "predicted",
        "BERT-age_predictions.csv",
        "predicted",
    ),
    (
        "HateBERT",
        "hateBERT-gender_predictions.csv",
        "predicted",
        "hateBERT-age_predictions.csv",
        "predicted",
    ),
    (
        "RoBERTa",
        "roberta_gender_predictions.csv",
        "pred_gender",
        "roberta_age_predictions.csv",
        "pred_age",
    ),
    (
        "RoBERTa-v2",
        "roberta_v2_gender_predictions.csv",
        "pred_gender",
        "roberta_v2_age_predictions.csv",
        "pred_age",
    ),
    (
        "LLaMA-3.1",
        "llm_final_results.csv",
        "pred_gender",
        "llm_final_results.csv",
        "pred_age",
        LLM_DIR,
    ),
    (
        "Qwen3-32B",
        "llm_final_results_qwen.csv",
        "pred_gender",
        "llm_final_results_qwen.csv",
        "pred_age",
        LLM_DIR,
    ),
]


def load(fname, pred_col, true_col, base_dir=None):
    base_dir = base_dir or RESULTS
    df = pd.read_csv(os.path.join(base_dir, fname))
    df = df.dropna(subset=[pred_col, true_col])
    return df[true_col].tolist(), df[pred_col].tolist()


def plot_cm(ax, y_true, y_pred, labels, title, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # row-normalise so colour = recall per class
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    # annotate with count (top) and recall % (bottom)
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            pct = cm_norm[i, j]
            colour = "white" if pct > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.0%})",
                ha="center",
                va="center",
                fontsize=8,
                color=colour,
                fontweight="bold" if i == j else "normal",
            )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    return im


def unpack(row):
    """Return (name, gf, gc, af, ac, base_dir) with optional 6th element."""
    if len(row) == 6:
        return row
    return (*row, RESULTS)


# ── Gender: 2×4 grid ──────────────────────────────────────────────────────────
print("Generating gender confusion matrices...")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, row in zip(axes.flat, MODELS):
    name, gf, gc, af, ac, bdir = unpack(row)
    y_true, y_pred = load(gf, gc, "gender", bdir)
    present = sorted(set(y_true) | set(y_pred))
    labels = [lbl for lbl in GENDER_LABELS if lbl in present]
    plot_cm(ax, y_true, y_pred, labels, name)
    f_correct = sum(1 for t, p in zip(y_true, y_pred) if t == "F" and p == "F")
    f_total = max(sum(1 for t in y_true if t == "F"), 1)
    print(f"  {name}: F recall={f_correct / f_total:.2f}")

# hide unused subplot if MODELS has fewer than 8 entries
for ax in axes.flat[len(MODELS) :]:
    ax.set_visible(False)

fig.suptitle(
    "Gender Confusion Matrices — LiLaH EN Hate Speech\n(cell = count / row recall %)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
out = os.path.join(OUT_DIR, "cm_gender_all.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ── Age: 2×4 grid ─────────────────────────────────────────────────────────────
print("\nGenerating age confusion matrices...")
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
for ax, row in zip(axes.flat, MODELS):
    name, gf, gc, af, ac, bdir = unpack(row)
    y_true, y_pred = load(af, ac, "age", bdir)
    present = sorted(set(y_true) | set(y_pred))
    labels = [lbl for lbl in AGE_LABELS if lbl in present]
    plot_cm(ax, y_true, y_pred, labels, name, cmap="Purples")
    rec_66 = sum(1 for t, p in zip(y_true, y_pred) if t == "66-" and p == "66-") / max(
        sum(1 for t in y_true if t == "66-"), 1
    )
    print(f"  {name}: 66- recall={rec_66:.2f}")

for ax in axes.flat[len(MODELS) :]:
    ax.set_visible(False)

fig.suptitle(
    "Age Confusion Matrices — LiLaH EN Hate Speech\n(cell = count / row recall %)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
out = os.path.join(OUT_DIR, "cm_age_all.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ── Individual high-res CMs for BERT and LLaMA (thesis figures) ───────────────
print("\nGenerating individual high-res CMs...")
for task, labels, cmap in [
    ("gender", GENDER_LABELS, "Blues"),
    ("age", AGE_LABELS, "Purples"),
]:
    for row in MODELS:
        name, gf, gc, af, ac, bdir = unpack(row)
        if name not in ("BERT", "LLaMA-3.1"):
            continue
        fname = gf if task == "gender" else af
        col = gc if task == "gender" else ac
        y_true, y_pred = load(fname, col, task, bdir)
        present = sorted(set(y_true) | set(y_pred))
        lbls = [lbl for lbl in labels if lbl in present]

        fig, ax = plt.subplots(figsize=(5 if task == "gender" else 6, 4.5))
        plot_cm(
            ax,
            y_true,
            y_pred,
            lbls,
            f"{name} — {task.capitalize()} (LiLaH EN)",
            cmap=cmap,
        )

        # print classification report
        print(f"\n  {name} {task}:")
        print(
            classification_report(
                y_true, y_pred, labels=lbls, digits=3, zero_division=0
            )
        )

        plt.tight_layout()
        safe = name.lower().replace("-", "_").replace(".", "")
        out = os.path.join(OUT_DIR, f"cm_{task}_{safe}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

print("\nDone. All confusion matrices saved to analysis/figures/confusion_matrices/")

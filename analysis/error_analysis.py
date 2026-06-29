# Error analysis: generates confusion matrices and error figures.
# Outputs saved to analysis/figures/

import io
import os
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

GENDER_LABELS = ["F", "M"]
AGE_LABELS = ["0-25", "26-35", "36-65", "66-"]

# Load predictions


def load_all():
    base = os.path.join(os.path.dirname(__file__), "..")

    def path(*parts):
        return os.path.join(base, *parts)

    # Age
    bert_a = pd.read_csv(path("results", "BERT-age_predictions.csv"))
    bert_a = bert_a[["id", "text", "age", "gender", "predicted"]].rename(
        columns={"predicted": "bert"}
    )

    hate_a = pd.read_csv(path("results", "hateBERT-age_predictions.csv"))
    hate_a = hate_a[["id", "predicted"]].rename(columns={"predicted": "hatebert"})

    rob_a = pd.read_csv(path("results", "roberta_age_predictions.csv"))
    rob_a = rob_a[["id", "pred_age"]].rename(columns={"pred_age": "roberta"})

    rob2_a = pd.read_csv(path("results", "roberta_v2_age_predictions.csv"))
    rob2_a = rob2_a[["id", "pred_age"]].rename(columns={"pred_age": "roberta_v2"})

    llama = pd.read_csv(path("LLM_code", "llm_final_results.csv"))
    llama_a = llama[["id", "pred_age", "pred_gender"]].rename(
        columns={"pred_age": "llama_age", "pred_gender": "llama_gender"}
    )

    qwen = pd.read_csv(path("LLM_code", "llm_final_results_qwen.csv"))
    qwen_a = qwen[["id", "pred_age", "pred_gender"]].rename(
        columns={"pred_age": "qwen_age", "pred_gender": "qwen_gender"}
    )

    age = (
        bert_a.merge(hate_a, "left", "id")
        .merge(rob_a, "left", "id")
        .merge(rob2_a, "left", "id")
        .merge(llama_a, "left", "id")
        .merge(qwen_a, "left", "id")
    )

    # Gender
    bert_g = pd.read_csv(path("results", "BERT-gender_predictions.csv"))
    bert_g = bert_g[["id", "predicted"]].rename(columns={"predicted": "bert_g"})

    hate_g = pd.read_csv(path("results", "hateBERT-gender_predictions.csv"))
    hate_g = hate_g[["id", "predicted"]].rename(columns={"predicted": "hatebert_g"})

    rob_g = pd.read_csv(path("results", "roberta_gender_predictions.csv"))
    rob_g = rob_g[["id", "pred_gender"]].rename(columns={"pred_gender": "roberta_g"})

    rob2_g = pd.read_csv(path("results", "roberta_v2_gender_predictions.csv"))
    rob2_g = rob2_g[["id", "pred_gender"]].rename(
        columns={"pred_gender": "roberta_v2_g"}
    )

    gender = (
        age[["id", "age", "gender", "text"]]
        .merge(bert_g, "left", "id")
        .merge(hate_g, "left", "id")
        .merge(rob_g, "left", "id")
        .merge(rob2_g, "left", "id")
        .merge(llama_a[["id", "llama_gender"]], "left", "id")
        .merge(qwen_a[["id", "qwen_gender"]], "left", "id")
    )

    age["wc"] = age["text"].str.split().str.len()
    gender["wc"] = gender["text"].str.split().str.len()

    return age, gender


AGE_MODEL_COLS = ["bert", "hatebert", "roberta", "roberta_v2", "llama_age", "qwen_age"]
AGE_MODEL_NAMES = [
    "BERT",
    "HateBERT",
    "RoBERTa",
    "RoBERTa-v2",
    "LLaMA-3.1",
    "Qwen3-32B",
]
GENDER_MODEL_COLS = [
    "bert_g",
    "hatebert_g",
    "roberta_g",
    "roberta_v2_g",
    "llama_gender",
    "qwen_gender",
]
GENDER_MODEL_NAMES = [
    "BERT",
    "HateBERT",
    "RoBERTa",
    "RoBERTa-v2",
    "LLaMA-3.1",
    "Qwen3-32B",
]

# Analysis 1 — Prediction Distribution Bias


def analysis1_distribution(age, gender):
    print("\n" + "=" * 70)
    print("ANALYSIS 1 — PREDICTION DISTRIBUTION")
    print("=" * 70)

    # AGE
    true_age_counts = age["age"].value_counts().reindex(AGE_LABELS, fill_value=0)
    print("\nTrue age distribution:")
    print(true_age_counts.to_string())

    rows = []
    for col, name in zip(AGE_MODEL_COLS, AGE_MODEL_NAMES):
        pred = age[col].str.strip()
        counts = pred.value_counts().reindex(AGE_LABELS, fill_value=0)
        bias = counts - true_age_counts
        row = {"Model": name}
        for lbl in AGE_LABELS:
            row[f"pred_{lbl}"] = counts[lbl]
            row[f"bias_{lbl}"] = bias[lbl]
        rows.append(row)

    dist_age = pd.DataFrame(rows)
    print("\nPredicted counts (age):")
    print(
        dist_age[["Model"] + [f"pred_{l}" for l in AGE_LABELS]].to_string(index=False)
    )
    print("\nBias = predicted - true (age):")
    print(
        dist_age[["Model"] + [f"bias_{l}" for l in AGE_LABELS]].to_string(index=False)
    )

    # Plot: grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(AGE_LABELS))
    width = 0.12
    ax = axes[0]
    # True distribution bar
    ax.bar(
        x - 3 * width,
        true_age_counts.values,
        width,
        label="True",
        color="black",
        alpha=0.6,
    )
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    for i, (col, name) in enumerate(zip(AGE_MODEL_COLS, AGE_MODEL_NAMES)):
        pred = age[col].str.strip()
        counts = pred.value_counts().reindex(AGE_LABELS, fill_value=0)
        ax.bar(
            x + (i - 2) * width,
            counts.values,
            width,
            label=name,
            color=colors[i],
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS)
    ax.set_title("Predicted vs True — Age")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # GENDER distribution
    true_g_counts = (
        gender["gender"].str.upper().value_counts().reindex(GENDER_LABELS, fill_value=0)
    )
    ax = axes[1]
    xg = np.arange(len(GENDER_LABELS))
    ax.bar(
        xg - 3 * width,
        true_g_counts.values,
        width,
        label="True",
        color="black",
        alpha=0.6,
    )
    for i, (col, name) in enumerate(zip(GENDER_MODEL_COLS, GENDER_MODEL_NAMES)):
        pred = gender[col].str.strip().str.upper()
        counts = pred.value_counts().reindex(GENDER_LABELS, fill_value=0)
        ax.bar(
            xg + (i - 2) * width,
            counts.values,
            width,
            label=name,
            color=colors[i],
            alpha=0.85,
        )
    ax.set_xticks(xg)
    ax.set_xticklabels(GENDER_LABELS)
    ax.set_title("Predicted vs True — Gender")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "fig1_prediction_distribution.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fpath}")


# Analysis 2 — Confusion Matrices


def plot_cm(cm, labels, title, fpath, fmt="d"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11)
    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()


def analysis2_confusion(age, gender):
    print("\n" + "=" * 70)
    print("ANALYSIS 2 — CONFUSION MATRICES")
    print("=" * 70)

    # Age: BERT and LLaMA (best encoders vs best LLM)
    for col, name in zip(AGE_MODEL_COLS, AGE_MODEL_NAMES):
        v = age.dropna(subset=[col])
        v = v[v[col].isin(AGE_LABELS)]
        cm = confusion_matrix(v["age"], v[col], labels=AGE_LABELS)
        fname = f"fig2_cm_age_{name.lower().replace('-','_').replace('.','')}.png"
        plot_cm(cm, AGE_LABELS, f"Age — {name}", os.path.join(OUT_DIR, fname))

        # Print the 66- row
        idx = AGE_LABELS.index("66-")
        row66 = cm[idx]
        total66 = row66.sum()
        print(
            f"{name:12s} | 66- row: "
            + " | ".join(
                f"{l}={row66[i]} ({100*row66[i]/max(total66,1):.0f}%)"
                for i, l in enumerate(AGE_LABELS)
            )
        )

    print()
    # Gender: all models
    for col, name in zip(GENDER_MODEL_COLS, GENDER_MODEL_NAMES):
        v = gender.copy()
        v["gold_g"] = v["gender"].str.strip().str.upper()
        v[col] = v[col].astype(str).str.strip().str.upper()
        v = v[v[col].isin(GENDER_LABELS)]
        cm = confusion_matrix(v["gold_g"], v[col], labels=GENDER_LABELS)
        fname = f"fig2_cm_gender_{name.lower().replace('-','_').replace('.','')}.png"
        plot_cm(cm, GENDER_LABELS, f"Gender — {name}", os.path.join(OUT_DIR, fname))

        # Print F row
        row_f = cm[0]
        total_f = row_f.sum()
        print(
            f"{name:12s} | F row: F={row_f[0]} ({100*row_f[0]/max(total_f,1):.0f}%), "
            f"M={row_f[1]} ({100*row_f[1]/max(total_f,1):.0f}%)"
        )

    # Composite figure: BERT + LLaMA age CMs side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, name in zip(axes, ["bert", "llama_age"], ["BERT", "LLaMA-3.1"]):
        v = age[age[col].isin(AGE_LABELS)]
        cm = confusion_matrix(v["age"], v[col], labels=AGE_LABELS)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(AGE_LABELS, fontsize=8)
        ax.set_yticklabels(AGE_LABELS, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Age — {name}")
        thresh = cm.max() / 2
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "fig2_cm_age_bert_llama.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved composite: {fpath}")


# Analysis 3 — 66– Deep Dive


def analysis3_66(age):
    print("\n" + "=" * 70)
    print("ANALYSIS 3 — 66– CLASS DEEP DIVE")
    print("=" * 70)

    df66 = age[age["age"] == "66-"].copy()
    print(f"Total 66- examples: {len(df66)}")
    print(f"Gender split: {dict(df66['gender'].value_counts())}")
    print(
        f"Avg word count: {df66['wc'].mean():.1f} (all LiLaH: {age['wc'].mean():.1f})"
    )

    print("\nPrediction distribution for true 66- examples:")
    for col, name in zip(AGE_MODEL_COLS, AGE_MODEL_NAMES):
        counts = df66[col].value_counts().reindex(AGE_LABELS, fill_value=0)
        print(f"  {name:12s}: " + " | ".join(f"{l}={counts[l]}" for l in AGE_LABELS))

    # Cross-model correctness
    correct = {col: (df66[col] == "66-") for col in AGE_MODEL_COLS}
    correct_df = pd.DataFrame(correct)
    n_correct_per_example = correct_df.sum(axis=1)

    print(f"\n66- examples: no model correct    = {(n_correct_per_example == 0).sum()}")
    print(
        f"66- examples: only BERT correct    = {(correct_df['bert'] & ~correct_df[['hatebert','roberta','roberta_v2','llama_age','qwen_age']].any(axis=1)).sum()}"
    )
    print(f"66- examples: 2+ models correct    = {(n_correct_per_example >= 2).sum()}")

    # BERT correct vs wrong breakdown
    bert_right = df66[df66["bert"] == "66-"]
    bert_wrong = df66[df66["bert"] != "66-"]
    print(f"\nBERT: correct={len(bert_right)}, wrong={len(bert_wrong)}")
    print(
        f"  Correct avg wc: {bert_right['wc'].mean():.1f}, wrong avg wc: {bert_wrong['wc'].mean():.1f}"
    )
    print(f"  Correct gender: {dict(bert_right['gender'].value_counts())}")
    print(f"  Wrong gender:   {dict(bert_wrong['gender'].value_counts())}")
    print(f"  Wrong predicted as: {dict(bert_wrong['bert'].value_counts())}")

    # Gender breakdown for 66-
    print("\n66- by author gender:")
    for g in ["M", "F"]:
        sub = df66[df66["gender"] == g]
        bert_ok = (sub["bert"] == "66-").sum()
        llama_ok = (sub["llama_age"] == "66-").sum()
        print(
            f"  {g}: n={len(sub)}, BERT_correct={bert_ok} ({100*bert_ok/len(sub):.0f}%), LLaMA_correct={llama_ok}"
        )

    # Bar chart: 66- predictions per model
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(AGE_LABELS))
    width = 0.12
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    for i, (col, name) in enumerate(zip(AGE_MODEL_COLS, AGE_MODEL_NAMES)):
        counts = df66[col].value_counts().reindex(AGE_LABELS, fill_value=0)
        ax.bar(
            x + (i - 2.5) * width,
            counts.values,
            width,
            label=name,
            color=colors[i],
            alpha=0.85,
        )
    # Mark the true class
    ax.axvline(
        x=AGE_LABELS.index("66-"),
        color="red",
        linestyle="--",
        alpha=0.4,
        label="True class",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS)
    ax.set_title("Model predictions for true 66– examples (n=64)")
    ax.set_ylabel("Number of examples predicted as each class")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "fig3_66minus_predictions.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fpath}")

    # Qualitative examples: BERT correct, LLaMA wrong
    bert_right_llama_wrong = df66[
        (df66["bert"] == "66-") & (df66["llama_age"] != "66-")
    ]
    print(f"\nBERT correct / LLaMA wrong: {len(bert_right_llama_wrong)} examples")
    print("\nSample texts (BERT=correct, LLaMA=wrong):")
    for _, row in bert_right_llama_wrong.head(5).iterrows():
        print(
            f"  gender={row['gender']} | llama_pred={row['llama_age']} | wc={row['wc']}"
        )
        print(f"  TEXT: {str(row['text'])[:200]}")
        print()

    # Qualitative examples: no model correct
    all_wrong_66 = df66[n_correct_per_example == 0]
    print(f"\nAll models wrong — 66- examples (n={len(all_wrong_66)}):")
    print("Sample texts:")
    for _, row in all_wrong_66.head(5).iterrows():
        preds = {name: row[col] for col, name in zip(AGE_MODEL_COLS, AGE_MODEL_NAMES)}
        print(f"  gender={row['gender']} | wc={row['wc']} | preds={preds}")
        print(f"  TEXT: {str(row['text'])[:200]}")
        print()

    return df66, bert_right, all_wrong_66


# Analysis 4 — Text Length vs. Accuracy


def analysis4_length(age, gender):
    print("\n" + "=" * 70)
    print("ANALYSIS 4 — TEXT LENGTH VS. ACCURACY")
    print("=" * 70)

    bins = [0, 15, 30, 60, 999]
    labels_bins = ["1–15", "16–30", "31–60", "61+"]
    age["wc_bucket"] = pd.cut(age["wc"], bins=bins, labels=labels_bins)

    print("\nBERT age accuracy by word count:")
    rows = []
    for bkt in labels_bins:
        sub = age[age["wc_bucket"] == bkt]
        acc_bert = (sub["bert"] == sub["age"]).mean()
        acc_llama = (sub["llama_age"] == sub["age"]).mean()
        mf1_bert = f1_score(
            sub["age"], sub["bert"], labels=AGE_LABELS, average="macro", zero_division=0
        )
        mf1_llama = f1_score(
            sub["age"],
            sub["llama_age"],
            labels=AGE_LABELS,
            average="macro",
            zero_division=0,
        )
        rows.append(
            {
                "bucket": bkt,
                "n": len(sub),
                "BERT_acc": round(acc_bert, 3),
                "BERT_macroF1": round(mf1_bert, 3),
                "LLaMA_acc": round(acc_llama, 3),
                "LLaMA_macroF1": round(mf1_llama, 3),
            }
        )
        print(
            f"  wc {bkt} (n={len(sub)}): BERT acc={acc_bert:.3f} macroF1={mf1_bert:.3f} | "
            f"LLaMA acc={acc_llama:.3f} macroF1={mf1_llama:.3f}"
        )

    df_len = pd.DataFrame(rows)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels_bins))
    width = 0.35
    ax.bar(x - width / 2, df_len["BERT_macroF1"], width, label="BERT", color="#4C72B0")
    ax.bar(
        x + width / 2,
        df_len["LLaMA_macroF1"],
        width,
        label="LLaMA-3.1",
        color="#8172B2",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={r})" for b, r in zip(labels_bins, df_len["n"])])
    ax.set_xlabel("Text length (words)")
    ax.set_ylabel("Macro F1 — age prediction")
    ax.set_title("Age prediction accuracy by text length")
    ax.legend()
    ax.set_ylim(0, 0.5)
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "fig4_length_vs_accuracy.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fpath}")


# Analysis 5 — Cross-Model Agreement


def analysis5_agreement(age, gender):
    print("\n" + "=" * 70)
    print("ANALYSIS 5 — CROSS-MODEL AGREEMENT")
    print("=" * 70)

    # Age
    correct_matrix = pd.DataFrame(
        {col: (age[col] == age["age"]) for col in AGE_MODEL_COLS}
    )
    age["n_correct"] = correct_matrix.sum(axis=1)

    print("\nAge — distribution of n_correct (0=all wrong, 6=all right):")
    print(age["n_correct"].value_counts().sort_index().to_string())

    # Profile easy vs hard
    easy = age[age["n_correct"] >= 5]
    hard = age[age["n_correct"] == 0]
    middle = age[(age["n_correct"] > 0) & (age["n_correct"] < 5)]

    print(f"\nEasy (5-6 correct): n={len(easy)}")
    print(f"  Age dist: {dict(easy['age'].value_counts())}")
    print(f"  Avg wc: {easy['wc'].mean():.1f}")
    print(f"  Gender: {dict(easy['gender'].value_counts())}")

    print(f"\nHard (0 correct): n={len(hard)}")
    print(f"  Age dist: {dict(hard['age'].value_counts())}")
    print(f"  Avg wc: {hard['wc'].mean():.1f}")
    print(f"  Gender: {dict(hard['gender'].value_counts())}")

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = age["n_correct"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Number of models correct (age prediction)")
    ax.set_ylabel("Number of examples")
    ax.set_title("Cross-model agreement — age (n=619)")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["0\n(all wrong)", "1", "2", "3", "4", "5", "6\n(all right)"])
    for i, v in enumerate(counts.values):
        ax.text(counts.index[i], v + 1, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "fig5_agreement_histogram.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fpath}")

    # Gender agreement
    correct_g = pd.DataFrame(
        {
            col: (
                gender[col].astype(str).str.strip().str.upper()
                == gender["gender"].str.strip().str.upper()
            )
            for col in GENDER_MODEL_COLS
        }
    )
    gender["n_correct_g"] = correct_g.sum(axis=1)
    print("\nGender — distribution of n_correct:")
    print(gender["n_correct_g"].value_counts().sort_index().to_string())


# Analysis 6 — Gender-Conditioned Age + Qualitative


def analysis6_gender_conditioned(age):
    print("\n" + "=" * 70)
    print("ANALYSIS 6 — GENDER-CONDITIONED AGE ACCURACY")
    print("=" * 70)

    for g in ["M", "F"]:
        sub = age[age["gender"] == g]
        print(f"\nAuthor gender = {g} (n={len(sub)}):")
        for col, name in zip(AGE_MODEL_COLS, AGE_MODEL_NAMES):
            v = sub[sub[col].isin(AGE_LABELS)]
            mf1 = f1_score(
                v["age"], v[col], labels=AGE_LABELS, average="macro", zero_division=0
            )
            f66 = f1_score(
                v["age"], v[col], labels=AGE_LABELS, average=None, zero_division=0
            )
            idx66 = AGE_LABELS.index("66-")
            print(f"  {name:12s}: macroF1={mf1:.3f}, F1_66-={f66[idx66]:.3f}")

    # Qualitative: gender errors — female comments predicted as male by all encoders
    # age already has llama_gender; load bert_g separately
    gdf = age.copy()
    bert_g_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "..", "results", "BERT-gender_predictions.csv"
        )
    )
    bert_g_df = bert_g_df[["id", "predicted"]].rename(columns={"predicted": "bert_g"})
    gdf = gdf.merge(bert_g_df, "left", "id")

    female_as_male = gdf[
        (gdf["gender"] == "F")
        & (gdf["bert_g"].astype(str).str.strip().str.upper() == "M")
        & (gdf["llama_gender"].astype(str).str.strip().str.upper() == "M")
    ]
    print(
        f"\nFemale comments predicted M by both BERT and LLaMA: {len(female_as_male)}"
    )
    print("Sample:")
    for _, row in female_as_male.head(5).iterrows():
        print(f"  age={row['age']} | wc={row['wc']}")
        print(f"  TEXT: {str(row['text'])[:200]}")
        print()


# Main


def main():
    print("Loading predictions...")
    age, gender = load_all()
    print(f"Loaded {len(age)} examples.")

    analysis1_distribution(age, gender)
    analysis2_confusion(age, gender)
    df66, bert_right_66, all_wrong_66 = analysis3_66(age)
    analysis4_length(age, gender)
    analysis5_agreement(age, gender)
    analysis6_gender_conditioned(age)

    print("\n" + "=" * 70)
    print(f"All figures saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

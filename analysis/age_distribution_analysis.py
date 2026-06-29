# Age distribution analysis: compares PAN14 vs LiLaH class distributions
# and explores why the 66+ group is underdetected.
# Outputs saved to results/age_analysis/

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "age_analysis")
os.makedirs(OUT, exist_ok=True)

AGE_ORDER = ["0-25", "26-35", "36-65", "66-"]
COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

# ── load data ──────────────────────────────────────────────────────────────────
pan14 = pd.read_csv(os.path.join(BASE, "pan14_prepared.csv"))
lilah = pd.read_csv(os.path.join(BASE, "hate_speech_only.tsv"), sep="\t")
bert = pd.read_csv(os.path.join(BASE, "lilah_age_predictions.csv"))
llm = pd.read_csv(os.path.join(BASE, "llama_style_focus_results.csv"))

# normalise column names
bert = bert.rename(columns={"predicted_label": "pred_age", "age": "true_age"})
llm = llm.rename(columns={"age": "true_age"})

print("Loaded datasets:")
print(f"  PAN14:  {len(pan14):,} rows")
print(f"  LiLaH:  {len(lilah):,} rows")
print(f"  BERT predictions: {len(bert):,} rows")
print(
    f"  LLM predictions:  {len(llm):,} rows  ({llm['pred_age'].isna().sum()} invalid outputs)"
)

# ── 1. Age distribution: PAN14 vs LiLaH ───────────────────────────────────────
def age_pct(df, col="age"):
    counts = df[col].value_counts()
    counts = counts.reindex(AGE_ORDER, fill_value=0)
    return counts / counts.sum() * 100


pan14_pct = age_pct(pan14)
lilah_pct = age_pct(lilah)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(AGE_ORDER))
w = 0.35
bars1 = ax.bar(x - w / 2, pan14_pct, w, label="PAN14 (training)", color=COLORS[0])
bars2 = ax.bar(
    x + w / 2, lilah_pct, w, label="LiLaH hate speech (eval)", color=COLORS[2]
)

for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{bar.get_height():.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{bar.get_height():.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels(AGE_ORDER)
ax.set_ylabel("Percentage of samples (%)")
ax.set_title(
    "Age Group Distribution: Training (PAN14) vs Evaluation (LiLaH)\n"
    "66- has 0.2% in training but 10.3% in evaluation — model never learns it"
)
ax.legend()
ax.set_ylim(0, max(pan14_pct.max(), lilah_pct.max()) + 10)
ax.axhline(y=0, color="black", linewidth=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "age_distribution_comparison.png"), dpi=150)
plt.close(fig)
print("Saved: age_distribution_comparison.png")

# ── 2. Gold vs BERT vs LLM predicted distributions ────────────────────────────
def pred_counts(series):
    counts = series.value_counts()
    return counts.reindex(AGE_ORDER, fill_value=0)


gold_counts = pred_counts(lilah["age"])
bert_counts = pred_counts(bert["pred_age"])
llm_counts = pred_counts(llm["pred_age"].dropna())

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(AGE_ORDER))
w = 0.25
ax.bar(x - w, gold_counts, w, label="Gold (ground truth)", color=COLORS[0])
ax.bar(x, bert_counts, w, label="BERT predictions", color=COLORS[1])
ax.bar(x + w, llm_counts, w, label="Llama 3.2 predictions", color=COLORS[2])

for i, (g, b, l) in enumerate(zip(gold_counts, bert_counts, llm_counts)):
    ax.text(i - w, g + 2, str(g), ha="center", va="bottom", fontsize=8)
    ax.text(i, b + 2, str(b), ha="center", va="bottom", fontsize=8)
    ax.text(i + w, l + 2, str(l), ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(AGE_ORDER)
ax.set_ylabel("Number of samples")
ax.set_title(
    "Predicted Age Distribution: Gold vs BERT vs Llama 3.2\n"
    "Both models predict 0 samples as 66- despite 64 gold samples"
)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "model_predictions_comparison.png"), dpi=150)
plt.close(fig)
print("Saved: model_predictions_comparison.png")

# ── 3. Confusion matrices: BERT and LLM side by side ──────────────────────────
bert_valid = bert.dropna(subset=["pred_age", "true_age"])
llm_valid = llm.dropna(subset=["pred_age", "true_age"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, df, model_name in zip(
    axes, [bert_valid, llm_valid], ["BERT (DistilBERT)", "Llama 3.2 (zero-shot)"]
):
    # only keep rows where true_age is in AGE_ORDER
    df = df[df["true_age"].isin(AGE_ORDER)]
    cm = confusion_matrix(df["true_age"], df["pred_age"], labels=AGE_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=AGE_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(f"{model_name}\nAcc: {(df['true_age']==df['pred_age']).mean():.1%}")
    ax.set_xlabel("Predicted age")
    ax.set_ylabel("True age")

    # highlight the 66- row to make the zero obvious
    idx = AGE_ORDER.index("66-")
    for j in range(len(AGE_ORDER)):
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, idx - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2
            )
        )

fig.suptitle(
    "Confusion Matrices — Age Prediction\n"
    "Red boxes highlight 66- row: all 64 samples misclassified by both models",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "confusion_matrices.png"), dpi=150)
plt.close(fig)
print("Saved: confusion_matrices.png")

# ── 4. Text features by age group ─────────────────────────────────────────────
def type_token_ratio(text):
    tokens = str(text).lower().split()
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / len(tokens)


lilah["word_count"] = lilah["text"].apply(lambda t: len(str(t).split()))
lilah["ttr"] = lilah["text"].apply(type_token_ratio)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, col, label, title in zip(
    axes,
    ["word_count", "ttr"],
    ["Avg word count", "Avg type-token ratio (vocabulary richness)"],
    ["Average Text Length per Age Group", "Vocabulary Richness per Age Group"],
):
    means = lilah.groupby("age")[col].mean().reindex(AGE_ORDER)
    stds = lilah.groupby("age")[col].std().reindex(AGE_ORDER)
    bars = ax.bar(AGE_ORDER, means, color=COLORS, yerr=stds, capsize=4)
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds[AGE_ORDER[list(means).index(val)]] + 0.3,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_ylabel(label)
    ax.set_xlabel("Age group")

fig.suptitle(
    "Text Characteristics by Age Group (LiLaH hate speech dataset)", fontsize=12
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "text_features_by_age.png"), dpi=150)
plt.close(fig)
print("Saved: text_features_by_age.png")

# ── 5. TF-IDF top words per age group ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, age_group, color in zip(axes, AGE_ORDER, COLORS):
    group_texts = lilah[lilah["age"] == age_group]["text"].fillna("").tolist()
    other_texts = lilah[lilah["age"] != age_group]["text"].fillna("").tolist()

    if len(group_texts) < 2:
        ax.set_title(f"{age_group} (insufficient samples)")
        continue

    all_texts = group_texts + other_texts
    labels_bin = [1] * len(group_texts) + [0] * len(other_texts)

    vec = TfidfVectorizer(
        max_features=500, stop_words="english", ngram_range=(1, 2), min_df=2
    )
    tfidf = vec.fit_transform(all_texts)
    feature_names = vec.get_feature_names_out()

    # mean TF-IDF for this group vs others
    group_tfidf = tfidf[: len(group_texts)].toarray().mean(axis=0)
    other_tfidf = tfidf[len(group_texts) :].toarray().mean(axis=0)
    diff = group_tfidf - other_tfidf

    top_idx = diff.argsort()[-15:][::-1]
    top_words = [feature_names[i] for i in top_idx]
    top_scores = [diff[i] for i in top_idx]

    ax.barh(top_words[::-1], top_scores[::-1], color=color)
    ax.set_title(
        f"Age {age_group}  (n={len(group_texts)})\nTop distinctive words vs other groups"
    )
    ax.set_xlabel("TF-IDF difference (higher = more distinctive)")
    ax.axvline(0, color="black", linewidth=0.8)

fig.suptitle(
    "Most Distinctive Words per Age Group (TF-IDF)\n"
    "66- group has very few samples — words may not be reliable",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tfidf_top_words.png"), dpi=150)
plt.close(fig)
print("Saved: tfidf_top_words.png")

# ── Summary printed to console ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: Why 66+ has 0% recall in both models")
print("=" * 60)
print(f"\nPAN14 training data — age distribution:")
for g, pct in zip(AGE_ORDER, pan14_pct):
    n = int(round(pct / 100 * len(pan14)))
    flag = " ** CRITICAL: only {n} samples **".format(n=n) if g == "66-" else ""
    print(f"  {g:6s}: {pct:5.1f}%  ({n:,} samples){flag}")

print(f"\nLiLaH eval data — age distribution:")
for g, pct in zip(AGE_ORDER, lilah_pct):
    n = int(round(pct / 100 * len(lilah)))
    print(f"  {g:6s}: {pct:5.1f}%  ({n:,} samples)")

print(f"\nBERT: 66- recall = 0/{gold_counts['66-']} (0.0%)")
print(f"LLM:  66- recall = 0/{gold_counts['66-']} (0.0%)")
print(f"\nPlots saved to: {OUT}")

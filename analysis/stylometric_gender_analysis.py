# Stylometric gender analysis on LiLaH EN hate speech.
# Tests whether general-domain gender cues (Schler et al. 2006) are suppressed in this domain.
# Output: analysis/figures/stylometric_gender.png

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

plt.switch_backend("Agg")

DATA = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "LILAH_data",
    "merged-en-meta-lilah.tsv",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── feature word lists ────────────────────────────────────────────────────────
FP_PRONOUNS = {"i", "me", "my", "mine", "myself"}
SP_PRONOUNS = {"you", "your", "yours", "yourself"}
TP_PRONOUNS = {"he", "she", "they", "him", "her", "them", "his", "their", "hers"}
HEDGE_WORDS = {
    "maybe",
    "perhaps",
    "probably",
    "possibly",
    "might",
    "could",
    "guess",
    "suppose",
    "believe",
    "think",
    "feel",
    "seems",
    "appear",
    "kind",
    "sort",
    "somewhat",
}
INTENSIFIERS = {
    "very",
    "really",
    "so",
    "absolutely",
    "totally",
    "incredibly",
    "amazing",
    "awful",
    "horrible",
    "terrible",
    "love",
    "hate",
    "wonderful",
    "disgusting",
    "fantastic",
}


def tokenise(text):
    return re.findall(r"\b[a-z']+\b", str(text).lower())


def ttr(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 0


def features(text):
    tokens = tokenise(text)
    n = max(len(tokens), 1)
    words = [t.strip("'") for t in tokens]

    fp = sum(1 for w in words if w in FP_PRONOUNS)
    sp = sum(1 for w in words if w in SP_PRONOUNS)
    tp = sum(1 for w in words if w in TP_PRONOUNS)
    hdg = sum(1 for w in words if w in HEDGE_WORDS)
    emo = sum(1 for w in words if w in INTENSIFIERS)
    qm = text.count("?")
    em = text.count("!")
    awl = np.mean([len(w) for w in words]) if words else 0

    scale = 100 / n
    return {
        "fp_rate": fp * scale,
        "sp_rate": sp * scale,
        "tp_rate": tp * scale,
        "hedge_rate": hdg * scale,
        "emo_rate": emo * scale,
        "qmark_rate": qm * scale,
        "emark_rate": em * scale,
        "avg_word_len": awl,
        "ttr": ttr(words),
        "n_words": n,
    }


# ── load & compute ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, sep="\t")
df = df[df["binary_label"] == 1].copy()
feat_df = pd.DataFrame(df["text"].apply(features).tolist())
feat_df["gender"] = df["gender"].values

F = feat_df[feat_df["gender"] == "F"]
M = feat_df[feat_df["gender"] == "M"]

FEATURES = [
    ("fp_rate", "1st-person\npronouns"),
    ("sp_rate", "2nd-person\npronouns"),
    ("tp_rate", "3rd-person\npronouns"),
    ("hedge_rate", "Hedging\nwords"),
    ("emo_rate", "Emotional\nintensifiers"),
    ("qmark_rate", "Question\nmarks"),
    ("emark_rate", "Exclamation\nmarks"),
    ("avg_word_len", "Avg word\nlength"),
    ("ttr", "Type-token\nratio"),
]

# ── significance test ─────────────────────────────────────────────────────────
print(f"\n{'Feature':<22} {'F mean':>8} {'M mean':>8}  {'p-value':>10}  sig")
print("-" * 60)
results = []
for col, label in FEATURES:
    f_vals = F[col].dropna()
    m_vals = M[col].dropna()
    stat, p = mannwhitneyu(f_vals, m_vals, alternative="two-sided")
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(
        f"{label.replace(chr(10),' '):<22} {f_vals.mean():>8.3f} {m_vals.mean():>8.3f}  {p:>10.4f}  {sig}"
    )
    results.append(
        (col, label, f_vals.mean(), f_vals.sem(), m_vals.mean(), m_vals.sem(), p, sig)
    )

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))

x = np.arange(len(FEATURES))
w = 0.35
f_means = [r[2] for r in results]
f_sems = [r[3] for r in results]
m_means = [r[4] for r in results]
m_sems = [r[5] for r in results]
labels = [r[1] for r in results]
sigs = [r[7] for r in results]

bars_f = ax.bar(
    x - w / 2,
    f_means,
    w,
    yerr=f_sems,
    label="Female (n=174)",
    color="#E91E8C",
    alpha=0.82,
    capsize=4,
)
bars_m = ax.bar(
    x + w / 2,
    m_means,
    w,
    yerr=m_sems,
    label="Male (n=445)",
    color="#1565C0",
    alpha=0.82,
    capsize=4,
)

# significance annotations
for i, (sig, fm, mm) in enumerate(zip(sigs, f_means, m_means)):
    top = max(fm, mm) + max(f_sems[i], m_sems[i]) + 0.08
    color = "#333333" if sig == "n.s." else "#CC0000"
    ax.text(
        i,
        top,
        sig,
        ha="center",
        va="bottom",
        fontsize=9,
        color=color,
        fontweight="bold" if sig != "n.s." else "normal",
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Rate per 100 words (or raw value)", fontsize=10)
ax.set_title(
    "Stylometric Features by Gender — LiLaH EN Hate Speech\n"
    "(error bars = SEM; *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant)",
    fontsize=11,
    fontweight="bold",
)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

out = os.path.join(OUT_DIR, "stylometric_gender.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out}")

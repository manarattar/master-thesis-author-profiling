"""
Emotional words analysis — checks whether words associated with female emotional
writing in general-domain text (Schler et al. 2006) are more frequent in female
hate speech than male hate speech on LiLaH EN.

Two outputs:
  1. Bar chart: per-word frequency (per 1000 words) for F vs M, sorted by F rate
  2. Summary: how many emotional words show F > M, F < M, no difference
"""

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

# Emotional/feeling words associated with female writing in general domain
# (from Schler et al. 2006, Argamon et al. 2007, Koppel et al. 2002)
EMOTIONAL_WORDS = [
    # feeling / emotion states
    "feel",
    "feeling",
    "felt",
    "feelings",
    "sad",
    "happy",
    "upset",
    "hurt",
    "scared",
    "worried",
    "angry",
    "afraid",
    "lonely",
    "proud",
    "ashamed",
    "guilty",
    "anxious",
    "nervous",
    "excited",
    "love",
    "hate",
    "fear",
    # empathy / relational
    "sorry",
    "care",
    "hope",
    "wish",
    "miss",
    "please",
    # hedging / softening
    "think",
    "guess",
    "maybe",
    "perhaps",
    "probably",
    # intensifiers (female-associated in general domain)
    "really",
    "very",
    "so",
    "absolutely",
    "terrible",
    "awful",
    "amazing",
    "wonderful",
    "horrible",
]

def word_rates(text_series):
    """Return dict: word -> list of per-1000-word rates, one per text."""
    rates = {w: [] for w in EMOTIONAL_WORDS}
    for text in text_series:
        tokens = re.findall(r"\b[a-z]+\b", str(text).lower())
        n = max(len(tokens), 1)
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for w in EMOTIONAL_WORDS:
            rates[w].append(counts.get(w, 0) * 1000 / n)
    return rates

df = pd.read_csv(DATA, sep="\t")
df = df[df["binary_label"] == 1].copy()
F_texts = df[df["gender"] == "F"]["text"]
M_texts = df[df["gender"] == "M"]["text"]

f_rates = word_rates(F_texts)
m_rates = word_rates(M_texts)

# ── significance testing ──────────────────────────────────────────────────────
rows = []
for w in EMOTIONAL_WORDS:
    fv = np.array(f_rates[w])
    mv = np.array(m_rates[w])
    if fv.sum() == 0 and mv.sum() == 0:
        continue
    stat, p = mannwhitneyu(fv, mv, alternative="two-sided")
    direction = "F>M" if fv.mean() > mv.mean() else "M>F"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    rows.append(
        {
            "word": w,
            "f_mean": fv.mean(),
            "m_mean": mv.mean(),
            "p": p,
            "sig": sig,
            "direction": direction,
        }
    )

results = pd.DataFrame(rows).sort_values("f_mean", ascending=False)

print("\nEmotional word frequencies (per 1000 words):")
print(f"{'Word':<15} {'F mean':>8} {'M mean':>8}  {'p':>8}  sig   direction")
print("-" * 60)
for _, r in results.iterrows():
    print(
        f"{r['word']:<15} {r['f_mean']:>8.3f} {r['m_mean']:>8.3f}  "
        f"{r['p']:>8.4f}  {r['sig']:<5} {r['direction']}"
    )

sig_f = results[(results["sig"] != "n.s.") & (results["direction"] == "F>M")]
sig_m = results[(results["sig"] != "n.s.") & (results["direction"] == "M>F")]
ns = results[results["sig"] == "n.s."]
print(
    f"\nSummary: {len(sig_f)} words F>M (sig), {len(sig_m)} words M>F (sig), {len(ns)} n.s."
)

# ── plot: top 20 by F rate, colour by significance ───────────────────────────
top = results.head(20).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(top))
w = 0.38

bar_colors_f = ["#C2185B" if s != "n.s." else "#F48FB1" for s in top["sig"]]
bar_colors_m = ["#1565C0" if s != "n.s." else "#90CAF9" for s in top["sig"]]

ax.bar(x - w / 2, top["f_mean"], w, color=bar_colors_f, label="Female (n=174)")
ax.bar(x + w / 2, top["m_mean"], w, color=bar_colors_m, label="Male (n=445)")

for i, row in top.iterrows():
    if row["sig"] != "n.s.":
        top_val = max(row["f_mean"], row["m_mean"]) + 0.05
        ax.text(
            i,
            top_val,
            row["sig"],
            ha="center",
            va="bottom",
            fontsize=8,
            color="#CC0000",
            fontweight="bold",
        )
    else:
        top_val = max(row["f_mean"], row["m_mean"]) + 0.05
        ax.text(
            i, top_val, "n.s.", ha="center", va="bottom", fontsize=7, color="#888888"
        )

ax.set_xticks(x)
ax.set_xticklabels(top["word"], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Frequency per 1000 words", fontsize=10)
ax.set_title(
    "Emotional Word Frequencies by Gender — LiLaH EN Hate Speech\n"
    "(dark bars = significant difference; light bars = n.s.)\n"
    "Sorted by female frequency (top 20 shown)",
    fontsize=11,
    fontweight="bold",
)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

out = os.path.join(OUT_DIR, "emotional_words_gender.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out}")

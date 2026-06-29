"""
Generate wordclouds for gender (M/F) and age groups from the EN LiLaH hate-speech subset.
Outputs saved to analysis/figures/wordclouds/
Run from project root: python analysis/wordclouds.py
"""

import os
import re
import string

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

plt.switch_backend("Agg")

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "LILAH_data", "merged-en-meta-lilah.tsv"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "wordclouds")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load & filter hate-speech only ────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, sep="\t")
df = df[df["binary_label"] == 1].copy()
print(f"Hate-speech rows: {len(df)}")

# ── stopwords ─────────────────────────────────────────────────────────────────
EXTRA_STOPS = {
    "people",
    "like",
    "just",
    "get",
    "one",
    "will",
    "know",
    "think",
    "would",
    "make",
    "way",
    "even",
    "dont",
    "can",
    "said",
    "say",
    "really",
    "go",
    "going",
    "want",
    "look",
    "much",
    "need",
    "got",
    "also",
    "come",
    "back",
    "us",
    "still",
    "well",
    "thing",
    "things",
    "good",
    "man",
    "woman",
    "men",
    "women",
    "s",
    "t",
}
STOPS = STOPWORDS | EXTRA_STOPS

def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def make_wc(
    text_series: pd.Series,
    title: str,
    filename: str,
    bg: str = "white",
    colormap: str = "viridis",
) -> None:
    corpus = " ".join(text_series.apply(clean))
    wc = WordCloud(
        width=900,
        height=500,
        background_color=bg,
        colormap=colormap,
        stopwords=STOPS,
        max_words=120,
        collocations=False,
        min_font_size=10,
    ).generate(corpus)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ── gender wordclouds ──────────────────────────────────────────────────────────
print("\n-- Gender --")
for gender, label, cmap in [("M", "Male", "Blues"), ("F", "Female", "RdPu")]:
    subset = df[df["gender"] == gender]["text"]
    print(f"  {label}: {len(subset)} texts")
    make_wc(
        subset,
        f"Wordcloud — {label} authors (EN hate speech)",
        f"wc_gender_{gender}.png",
        colormap=cmap,
    )

# ── age group wordclouds ───────────────────────────────────────────────────────
AGE_GROUPS = [
    ("0-25", "Age 0–25", "Greens"),
    ("26-35", "Age 26–35", "Oranges"),
    ("36-65", "Age 36–65", "Purples"),
    ("66-", "Age 66+", "YlOrBr"),
]
print("\n-- Age groups --")
for age_val, age_label, cmap in AGE_GROUPS:
    subset = df[df["age"] == age_val]["text"]
    print(f"  {age_label}: {len(subset)} texts")
    safe = age_val.replace("-", "_").replace("+", "plus")
    make_wc(
        subset,
        f"Wordcloud — {age_label} authors (EN hate speech)",
        f"wc_age_{safe}.png",
        colormap=cmap,
    )

# ── combined figure: 2×3 grid (M, F, 0-25, 26-35, 36-65, 66+) ────────────────
print("\nGenerating combined overview figure...")
GROUPS = [
    (df[df["gender"] == "M"]["text"], "Male", "Blues"),
    (df[df["gender"] == "F"]["text"], "Female", "RdPu"),
    (df[df["age"] == "0-25"]["text"], "Age 0–25", "Greens"),
    (df[df["age"] == "26-35"]["text"], "Age 26–35", "Oranges"),
    (df[df["age"] == "36-65"]["text"], "Age 36–65", "Purples"),
    (df[df["age"] == "66-"]["text"], "Age 66+", "YlOrBr"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for ax, (series, title, cmap) in zip(axes.flat, GROUPS):
    corpus = " ".join(series.apply(clean))
    wc = WordCloud(
        width=600,
        height=340,
        background_color="white",
        colormap=cmap,
        stopwords=STOPS,
        max_words=80,
        collocations=False,
        min_font_size=8,
    ).generate(corpus)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

fig.suptitle(
    "Wordclouds — EN Hate Speech Authors by Gender and Age Group",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()
out_combined = os.path.join(OUT_DIR, "wc_combined.png")
fig.savefig(out_combined, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_combined}")
print("\nDone. All wordclouds saved to analysis/figures/wordclouds/")

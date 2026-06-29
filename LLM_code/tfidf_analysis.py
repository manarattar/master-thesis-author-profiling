# TF-IDF lexical analysis: finds characteristic and discriminative vocabulary
# per gender and age group in the LiLaH EN hate speech corpus.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── config ───────────────────────────────────────────────────────────────────
DATA_PATH = "../data/hate_speech_only.tsv"
TOP_N = 20  # words shown in bar charts
MIN_DF = 3  # ignore words appearing in fewer than 3 documents
MAX_FEATURES = 10000
FIGDIR = "."


def load_data():
    df = pd.read_csv(DATA_PATH, sep="\t")
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["age"] = df["age"].astype(str).str.strip()
    df = df.dropna(subset=["text", "gender", "age"])
    df = df[df["gender"].isin(["M", "F"])]
    df = df[df["age"].isin(["0-25", "26-35", "36-65", "66-"])]
    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)


def fit_tfidf(texts):
    vec = TfidfVectorizer(
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 1),
    )
    matrix = vec.fit_transform(texts)
    return vec, matrix


def group_means(matrix, labels, groups):
    """Return dict of {group: mean TF-IDF vector (array)}."""
    means = {}
    for g in groups:
        idx = [i for i, label in enumerate(labels) if label == g]
        means[g] = np.asarray(matrix[idx].mean(axis=0)).flatten()
    return means


def top_words(mean_vec, feature_names, n=TOP_N):
    top_idx = np.argsort(mean_vec)[::-1][:n]
    return [(feature_names[i], mean_vec[i]) for i in top_idx]


def discriminative_words(means, group_a, group_b, feature_names, n=TOP_N):
    """Words with highest (mean_a - mean_b) — most distinctive for group_a."""
    diff = means[group_a] - means[group_b]
    top_idx = np.argsort(diff)[::-1][:n]
    return [(feature_names[i], diff[i]) for i in top_idx]


def save_bar(words_scores, title, filename, color="steelblue"):
    words, scores = zip(*words_scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(list(reversed(words)), list(reversed(scores)), color=color)
    ax.set_xlabel("Mean TF-IDF weight")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def save_dual_bar(
    words_a,
    words_b,
    label_a,
    label_b,
    title,
    filename,
    color_a="#4878CF",
    color_b="#D65F5F",
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, words_scores, label, color in [
        (ax1, words_a, label_a, color_a),
        (ax2, words_b, label_b, color_b),
    ]:
        words, scores = zip(*words_scores)
        ax.barh(list(reversed(words)), list(reversed(scores)), color=color)
        ax.set_xlabel("Mean TF-IDF weight")
        ax.set_title(f"{label}")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def save_quad_bar(group_words, title, filename):
    colors = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7"]
    groups = list(group_words.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, group, color in zip(axes.flatten(), groups, colors):
        words, scores = zip(*group_words[group])
        ax.barh(list(reversed(words)), list(reversed(scores)), color=color)
        ax.set_xlabel("Mean TF-IDF weight")
        ax.set_title(group)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ─── discriminative dual bar ─────────────────────────────────────────────────


def save_discriminative_gender(means, feature_names, filename):
    """Left: words most M-over-F; Right: most F-over-M."""
    m_over_f = discriminative_words(means, "M", "F", feature_names)
    f_over_m = discriminative_words(means, "F", "M", feature_names)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ws, label, color in [
        (ax1, m_over_f, "M > F  (male-skewed)", "#4878CF"),
        (ax2, f_over_m, "F > M  (female-skewed)", "#D65F5F"),
    ]:
        words, scores = zip(*ws)
        ax.barh(list(reversed(words)), list(reversed(scores)), color=color)
        ax.set_xlabel("TF-IDF weight difference")
        ax.set_title(label)
    fig.suptitle("Discriminative vocabulary: Gender", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def save_discriminative_age(means, feature_names, filename):
    """One subplot per age group: words most characteristic vs. all other groups."""
    age_groups = ["0-25", "26-35", "36-65", "66-"]
    colors = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, group, color in zip(axes.flatten(), age_groups, colors):
        # mean for this group minus mean across all others
        other_mean = np.mean([means[g] for g in age_groups if g != group], axis=0)
        diff = means[group] - other_mean
        top_idx = np.argsort(diff)[::-1][:TOP_N]
        words = [feature_names[i] for i in top_idx]
        scores = [diff[i] for i in top_idx]
        ax.barh(list(reversed(words)), list(reversed(scores)), color=color)
        ax.set_xlabel("TF-IDF weight vs. other groups")
        ax.set_title(f"Age {group} — distinctive words")
    fig.suptitle("Discriminative vocabulary: Age group", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print(df.groupby("gender").size().to_string(), "\n")
    print(df.groupby("age").size().to_string(), "\n")

    vec, matrix = fit_tfidf(df["text"].tolist())
    feature_names = vec.get_feature_names_out()
    print(f"Vocabulary size: {len(feature_names)}")

    # ── Gender ────────────────────────────────────────────────────────────────
    gender_means = group_means(matrix, df["gender"].tolist(), ["M", "F"])
    gender_top = {g: top_words(gender_means[g], feature_names) for g in ["M", "F"]}

    save_dual_bar(
        gender_top["M"],
        gender_top["F"],
        "Male (M)",
        "Female (F)",
        "Top TF-IDF words by gender",
        os.path.join(FIGDIR, "tfidf_gender_bars.png"),
    )
    save_discriminative_gender(
        gender_means,
        feature_names,
        os.path.join(FIGDIR, "tfidf_gender_discriminative.png"),
    )

    gender_rows = []
    for word, score in gender_top["M"]:
        gender_rows.append({"group": "M", "word": word, "mean_tfidf": round(score, 5)})
    for word, score in gender_top["F"]:
        gender_rows.append({"group": "F", "word": word, "mean_tfidf": round(score, 5)})
    pd.DataFrame(gender_rows).to_csv(
        os.path.join(FIGDIR, "tfidf_gender_top_words.csv"), index=False
    )
    print("Saved: tfidf_gender_top_words.csv")

    # ── Age ───────────────────────────────────────────────────────────────────
    age_groups = ["0-25", "26-35", "36-65", "66-"]
    age_means = group_means(matrix, df["age"].tolist(), age_groups)
    age_top = {g: top_words(age_means[g], feature_names) for g in age_groups}

    save_quad_bar(
        age_top,
        "Top TF-IDF words by age group",
        os.path.join(FIGDIR, "tfidf_age_bars.png"),
    )
    save_discriminative_age(
        age_means,
        feature_names,
        os.path.join(FIGDIR, "tfidf_age_discriminative.png"),
    )

    age_rows = []
    for g in age_groups:
        for word, score in age_top[g]:
            age_rows.append({"group": g, "word": word, "mean_tfidf": round(score, 5)})
    pd.DataFrame(age_rows).to_csv(
        os.path.join(FIGDIR, "tfidf_age_top_words.csv"), index=False
    )
    print("Saved: tfidf_age_top_words.csv")

    # -- Print summary --------------------------------------------------------
    print("\n-- Top 10 words per gender --")
    for g in ["M", "F"]:
        words = ", ".join(w for w, _ in gender_top[g][:10])
        print(f"  {g}: {words}")

    print("\n-- Top 10 words per age group --")
    for g in age_groups:
        words = ", ".join(w for w, _ in age_top[g][:10])
        print(f"  {g:6s}: {words}")

    print("\n-- Most discriminative (M vs F) --")
    print(
        "  M > F:",
        ", ".join(
            w
            for w, _ in discriminative_words(gender_means, "M", "F", feature_names, 10)
        ),
    )
    print(
        "  F > M:",
        ", ".join(
            w
            for w, _ in discriminative_words(gender_means, "F", "M", feature_names, 10)
        ),
    )


if __name__ == "__main__":
    main()

# Run the best prompt config on the full LiLaH dataset.
# Set BEST_PROMPT and BEST_MODE below before running.
# Resumes from checkpoint if interrupted.

import os
import re
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

sys.path.insert(0, os.path.dirname(__file__))
from prompts import (AGE_LABELS, GENDER_LABELS, JOINT_PROMPTS,
                     SEPARATE_AGE_PROMPTS, SEPARATE_GENDER_PROMPTS)

# CONFIG — set these after reviewing summary.csv

BEST_PROMPT = "demographic"  # "baseline", "style", or "demographic"
BEST_MODE = "joint"  # "joint" or "separate"

MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DATA_PATH = "../hate_speech_only.tsv"
OUTPUT_CSV = "llm_final_results.csv"
CHECKPOINT_CSV = "llm_final_results_checkpoint.csv"
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_CALLS = 2.5  # 30 RPM limit → 2.0s minimum; 2.5s gives headroom
MAX_RETRIES = 5  # real errors only; 429s do not count
CHECKPOINT_EVERY = 50
MAX_TEXT_LENGTH = 1500

# helpers


def query_llm(prompt: str) -> str:
    """Call Groq API. 429s retry indefinitely (don't count as errors)."""
    error_count = 0
    while True:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 20,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            wait = int(response.headers.get("retry-after", 30))
            print(f"    Rate limited — waiting {wait}s")
            time.sleep(wait)
            continue  # 429 never counts as an error attempt
        try:
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            error_count += 1
            if error_count >= MAX_RETRIES:
                raise Exception(f"Failed after {MAX_RETRIES} real errors: {e}")
            time.sleep(5 * error_count)


def parse_gender(text: str):
    m = re.search(r"Gender:\s*(M|F)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def parse_age(text: str):
    m = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", text)
    return m.group(1) if m else None


def save_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def print_metrics(gold, pred, labels, task):
    df = pd.DataFrame({"gold": gold, "pred": pred}).dropna()
    print(f"\n{'='*60}")
    print(f"{task.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"Samples with valid predictions: {len(df)} / {len(gold)}")
    print(f"Parse failure rate: {(len(gold) - len(df)) / len(gold):.3f}")
    if len(df) > 0:
        print(f"Accuracy:   {accuracy_score(df['gold'], df['pred']):.4f}")
        print(
            f"Macro F1:   {f1_score(df['gold'], df['pred'], average='macro', zero_division=0):.4f}"
        )
        print(
            f"Weighted F1:{f1_score(df['gold'], df['pred'], average='weighted', zero_division=0):.4f}"
        )
        print(
            classification_report(
                df["gold"], df["pred"], labels=labels, zero_division=0
            )
        )


# main


def main():
    print(f"Running BEST config: prompt='{BEST_PROMPT}', mode='{BEST_MODE}'")
    print(f"Model: {MODEL} | Full dataset")

    df = pd.read_csv(DATA_PATH, sep="\t")
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["age"] = df["age"].astype(str).str.strip()
    df = df.dropna(subset=["text", "gender", "age"]).reset_index(drop=True).copy()
    print(f"Dataset size: {len(df)} rows")

    # Load checkpoint if it exists
    start_idx = 0
    pred_gender = [None] * len(df)
    pred_age = [None] * len(df)
    raw_outputs = [None] * len(df)

    if os.path.exists(CHECKPOINT_CSV):
        ckpt = pd.read_csv(CHECKPOINT_CSV)
        done = len(ckpt.dropna(subset=["pred_gender", "pred_age"]))
        if done > 0:
            start_idx = done
            pred_gender[:done] = ckpt["pred_gender"].tolist()[:done]
            pred_age[:done] = ckpt["pred_age"].tolist()[:done]
            raw_outputs[:done] = ckpt["raw_output"].tolist()[:done]
            print(f"Resuming from row {start_idx} (checkpoint loaded)")

    prompt_template = JOINT_PROMPTS[f"joint_{BEST_PROMPT}"]

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        text = str(row["text"])[:MAX_TEXT_LENGTH]
        prompt = prompt_template.replace("{TEXT}", text)
        try:
            output = query_llm(prompt)
            g, a = parse_gender(output), parse_age(output)
        except Exception as e:
            print(f"Error row {i}: {e}")
            output, g, a = None, None, None
        pred_gender[i] = g if g in GENDER_LABELS else None
        pred_age[i] = a if a in AGE_LABELS else None
        raw_outputs[i] = output

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(df) - 1:
            ckpt_df = df.copy()
            ckpt_df["pred_gender"] = pred_gender
            ckpt_df["pred_age"] = pred_age
            ckpt_df["raw_output"] = raw_outputs
            ckpt_df.to_csv(CHECKPOINT_CSV, index=False)
            print(f"  {i+1}/{len(df)} rows — checkpoint saved")

        time.sleep(SLEEP_BETWEEN_CALLS)

    df["pred_gender"] = pred_gender
    df["pred_age"] = pred_age
    df["raw_output"] = raw_outputs

    print_metrics(df["gender"].tolist(), pred_gender, GENDER_LABELS, "gender")
    print_metrics(df["age"].tolist(), pred_age, AGE_LABELS, "age")

    gender_df = df.dropna(subset=["gender", "pred_gender"])
    age_df = df.dropna(subset=["age", "pred_age"])

    if len(gender_df) > 0:
        save_confusion_matrix(
            gender_df["gender"],
            gender_df["pred_gender"],
            GENDER_LABELS,
            f"Gender — {BEST_PROMPT} ({BEST_MODE})",
            f"llm_gender_cm_{BEST_PROMPT}_{BEST_MODE}.png",
        )
    if len(age_df) > 0:
        save_confusion_matrix(
            age_df["age"],
            age_df["pred_age"],
            AGE_LABELS,
            f"Age — {BEST_PROMPT} ({BEST_MODE})",
            f"llm_age_cm_{BEST_PROMPT}_{BEST_MODE}.png",
        )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinal results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

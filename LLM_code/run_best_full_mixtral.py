"""
Mixtral-8x7B run — same best config (demographic, joint) as the llama run.
Run this after run_best_full.py (llama-3.3-70b) has finished.
"""

import os
import re
import sys
import time
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from prompts import (
    JOINT_PROMPTS, SEPARATE_GENDER_PROMPTS, SEPARATE_AGE_PROMPTS,
    GENDER_LABELS, AGE_LABELS
)

# config

BEST_PROMPT = "demographic"
BEST_MODE   = "joint"

MODEL = "mixtral-8x7b-32768"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DATA_PATH = "../hate_speech_only.tsv"
OUTPUT_CSV = "llm_final_results_mixtral.csv"
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_CALLS = 2.0   # Mixtral has higher token/min limits than 70B
MAX_RETRIES = 5
MAX_TEXT_LENGTH = 1500

# helpers

def query_llm(prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 150,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            wait = int(response.headers.get("retry-after", 30))
            print(f"    Rate limited — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    raise Exception(f"Failed after {MAX_RETRIES} retries")

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
    print(f"{task.upper()} RESULTS — {MODEL}")
    print(f"{'='*60}")
    print(f"Samples with valid predictions: {len(df)} / {len(gold)}")
    print(f"Parse failure rate: {(len(gold) - len(df)) / len(gold):.3f}")
    if len(df) > 0:
        print(f"Accuracy:   {accuracy_score(df['gold'], df['pred']):.4f}")
        print(f"Macro F1:   {f1_score(df['gold'], df['pred'], average='macro', zero_division=0):.4f}")
        print(f"Weighted F1:{f1_score(df['gold'], df['pred'], average='weighted', zero_division=0):.4f}")
        print(classification_report(df["gold"], df["pred"], labels=labels, zero_division=0))

# main

def main():
    print(f"Running BEST config: prompt='{BEST_PROMPT}', mode='{BEST_MODE}'")
    print(f"Model: {MODEL} | Full dataset")

    df = pd.read_csv(DATA_PATH, sep="\t")
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["age"] = df["age"].astype(str).str.strip()
    df = df.dropna(subset=["text", "gender", "age"]).copy()
    print(f"Dataset size: {len(df)} rows")

    pred_gender, pred_age, raw_outputs = [], [], []

    prompt_template = JOINT_PROMPTS[f"joint_{BEST_PROMPT}"]

    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row["text"])[:MAX_TEXT_LENGTH]
        prompt = prompt_template.replace("{TEXT}", text)
        try:
            output = query_llm(prompt)
            g, a = parse_gender(output), parse_age(output)
        except Exception as e:
            print(f"Error row {i}: {e}")
            output, g, a = None, None, None
        pred_gender.append(g if g in GENDER_LABELS else None)
        pred_age.append(a if a in AGE_LABELS else None)
        raw_outputs.append(output)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} rows processed...")
        time.sleep(SLEEP_BETWEEN_CALLS)

    df["pred_gender"] = pred_gender
    df["pred_age"] = pred_age
    df["raw_output"] = raw_outputs

    print_metrics(df["gender"].tolist(), pred_gender, GENDER_LABELS, "gender")
    print_metrics(df["age"].tolist(), pred_age, AGE_LABELS, "age")

    gender_df = df.dropna(subset=["gender", "pred_gender"])
    age_df    = df.dropna(subset=["age", "pred_age"])

    if len(gender_df) > 0:
        save_confusion_matrix(
            gender_df["gender"], gender_df["pred_gender"], GENDER_LABELS,
            f"Gender — Mixtral ({BEST_PROMPT})",
            f"llm_gender_cm_mixtral_{BEST_PROMPT}.png"
        )
    if len(age_df) > 0:
        save_confusion_matrix(
            age_df["age"], age_df["pred_age"], AGE_LABELS,
            f"Age — Mixtral ({BEST_PROMPT})",
            f"llm_age_cm_mixtral_{BEST_PROMPT}.png"
        )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinal results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

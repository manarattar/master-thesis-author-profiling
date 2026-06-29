"""
Few-shot v3 LLM experiment on LiLaH EN.

Key changes from v2:
- Model outputs brief stylometric observations before labels (chain-of-thought)
- All 6 examples include observation traces showing HOW to reason
- Explicit 66- marker checklist with concrete linguistic examples
- Contrastive note distinguishing 36-65 vs 66- for female authors
- max_tokens raised to 80 to accommodate the Signals line
"""

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
from prompts import AGE_LABELS, GENDER_LABELS, JOINT_FEWSHOT_V3  # noqa: E402

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DATA_PATH = "../data/hate_speech_only.tsv"
OUTPUT_CSV = "llm_fewshot_v3_results.csv"
CHECKPOINT_CSV = "llm_fewshot_v3_checkpoint.csv"

# IDs used as examples in JOINT_FEWSHOT_V3 â€” hold these out from evaluation
HOLDOUT_IDS = {
    "https://www.facebook.com/228735667216_10155108633747217_10155109198317217",
    "https://www.facebook.com/228735667216_10153107791887217_10153107858827217",
    "https://www.facebook.com/228735667216_10153107791887217_10153108725082217",
    "https://www.facebook.com/228735667216_10153107791887217_10153108127337217",
    "https://www.facebook.com/228735667216_10154942395147217_10154942470132217",
    "https://www.facebook.com/228735667216_10153107791887217_10153108123482217",
}

REQUEST_TIMEOUT = 60
SLEEP_BETWEEN = 14.0
MAX_RETRIES = 5
CHECKPOINT_EVERY = 50
MAX_TEXT_LENGTH = 1500

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def query_llm(prompt: str) -> str:
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
                "temperature": 0.0,
                "max_tokens": 80,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            wait = int(response.headers.get("retry-after", 30))
            print(f"    Rate limited â€” waiting {wait}s | {response.text[:200]}")
            time.sleep(wait)
            continue
        try:
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            error_count += 1
            if error_count >= MAX_RETRIES:
                raise Exception(f"Failed after {MAX_RETRIES} errors: {e}")
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
    print(f"{task.upper()} RESULTS  (few-shot v3)")
    print(f"{'='*60}")
    print(
        f"Evaluated: {len(df)} / {len(gold)}  "
        f"(parse failures: {len(gold) - len(df)})"
    )
    if len(df) > 0:
        print(f"Accuracy : {accuracy_score(df['gold'], df['pred']):.4f}")
        print(
            f"Macro F1 : "
            f"{f1_score(df['gold'], df['pred'], average='macro', zero_division=0):.4f}"
        )
        print(
            classification_report(
                df["gold"], df["pred"], labels=labels, zero_division=0
            )
        )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# main
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    print(f"Holding out {len(HOLDOUT_IDS)} example IDs from evaluation\n")

    df = pd.read_csv(DATA_PATH, sep="\t")
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["age"] = df["age"].astype(str).str.strip()
    df = df.dropna(subset=["text", "gender", "age"]).reset_index(drop=True)
    df = df[~df["id"].isin(HOLDOUT_IDS)].reset_index(drop=True)
    print(f"Test set after holdout: {len(df)} rows")

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
            print(f"Resuming from row {start_idx}")

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        text = str(row["text"])[:MAX_TEXT_LENGTH]
        prompt = JOINT_FEWSHOT_V3.replace("{TEXT}", text)
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
            print(f"  {i+1}/{len(df)} â€” checkpoint saved")

        time.sleep(SLEEP_BETWEEN)

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
            "Gender â€” few-shot v3",
            "llm_gender_cm_fewshot_v3.png",
        )
    if len(age_df) > 0:
        save_confusion_matrix(
            age_df["age"],
            age_df["pred_age"],
            AGE_LABELS,
            "Age â€” few-shot v3",
            "llm_age_cm_fewshot_v3.png",
        )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()


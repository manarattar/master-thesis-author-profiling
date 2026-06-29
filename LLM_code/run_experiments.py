# Prompt comparison on a 300-row stratified subset.
# Tests baseline, style-focus, and demographic prompts in joint and separate modes.

import os
import re
import sys
import time

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from prompts import (AGE_LABELS, GENDER_LABELS, JOINT_PROMPTS,
                     SEPARATE_AGE_PROMPTS, SEPARATE_GENDER_PROMPTS)

# config

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"
DATA_PATH = "hate_speech_only.tsv"
OUTPUT_DIR = "llm_results"
SUBSET_SIZE = 300
RANDOM_STATE = 42
REQUEST_TIMEOUT = 120
SLEEP_BETWEEN_CALLS = 0.3
MAX_TEXT_LENGTH = 1500

# helpers


def query_llm(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def parse_gender(text: str):
    m = re.search(r"Gender:\s*(M|F)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def parse_age(text: str):
    m = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", text)
    return m.group(1) if m else None


def evaluate(gold, pred, labels, task_name):
    df = pd.DataFrame({"gold": gold, "pred": pred}).dropna()
    n = len(df)
    if n == 0:
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "parse_fail_rate": (len(gold) - n) / len(gold),
        }
    acc = accuracy_score(df["gold"], df["pred"])
    f1 = f1_score(df["gold"], df["pred"], average="macro", zero_division=0)
    print(f"\n  [{task_name}] n={n}, accuracy={acc:.3f}, macro_f1={f1:.3f}")
    print(classification_report(df["gold"], df["pred"], labels=labels, zero_division=0))
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "parse_fail_rate": round((len(gold) - n) / len(gold), 4),
    }


def run_joint(name, prompt_template, df):
    print(f"\n{'='*60}")
    print(f"Running: {name}  (joint mode)")
    print(f"{'='*60}")

    pred_gender, pred_age, raw = [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row["text"])[:MAX_TEXT_LENGTH]
        prompt = prompt_template.replace("{TEXT}", text)
        try:
            output = query_llm(prompt)
            g, a = parse_gender(output), parse_age(output)
        except Exception as e:
            print(f"  Error row {i}: {e}")
            g, a, output = None, None, None
        pred_gender.append(g if g in GENDER_LABELS else None)
        pred_age.append(a if a in AGE_LABELS else None)
        raw.append(output)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} done...")
        time.sleep(SLEEP_BETWEEN_CALLS)

    gender_metrics = evaluate(
        df["gender"].tolist(), pred_gender, GENDER_LABELS, "gender"
    )
    age_metrics = evaluate(df["age"].tolist(), pred_age, AGE_LABELS, "age")

    out_df = df.copy()
    out_df["pred_gender"] = pred_gender
    out_df["pred_age"] = pred_age
    out_df["raw_output"] = raw
    out_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)

    return gender_metrics, age_metrics


def run_separate(name, gender_prompt_template, age_prompt_template, df):
    print(f"\n{'='*60}")
    print(f"Running: {name}  (separate mode)")
    print(f"{'='*60}")

    pred_gender, pred_age, raw_g, raw_a = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row["text"])[:MAX_TEXT_LENGTH]

        # gender call
        try:
            out_g = query_llm(gender_prompt_template.replace("{TEXT}", text))
            g = parse_gender(out_g)
        except Exception as e:
            print(f"  Error gender row {i}: {e}")
            out_g, g = None, None
        time.sleep(SLEEP_BETWEEN_CALLS)

        # age call
        try:
            out_a = query_llm(age_prompt_template.replace("{TEXT}", text))
            a = parse_age(out_a)
        except Exception as e:
            print(f"  Error age row {i}: {e}")
            out_a, a = None, None
        time.sleep(SLEEP_BETWEEN_CALLS)

        pred_gender.append(g if g in GENDER_LABELS else None)
        pred_age.append(a if a in AGE_LABELS else None)
        raw_g.append(out_g)
        raw_a.append(out_a)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} done...")

    gender_metrics = evaluate(
        df["gender"].tolist(), pred_gender, GENDER_LABELS, "gender"
    )
    age_metrics = evaluate(df["age"].tolist(), pred_age, AGE_LABELS, "age")

    out_df = df.copy()
    out_df["pred_gender"] = pred_gender
    out_df["pred_age"] = pred_age
    out_df["raw_gender"] = raw_g
    out_df["raw_age"] = raw_a
    out_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)

    return gender_metrics, age_metrics


# main


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH, sep="\t")
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["age"] = df["age"].astype(str).str.strip()
    df = df.dropna(subset=["text", "gender", "age"]).copy()

    # Stratified sample by age to ensure 66- is represented
    subset = df.groupby("age", group_keys=False).apply(
        lambda x: x.sample(
            min(len(x), SUBSET_SIZE // len(df["age"].unique())),
            random_state=RANDOM_STATE,
        )
    )
    # Top up to SUBSET_SIZE if needed
    if len(subset) < SUBSET_SIZE:
        remaining = df.drop(subset.index).sample(
            SUBSET_SIZE - len(subset), random_state=RANDOM_STATE
        )
        subset = pd.concat([subset, remaining])
    subset = subset.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Subset size: {len(subset)}")
    print("Age distribution in subset:")
    print(subset["age"].value_counts())

    summary = []
    prompt_names = list(JOINT_PROMPTS.keys())  # baseline, style, demographic

    # Run joint configs
    for name, prompt in JOINT_PROMPTS.items():
        g_metrics, a_metrics = run_joint(name, prompt, subset)
        summary.append(
            {
                "config": name,
                "mode": "joint",
                "gender_accuracy": g_metrics["accuracy"],
                "gender_macro_f1": g_metrics["macro_f1"],
                "gender_parse_fail": g_metrics["parse_fail_rate"],
                "age_accuracy": a_metrics["accuracy"],
                "age_macro_f1": a_metrics["macro_f1"],
                "age_parse_fail": a_metrics["parse_fail_rate"],
            }
        )

    # Run separate configs
    for name in prompt_names:
        g_prompt = SEPARATE_GENDER_PROMPTS[f"separate_{name.split('_')[1]}"]
        a_prompt = SEPARATE_AGE_PROMPTS[f"separate_{name.split('_')[1]}"]
        sep_name = f"separate_{name.split('_')[1]}"
        g_metrics, a_metrics = run_separate(sep_name, g_prompt, a_prompt, subset)
        summary.append(
            {
                "config": sep_name,
                "mode": "separate",
                "gender_accuracy": g_metrics["accuracy"],
                "gender_macro_f1": g_metrics["macro_f1"],
                "gender_parse_fail": g_metrics["parse_fail_rate"],
                "age_accuracy": a_metrics["accuracy"],
                "age_macro_f1": a_metrics["macro_f1"],
                "age_parse_fail": a_metrics["parse_fail_rate"],
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY OF ALL 6 CONFIGURATIONS")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {summary_path}")
    print(
        "\nNext step: review summary.csv, pick the best config, set it in run_best_full.py"
    )


if __name__ == "__main__":
    main()

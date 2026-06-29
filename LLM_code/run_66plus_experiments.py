# Five zero-shot experiments targeting the 66+ age class invisibility problem.
# Each experiment checkpoints independently and can be resumed if interrupted.

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
from prompts import AGE_LABELS, GENDER_LABELS, JOINT_DEMOGRAPHIC

# api config

MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    try:
        import winreg

        _k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        GROQ_API_KEY, _ = winreg.QueryValueEx(_k, "GROQ_API_KEY")
    except Exception:
        pass

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DATA_PATH = "../hate_speech_only.tsv"
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN = 2.5
MAX_RETRIES = 5
CHECKPOINT_EVERY = 50
MAX_TEXT_LENGTH = 1500

# prompt variants

# 1. Bias-aware: explicit reminder that 66+ exists
PROMPT_BIAS_AWARE = """
You are a linguistic analyst specialized in demographic author profiling.

Predict the gender and age group of the author based on linguistic patterns.

Demographic markers to consider:

Gender:
- M: direct assertions, less hedging, more aggressive or blunt language, fewer qualifiers
- F: hedging phrases ("I think", "maybe", "perhaps"), more emotional expression, more qualifiers and politeness markers

Age:
- 0-25: casual language, internet slang, abbreviations, short sentences, informal tone, minimal punctuation
- 26-35: mix of formal and informal, complete sentences, references to work or adult life
- 36-65: more formal vocabulary, longer sentences, traditional expressions, less slang
- 66-: very formal language, old-fashioned expressions, avoids slang, longer and more complex sentences

Important: elderly people aged 66 and older are actively present in this dataset and write online, including hateful content. Do not default to younger age groups — treat 66- as a genuine and equally valid option.

Allowed labels:

Gender:
M
F

Age:
0-25
26-35
36-65
66-

Rules:
- Choose exactly one gender and one age group.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"{TEXT}\"\"\"
""".strip()

# 2. Distribution-informed: provide actual label frequencies
PROMPT_DISTRIBUTION = """
You are a linguistic analyst specialized in demographic author profiling.

Predict the gender and age group of the author based on linguistic patterns.

Demographic markers to consider:

Gender:
- M: direct assertions, less hedging, more aggressive or blunt language, fewer qualifiers
- F: hedging phrases ("I think", "maybe", "perhaps"), more emotional expression, more qualifiers and politeness markers

Age:
- 0-25: casual language, internet slang, abbreviations, short sentences, informal tone, minimal punctuation
- 26-35: mix of formal and informal, complete sentences, references to work or adult life
- 36-65: more formal vocabulary, longer sentences, traditional expressions, less slang
- 66-: very formal language, old-fashioned expressions, avoids slang, longer and more complex sentences

Dataset statistics: in this collection, approximately 5% of authors are 0-25, 37% are 26-35, 47% are 36-65, and 11% are 66 or older. Calibrate your predictions to reflect this distribution.

Allowed labels:

Gender:
M
F

Age:
0-25
26-35
36-65
66-

Rules:
- Choose exactly one gender and one age group.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"{TEXT}\"\"\"
""".strip()

# 3. Stronger 66+ description: richer, more distinctive markers
PROMPT_STRONG_66 = """
You are a linguistic analyst specialized in demographic author profiling.

Predict the gender and age group of the author based on linguistic patterns.

Demographic markers to consider:

Gender:
- M: direct assertions, less hedging, more aggressive or blunt language, fewer qualifiers
- F: hedging phrases ("I think", "maybe", "perhaps"), more emotional expression, more qualifiers and politeness markers

Age:
- 0-25: casual language, internet slang (lol, wtf, omg), abbreviations, short sentences, informal tone, minimal punctuation, emoji-like expressions
- 26-35: mix of formal and informal, complete sentences, references to work or adult life, some slang but less than 0-25
- 36-65: more formal vocabulary, longer sentences, traditional expressions, little slang, proper punctuation
- 66-: highly formal register; archaic or dated vocabulary; avoidance of all abbreviations and internet acronyms; full punctuation and capitalization throughout; longer and more elaborate sentence constructions; references to traditional values or past decades; spelling errors from unfamiliarity with typing; no emoji or internet slang whatsoever; may use phrases like "in my day", "I am of the opinion that", "one ought to"

Allowed labels:

Gender:
M
F

Age:
0-25
26-35
36-65
66-

Rules:
- Choose exactly one gender and one age group.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"{TEXT}\"\"\"
""".strip()

# 4a. Two-stage — stage 1: is author 66+?
PROMPT_TWO_STAGE_1 = """
Based solely on the writing style of the text below, is the author 66 years old or older?

Answer only with Yes or No.

Text:
\"\"\"{TEXT}\"\"\"
""".strip()

# 4b. Two-stage — stage 2: classify among remaining three age groups (+ gender)
PROMPT_TWO_STAGE_2 = """
You are a linguistic analyst specialized in demographic author profiling.

Predict the gender and age group of the author based on linguistic patterns.
The author is NOT in the 66+ age group.

Demographic markers to consider:

Gender:
- M: direct assertions, less hedging, more aggressive or blunt language, fewer qualifiers
- F: hedging phrases ("I think", "maybe", "perhaps"), more emotional expression, more qualifiers and politeness markers

Age:
- 0-25: casual language, internet slang, abbreviations, short sentences, informal tone, minimal punctuation
- 26-35: mix of formal and informal, complete sentences, references to work or adult life
- 36-65: more formal vocabulary, longer sentences, traditional expressions, less slang

Allowed labels:

Gender:
M
F

Age:
0-25
26-35
36-65

Rules:
- Choose exactly one gender and one age group.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format, nothing else.

Output format:
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65>

Text:
\"\"\"{TEXT}\"\"\"
""".strip()

# experiment registry

EXPERIMENTS = [
    {
        "name": "bias_aware",
        "mode": "joint",
        "prompt": PROMPT_BIAS_AWARE,
        "temperature": 0.1,
    },
    {
        "name": "distribution",
        "mode": "joint",
        "prompt": PROMPT_DISTRIBUTION,
        "temperature": 0.1,
    },
    {
        "name": "strong_66",
        "mode": "joint",
        "prompt": PROMPT_STRONG_66,
        "temperature": 0.1,
    },
    {"name": "two_stage", "mode": "two_stage", "prompt": None, "temperature": 0.1},
    {
        "name": "temp_07",
        "mode": "joint",
        "prompt": JOINT_DEMOGRAPHIC,
        "temperature": 0.7,
    },
]

# helpers


def query_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 20) -> str:
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
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429:
            wait = int(response.headers.get("retry-after", 30))
            print(f"    Rate limited — waiting {wait}s")
            time.sleep(wait)
            continue
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
    m = re.search(r"Age:\s*(0-25|26-35|36-65|66-)\b", text)
    return m.group(1) if m else None


def parse_yes_no(text: str):
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


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
    print(f"  Saved: {filename}")


def print_metrics(gold, pred, labels, task, exp_name):
    df = pd.DataFrame({"gold": gold, "pred": pred}).dropna()
    print(f"\n{'='*60}")
    print(f"{task.upper()} — {exp_name}")
    print(f"{'='*60}")
    print(
        f"Valid: {len(df)}/{len(gold)}  |  Parse failure: {(len(gold)-len(df))/len(gold):.3f}"
    )
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


# per-experiment runner


def run_experiment(exp: dict, df: pd.DataFrame):
    name = exp["name"]
    mode = exp["mode"]
    temp = exp["temperature"]
    prompt = exp.get("prompt")

    ckpt_path = f"66plus_exp_{name}_checkpoint.csv"
    output_path = f"66plus_exp_{name}_results.csv"

    if os.path.exists(output_path):
        print(f"\n[{name}] Already complete — skipping.")
        return pd.read_csv(output_path)

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT: {name}  (mode={mode}, temp={temp})")
    print(f"{'#'*60}")

    n = len(df)
    pred_gender = [None] * n
    pred_age = [None] * n
    raw_outputs = [None] * n
    start_idx = 0

    if os.path.exists(ckpt_path):
        ckpt = pd.read_csv(ckpt_path)
        done = len(ckpt.dropna(subset=["pred_gender", "pred_age"]))
        if done > 0:
            start_idx = done
            pred_gender[:done] = ckpt["pred_gender"].tolist()[:done]
            pred_age[:done] = ckpt["pred_age"].tolist()[:done]
            raw_outputs[:done] = ckpt["raw_output"].tolist()[:done]
            print(f"  Resuming from row {start_idx}")

    for i in range(start_idx, n):
        text = str(df.iloc[i]["text"])[:MAX_TEXT_LENGTH]

        try:
            if mode == "joint":
                raw = query_llm(prompt.replace("{TEXT}", text), temp)
                g = parse_gender(raw)
                a = parse_age(raw)

            elif mode == "two_stage":
                # Stage 1: is author 66+?
                stage1_raw = query_llm(
                    PROMPT_TWO_STAGE_1.replace("{TEXT}", text), temp, max_tokens=5
                )
                time.sleep(SLEEP_BETWEEN)
                answer = parse_yes_no(stage1_raw)

                if answer == "yes":
                    raw = f"[stage1=yes] {stage1_raw}"
                    g = None  # stage 2 needed for gender
                    a = "66-"
                    # still need gender — run stage 2 for gender only
                    stage2_raw = query_llm(
                        PROMPT_TWO_STAGE_2.replace("{TEXT}", text), temp
                    )
                    raw = f"[stage1=yes] {stage1_raw} | [stage2] {stage2_raw}"
                    g = parse_gender(stage2_raw)
                    a = "66-"
                elif answer == "no":
                    stage2_raw = query_llm(
                        PROMPT_TWO_STAGE_2.replace("{TEXT}", text), temp
                    )
                    raw = f"[stage1=no] {stage1_raw} | [stage2] {stage2_raw}"
                    g = parse_gender(stage2_raw)
                    a = parse_age(stage2_raw)
                else:
                    raw = f"[stage1=unclear] {stage1_raw}"
                    g, a = None, None

        except Exception as e:
            print(f"  Error row {i}: {e}")
            raw, g, a = None, None, None

        pred_gender[i] = g if g in GENDER_LABELS else None
        pred_age[i] = a if a in AGE_LABELS else None
        raw_outputs[i] = raw

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == n - 1:
            ckpt_df = df.copy()
            ckpt_df["pred_gender"] = pred_gender
            ckpt_df["pred_age"] = pred_age
            ckpt_df["raw_output"] = raw_outputs
            ckpt_df.to_csv(ckpt_path, index=False)
            print(f"  {i+1}/{n} — checkpoint saved")

        time.sleep(SLEEP_BETWEEN)

    out_df = df.copy()
    out_df["pred_gender"] = pred_gender
    out_df["pred_age"] = pred_age
    out_df["raw_output"] = raw_outputs
    out_df.to_csv(output_path, index=False)
    print(f"  Results saved: {output_path}")

    print_metrics(out_df["gender"].tolist(), pred_gender, GENDER_LABELS, "GENDER", name)
    print_metrics(out_df["age"].tolist(), pred_age, AGE_LABELS, "AGE", name)

    g_df = out_df.dropna(subset=["gender", "pred_gender"])
    a_df = out_df.dropna(subset=["age", "pred_age"])
    if len(g_df) > 0:
        save_confusion_matrix(
            g_df["gender"],
            g_df["pred_gender"],
            GENDER_LABELS,
            f"Gender — {name}",
            f"66plus_exp_gender_cm_{name}.png",
        )
    if len(a_df) > 0:
        save_confusion_matrix(
            a_df["age"],
            a_df["pred_age"],
            AGE_LABELS,
            f"Age — {name}",
            f"66plus_exp_age_cm_{name}.png",
        )

    return out_df


# summary table


def print_summary(results: dict):
    print(f"\n{'='*70}")
    print("SUMMARY — 66+ Class F1 across experiments")
    print(f"{'='*70}")
    print(
        f"{'Experiment':<18} {'66+ F1':>8} {'Macro F1':>10} {'Accuracy':>10} {'Valid':>8}"
    )
    print("-" * 70)

    # Include original baseline for comparison
    orig_path = "llm_final_results.csv"
    if os.path.exists(orig_path):
        df = pd.read_csv(orig_path).dropna(subset=["age", "pred_age"])
        from sklearn.metrics import accuracy_score as _acc
        from sklearn.metrics import f1_score as _f1

        f66 = _f1(
            df["age"], df["pred_age"], average=None, labels=AGE_LABELS, zero_division=0
        )
        macro = _f1(
            df["age"],
            df["pred_age"],
            average="macro",
            labels=AGE_LABELS,
            zero_division=0,
        )
        acc = _acc(df["age"], df["pred_age"])
        idx66 = AGE_LABELS.index("66-")
        print(
            f"{'original (66-)':<18} {f66[idx66]:>8.4f} {macro:>10.4f} {acc:>10.4f} {len(df):>8}"
        )

    for name, df in results.items():
        df_v = df.dropna(subset=["age", "pred_age"])
        if len(df_v) == 0:
            continue
        from sklearn.metrics import accuracy_score as _acc
        from sklearn.metrics import f1_score as _f1

        f66 = _f1(
            df_v["age"],
            df_v["pred_age"],
            average=None,
            labels=AGE_LABELS,
            zero_division=0,
        )
        macro = _f1(
            df_v["age"],
            df_v["pred_age"],
            average="macro",
            labels=AGE_LABELS,
            zero_division=0,
        )
        acc = _acc(df_v["age"], df_v["pred_age"])
        idx66 = AGE_LABELS.index("66-")
        print(
            f"{name:<18} {f66[idx66]:>8.4f} {macro:>10.4f} {acc:>10.4f} {len(df_v):>8}"
        )


# main


def main():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not found in environment or Windows registry.")

    df = pd.read_csv(DATA_PATH, sep="\t")
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["age"] = df["age"].astype(str).str.strip()
    df = df.dropna(subset=["text", "gender", "age"]).reset_index(drop=True).copy()
    print(f"Dataset: {len(df)} rows\n")

    results = {}
    for exp in EXPERIMENTS:
        results[exp["name"]] = run_experiment(exp, df)

    print_summary(results)


if __name__ == "__main__":
    main()

import pandas as pd
import requests
import re
import time
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# ======================
# config
# ======================

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

DATA_PATH = "hate_speech_only.tsv"
OUTPUT_PATH = "llama_style_focus_results.csv"

# Set to None for full dataset, or a number for testing
N_ROWS = None

# Increase or decrease if needed
REQUEST_TIMEOUT = 120
SLEEP_BETWEEN_CALLS = 0.3

GENDER_LABELS = ["F", "M"]
AGE_LABELS = ["0-25", "26-35", "36-65", "66-"]

# ======================
# prompt
# ======================

def build_prompt(text: str) -> str:
    return f"""
You are a linguistic analyst specialized in author profiling.

Determine the author's gender and age based ONLY on writing style.

Focus on:
- vocabulary
- slang
- punctuation
- tone
- sentence length
- formality

Rules:
- Use only the allowed labels below.
- Choose exactly one gender and one age group.
- If uncertain, choose the most likely option.
- Do not explain your reasoning.
- Output only in the required format.

Allowed labels:

Gender:
M
F

Age:
0-25
26-35
36-65
66-

Output format:
Gender: <M or F>
Age: <0-25 | 26-35 | 36-65 | 66->

Text:
\"\"\"
{text[:1500]}
\"\"\"
""".strip()

# ======================
# query model
# ======================

def query_llm(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        },
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()["response"].strip()

# ======================
# parse output
# ======================

def parse_output(text: str):
    gender_match = re.search(r"Gender:\s*(M|F)\b", text, re.IGNORECASE)
    age_match = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", text)

    gender = gender_match.group(1).upper() if gender_match else None
    age = age_match.group(1) if age_match else None

    return gender, age

# ======================
# confusion matrix plot
# ======================

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

    # write numbers inside cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# ======================
# load data
# ======================

df = pd.read_csv(DATA_PATH, sep="\t")

df["gender"] = df["gender"].astype(str).str.strip().str.upper()
df["age"] = df["age"].astype(str).str.strip()

if N_ROWS is not None:
    df = df.head(N_ROWS).copy()

# ======================
# run inference
# ======================

pred_gender = []
pred_age = []
raw_outputs = []

print("Running style-focused Llama 3.2 experiment...\n")

for i, row in df.iterrows():
    text = str(row["text"])
    prompt = build_prompt(text)

    try:
        output = query_llm(prompt)
        g, a = parse_output(output)

        pred_gender.append(g if g in GENDER_LABELS else None)
        pred_age.append(a if a in AGE_LABELS else None)
        raw_outputs.append(output)

        print(f"Row {i}: gold=({row['gender']}, {row['age']}) pred=({pred_gender[-1]}, {pred_age[-1]})")

    except Exception as e:
        print(f"Error on row {i}: {e}")
        pred_gender.append(None)
        pred_age.append(None)
        raw_outputs.append(None)

    time.sleep(SLEEP_BETWEEN_CALLS)

df["pred_gender"] = pred_gender
df["pred_age"] = pred_age
df["raw_output"] = raw_outputs

# ======================
# evaluation: gender
# ======================

print("\n============================")
print("EVALUATION RESULTS")
print("============================")

gender_eval = df.dropna(subset=["gender", "pred_gender"]).copy()

if len(gender_eval) > 0:
    gender_acc = accuracy_score(gender_eval["gender"], gender_eval["pred_gender"])
    gender_f1_macro = f1_score(gender_eval["gender"], gender_eval["pred_gender"], average="macro")
    gender_f1_weighted = f1_score(gender_eval["gender"], gender_eval["pred_gender"], average="weighted")

    print("\nGender results")
    print(f"Samples evaluated: {len(gender_eval)}")
    print(f"Accuracy: {gender_acc:.4f}")
    print(f"Macro F1: {gender_f1_macro:.4f}")
    print(f"Weighted F1: {gender_f1_weighted:.4f}")
    print(classification_report(gender_eval["gender"], gender_eval["pred_gender"], labels=GENDER_LABELS, zero_division=0))

    save_confusion_matrix(
        gender_eval["gender"],
        gender_eval["pred_gender"],
        GENDER_LABELS,
        "Gender Confusion Matrix - Llama 3.2 Style Focus",
        "gender_confusion_matrix_llama32_style_focus.png"
    )
else:
    print("\nNo valid gender predictions to evaluate.")

# ======================
# evaluation: age
# ======================

age_eval = df.dropna(subset=["age", "pred_age"]).copy()

if len(age_eval) > 0:
    age_acc = accuracy_score(age_eval["age"], age_eval["pred_age"])
    age_f1_macro = f1_score(age_eval["age"], age_eval["pred_age"], average="macro")
    age_f1_weighted = f1_score(age_eval["age"], age_eval["pred_age"], average="weighted")

    print("\nAge results")
    print(f"Samples evaluated: {len(age_eval)}")
    print(f"Accuracy: {age_acc:.4f}")
    print(f"Macro F1: {age_f1_macro:.4f}")
    print(f"Weighted F1: {age_f1_weighted:.4f}")
    print(classification_report(age_eval["age"], age_eval["pred_age"], labels=AGE_LABELS, zero_division=0))

    save_confusion_matrix(
        age_eval["age"],
        age_eval["pred_age"],
        AGE_LABELS,
        "Age Confusion Matrix - Llama 3.2 Style Focus",
        "age_confusion_matrix_llama32_style_focus.png"
    )
else:
    print("\nNo valid age predictions to evaluate.")

# ======================
# invalid outputs
# ======================

print("\nInvalid / unparsable outputs")
print(f"Gender invalid predictions: {df['pred_gender'].isna().sum()}")
print(f"Age invalid predictions: {df['pred_age'].isna().sum()}")

# ======================
# save results
# ======================

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to {OUTPUT_PATH}")
print("Saved confusion matrices:")
print("- gender_confusion_matrix_llama32_style_focus.png")
print("- age_confusion_matrix_llama32_style_focus.png")
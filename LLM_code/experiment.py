import pandas as pd
import requests
import time
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =========================
# settings
# =========================
MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"
DATASET_PATH = "hate_speech_only.tsv"
OUTPUT_PATH = "experiment_results.csv"

# Set to None to run on the full dataset
N_ROWS = None

# =========================
# valid labels
# =========================
VALID_GENDERS = {"M", "F"}
VALID_AGES = {"0-25", "26-35", "36-65", "66-"}

# =========================
# load data
# =========================
df = pd.read_csv(DATASET_PATH, sep="\t")

if N_ROWS is not None:
    df = df.head(N_ROWS).copy()

# Normalize gold labels
df["gender"] = df["gender"].astype(str).str.strip().str.upper()
df["age"] = df["age"].astype(str).str.strip()

# =========================
# prompt
# =========================
def build_prompt(text: str) -> str:
    return f"""
You are a closed-set classifier for author profiling.

Task:
Given a hateful text, choose exactly one gender label and exactly one age-group label for the author.

Allowed gender labels:
M
F

Allowed age labels:
0-25
26-35
36-65
66-

Rules:
- Output exactly one gender label from the allowed gender labels.
- Output exactly one age label from the allowed age labels.
- Do not explain.
- Do not add any extra text.
- If uncertain, choose the single most likely label from the allowed options.

Return exactly in this format:
Gender: <M/F>
Age: <0-25/26-35/36-65/66->

Text:
{text}
""".strip()

# =========================
# query model
# =========================
def query_model(text: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": build_prompt(text),
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"].strip()

# =========================
# parse model output
# =========================
def parse_prediction(output: str):
    gender = None
    age = None

    gender_match = re.search(r"Gender:\s*(M|F)\b", output, re.IGNORECASE)
    age_match = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", output)

    if gender_match:
        gender = gender_match.group(1).upper()

    if age_match:
        age = age_match.group(1)

    return gender, age

# =========================
# run experiment
# =========================
pred_gender = []
pred_age = []
raw_outputs = []

print("Running predictions...\n")

for i, row in df.iterrows():
    text = str(row["text"])

    try:
        output = query_model(text)
        gender, age = parse_prediction(output)

        raw_outputs.append(output)
        pred_gender.append(gender if gender in VALID_GENDERS else None)
        pred_age.append(age if age in VALID_AGES else None)

        print(f"Row {i}")
        print(f"Gold gender: {row['gender']} | Pred gender: {pred_gender[-1]}")
        print(f"Gold age:    {row['age']} | Pred age:    {pred_age[-1]}")
        print("-" * 50)

    except Exception as e:
        print(f"Error on row {i}: {e}")
        raw_outputs.append(None)
        pred_gender.append(None)
        pred_age.append(None)

    time.sleep(0.5)

df["pred_gender"] = pred_gender
df["pred_age"] = pred_age
df["raw_output"] = raw_outputs

# =========================
# evaluation
# =========================
print("\n============================")
print("EVALUATION RESULTS")
print("============================")

# Gender
gender_eval = df.dropna(subset=["gender", "pred_gender"]).copy()
if len(gender_eval) > 0:
    gender_accuracy = accuracy_score(gender_eval["gender"], gender_eval["pred_gender"])
    gender_f1_macro = f1_score(gender_eval["gender"], gender_eval["pred_gender"], average="macro")
    gender_f1_weighted = f1_score(gender_eval["gender"], gender_eval["pred_gender"], average="weighted")

    print("\nGender results")
    print(f"Samples evaluated: {len(gender_eval)}")
    print(f"Accuracy: {gender_accuracy:.4f}")
    print(f"Macro F1: {gender_f1_macro:.4f}")
    print(f"Weighted F1: {gender_f1_weighted:.4f}")
    print(classification_report(gender_eval["gender"], gender_eval["pred_gender"], zero_division=0))
else:
    print("\nGender: no valid predictions to evaluate.")

# Age
age_eval = df.dropna(subset=["age", "pred_age"]).copy()
if len(age_eval) > 0:
    age_accuracy = accuracy_score(age_eval["age"], age_eval["pred_age"])
    age_f1_macro = f1_score(age_eval["age"], age_eval["pred_age"], average="macro")
    age_f1_weighted = f1_score(age_eval["age"], age_eval["pred_age"], average="weighted")

    print("\nAge results")
    print(f"Samples evaluated: {len(age_eval)}")
    print(f"Accuracy: {age_accuracy:.4f}")
    print(f"Macro F1: {age_f1_macro:.4f}")
    print(f"Weighted F1: {age_f1_weighted:.4f}")
    print(classification_report(age_eval["age"], age_eval["pred_age"], zero_division=0))
else:
    print("\nAge: no valid predictions to evaluate.")

# Invalid outputs
invalid_gender = df["pred_gender"].isna().sum()
invalid_age = df["pred_age"].isna().sum()

print("\nInvalid / unparsable outputs")
print(f"Gender invalid predictions: {invalid_gender}")
print(f"Age invalid predictions: {invalid_age}")

# =========================
# save results
# =========================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to {OUTPUT_PATH}")
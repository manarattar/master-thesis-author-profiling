import pandas as pd
import requests
import re
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ======================
# config
# ======================

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

DATA_PATH = "hate_speech_only.tsv"
SUBSET_SIZE = 50
MAX_WORKERS = 5   # parallel requests

# ======================
# prompts
# ======================

PROMPTS = {
    "baseline": """
You are an expert in linguistic profiling.

Determine the author's gender and age group.

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
{TEXT}
\"\"\"
""",

    "style_focus": """
You are a linguistic analyst specialized in author profiling.

Determine gender and age based on writing style.

Consider:
- vocabulary
- slang
- punctuation
- tone
- sentence length

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
{TEXT}
\"\"\"
""",

    "reasoning": """
Perform author profiling.

First internally analyze linguistic cues.

Then output ONLY the labels.

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
{TEXT}
\"\"\"
"""
}

# ======================
# llm query
# ======================

def query_llm(prompt):

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
        timeout=60
    )

    return response.json()["response"]

# ======================
# parse output
# ======================

def parse_output(text):

    gender_match = re.search(r"Gender:\s*(M|F)", text)
    age_match = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", text)

    gender = gender_match.group(1) if gender_match else None
    age = age_match.group(1) if age_match else None

    return gender, age

# ======================
# process single text
# ======================

def process_text(prompt_template, text):

    prompt = prompt_template.replace("{TEXT}", text[:1500])

    try:
        output = query_llm(prompt)
        gender, age = parse_output(output)

    except Exception:
        gender, age = None, None

    return gender, age

# ======================
# load data
# ======================

df = pd.read_csv(DATA_PATH, sep="\t")

subset = df.sample(SUBSET_SIZE, random_state=42)

texts = subset["text"].tolist()
gold_gender = subset["gender"].tolist()
gold_age = subset["age"].tolist()

# ======================
# run prompt experiments
# ======================

results = {}

for prompt_name, prompt_template in PROMPTS.items():

    print(f"\nRunning prompt: {prompt_name}")

    pred_gender = []
    pred_age = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(process_text, prompt_template, text)
            for text in texts
        ]

        for future in tqdm(futures):

            g, a = future.result()

            pred_gender.append(g)
            pred_age.append(a)

    # remove invalid predictions
    valid_gender = [i for i in range(len(pred_gender)) if pred_gender[i] is not None]
    valid_age = [i for i in range(len(pred_age)) if pred_age[i] is not None]

    gender_acc = accuracy_score(
        [gold_gender[i] for i in valid_gender],
        [pred_gender[i] for i in valid_gender]
    )

    gender_f1 = f1_score(
        [gold_gender[i] for i in valid_gender],
        [pred_gender[i] for i in valid_gender],
        average="macro"
    )

    age_acc = accuracy_score(
        [gold_age[i] for i in valid_age],
        [pred_age[i] for i in valid_age]
    )

    age_f1 = f1_score(
        [gold_age[i] for i in valid_age],
        [pred_age[i] for i in valid_age],
        average="macro"
    )

    results[prompt_name] = {
        "gender_accuracy": gender_acc,
        "gender_macro_f1": gender_f1,
        "age_accuracy": age_acc,
        "age_macro_f1": age_f1
    }

# ======================
# print results
# ======================

print("\nPrompt Comparison Results\n")

for name, metrics in results.items():

    print(name)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print()

# ======================
# save results
# ======================

pd.DataFrame(results).T.to_csv("prompt_experiment_results.csv")

print("Results saved to prompt_experiment_results.csv")
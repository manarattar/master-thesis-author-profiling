"""
Slovene LLaMA experiment.
Goal: compare LLaMA-3.1-8b gender macro F1 on SL vs EN to test whether
morphological richness (Slovene has grammatical gender) makes prediction easier.

Samples 619 examples from the SL subset, stratified by gender,
to match the English test set size.
"""

import os, re, sys, time, requests
import pandas as pd
from sklearn.metrics import f1_score, classification_report

# ── Groq config ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    try:
        import winreg
        _k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        GROQ_API_KEY, _ = winreg.QueryValueEx(_k, "GROQ_API_KEY")
    except Exception:
        pass

GROQ_URL  = "https://api.groq.com/openai/v1/chat/completions"
MODEL     = "llama-3.1-8b-instant"

SAMPLE_N         = 619
SLEEP_PER_CALL   = 2.5
CHECKPOINT_EVERY = 50
MAX_TEXT_LENGTH  = 1500
RANDOM_SEED      = 42

base = os.path.dirname(__file__)
DATA_PATH      = os.path.join(base, "..", "data", "LILAH_data", "merged-sl-meta-lilah.tsv")
OUTPUT_CSV     = os.path.join(base, "llm_slovene_results.csv")
CHECKPOINT_CSV = os.path.join(base, "llm_slovene_checkpoint.csv")

# ── Same demographic prompt as English run ────────────────────────────────────
DEMOGRAPHIC_PROMPT = """You are a linguistic analyst specialized in demographic author profiling.

Predict the gender and age group of the author based on linguistic patterns.

Demographic markers to consider:

Gender:
- M: direct assertions, less hedging, more aggressive or blunt language, fewer qualifiers
- F: hedging phrases ("I think", "maybe", "perhaps"), more emotional expression, more qualifiers and politeness markers

Age:
- 0-25:  casual language, internet slang, abbreviations, short sentences, informal tone, minimal punctuation
- 26-35: mix of formal and informal, complete sentences, references to work or adult life
- 36-65: more formal vocabulary, longer sentences, traditional expressions, less slang
- 66-:   very formal language, old-fashioned expressions, avoids slang, longer and more complex sentences

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
"""

GENDER_LABELS = ["F", "M"]
AGE_LABELS    = ["0-25", "26-35", "36-65", "66-"]

def query_llm(text: str) -> str:
    prompt = DEMOGRAPHIC_PROMPT.replace("{TEXT}", text)
    error_count = 0
    while True:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 20},
            timeout=60,
        )
        if r.status_code == 429:
            wait = int(r.headers.get("retry-after", 30))
            print(f"    Rate limited — waiting {wait}s")
            time.sleep(wait)
            continue
        try:
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            error_count += 1
            if error_count >= 5:
                raise
            time.sleep(5 * error_count)

def parse_gender(t): m = re.search(r"Gender:\s*(M|F)\b", t, re.I); return m.group(1).upper() if m else None
def parse_age(t):    m = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", t);    return m.group(1) if m else None

def print_metrics(gold, pred, labels, task):
    df = pd.DataFrame({"gold": gold, "pred": pred}).dropna()
    valid = len(df)
    print(f"\n{'='*60}")
    print(f"{task.upper()} — SL (n={valid}/{len(gold)} valid)")
    print(f"{'='*60}")
    if valid == 0:
        print("No valid predictions."); return
    print(f"Parse failure rate: {(len(gold)-valid)/len(gold):.3f}")
    print(f"Macro F1:  {f1_score(df['gold'], df['pred'], average='macro',    labels=labels, zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(df['gold'], df['pred'], average='weighted', labels=labels, zero_division=0):.4f}")
    print(classification_report(df["gold"], df["pred"], labels=labels, zero_division=0))

def main():
    # Load and filter
    df = pd.read_csv(DATA_PATH, sep="\t")
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["age"]    = df["age"].astype(str).str.strip()
    df = df.dropna(subset=["text","gender","age"])
    df = df[df["text"].str.strip() != ""]
    df = df[df["gender"].isin(["M","F"])].reset_index(drop=True)

    print(f"Slovene data: {len(df)} rows after filtering")
    print("Gender dist:\n", df["gender"].value_counts())
    print("Age dist:\n",    df["age"].value_counts())

    # Stratified sample by gender to match EN n=619
    frac = SAMPLE_N / len(df)
    parts = [grp.sample(frac=frac, random_state=RANDOM_SEED) for _, grp in df.groupby("gender")]
    sample = pd.concat(parts).sample(n=min(SAMPLE_N, sum(len(p) for p in parts)),
                                     random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"\nSample: {len(sample)} rows")
    print("Sample gender dist:\n", sample["gender"].value_counts())

    # Load checkpoint
    start_idx   = 0
    pred_gender = [None] * len(sample)
    pred_age    = [None] * len(sample)
    raw_outputs = [None] * len(sample)

    if os.path.exists(CHECKPOINT_CSV):
        ckpt = pd.read_csv(CHECKPOINT_CSV)
        done = int(ckpt["pred_gender"].notna().sum())
        if done > 0:
            start_idx = done
            pred_gender[:done] = ckpt["pred_gender"].tolist()[:done]
            pred_age[:done]    = ckpt["pred_age"].tolist()[:done]
            raw_outputs[:done] = ckpt["raw_output"].tolist()[:done]
            print(f"Resuming from row {start_idx}")

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set"); return

    est_min = (len(sample) - start_idx) * SLEEP_PER_CALL / 60
    print(f"\nRunning {len(sample) - start_idx} calls (~{est_min:.0f} min estimated)...")

    for i in range(start_idx, len(sample)):
        text = str(sample.iloc[i]["text"])[:MAX_TEXT_LENGTH]
        try:
            raw  = query_llm(text)
            g, a = parse_gender(raw), parse_age(raw)
        except Exception as e:
            print(f"  Error row {i}: {e}")
            raw, g, a = None, None, None

        pred_gender[i] = g
        pred_age[i]    = a
        raw_outputs[i] = raw

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(sample) - 1:
            ckpt_df = sample.copy()
            ckpt_df["pred_gender"] = pred_gender
            ckpt_df["pred_age"]    = pred_age
            ckpt_df["raw_output"]  = raw_outputs
            ckpt_df.to_csv(CHECKPOINT_CSV, index=False)
            print(f"  {i+1}/{len(sample)} — checkpoint saved")

        time.sleep(SLEEP_PER_CALL)

    sample["pred_gender"] = pred_gender
    sample["pred_age"]    = pred_age
    sample["raw_output"]  = raw_outputs
    sample.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinal results saved to: {OUTPUT_CSV}")

    print_metrics(sample["gender"].tolist(), pred_gender, GENDER_LABELS, "gender")
    print_metrics(sample["age"].tolist(),    pred_age,    AGE_LABELS,    "age")

    # Quick EN vs SL comparison
    print("\n" + "="*60)
    print("EN vs SL GENDER COMPARISON (LLaMA-3.1-8b, zero-shot)")
    print("="*60)
    print("EN macro F1:  0.515  (F1-F=0.240, F1-M=0.791)  [n=619]")
    sl_valid = pd.DataFrame({"gold": sample["gender"], "pred": pred_gender}).dropna()
    if len(sl_valid) > 0:
        sl_macro = f1_score(sl_valid["gold"], sl_valid["pred"], average="macro", labels=GENDER_LABELS, zero_division=0)
        sl_f = f1_score(sl_valid["gold"], sl_valid["pred"], average=None, labels=GENDER_LABELS, zero_division=0)
        print(f"SL macro F1:  {sl_macro:.3f}  (F1-F={sl_f[0]:.3f}, F1-M={sl_f[1]:.3f})  [n={len(sl_valid)}]")

if __name__ == "__main__":
    main()

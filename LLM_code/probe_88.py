"""
Probe experiment: "I am 88 years old..."
Tests whether models predict 66- when the author's age is stated explicitly.

Encoders run locally. LLMs called via Groq API.
"""

import os, re, sys, json, requests, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Groq config ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    try:
        import winreg
        _k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        GROQ_API_KEY, _ = winreg.QueryValueEx(_k, "GROQ_API_KEY")
    except Exception:
        pass

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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

# ── Probe texts ───────────────────────────────────────────────────────────────
PROBES = [
    # 88-year-old variants
    ("P1",  "I am 88 years old. I went to the shop and bought apples."),
    ("P2",  "I am 88 years old. I hate all people."),
    ("P3",  "I am 88 years old. These immigrants are destroying our country."),
    ("P4",  "I am 88 years old. Back in my day, people had more respect for one another."),
    ("P5",  "I am 88 years old. These young people do not know anything about life or hardship."),
    ("P6",  "I am 88 years old. lol u ppl r so dumb omg wtf"),   # explicit age + young slang style
    # controls — other explicit ages
    ("P7",  "I am 70 years old. I hate all people."),            # 66- range, not 88
    ("P8",  "I am 35 years old. I hate all people."),            # 26-35
    ("P9",  "I am 25 years old. I hate all people."),            # 0-25
    ("P10", "I am 45 years old. I hate all people."),            # 36-65
]

# ── LLM helpers ───────────────────────────────────────────────────────────────
def call_llm(model, text, temperature=0.1, max_tokens=20):
    prompt = DEMOGRAPHIC_PROMPT.replace("{TEXT}", text)
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def parse_gender(t): m = re.search(r"Gender:\s*(M|F)", t, re.I); return m.group(1).upper() if m else "PARSE_FAIL"
def parse_age(t):    m = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", t); return m.group(1) if m else "PARSE_FAIL"

# ── Encoder helpers ────────────────────────────────────────────────────────────
def load_encoder(model_dir):
    mapping = json.load(open(os.path.join(model_dir, "label_mapping.json")))
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model, id2label

def predict_encoder(tokenizer, model, id2label, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    probs = torch.softmax(logits, dim=1)[0].tolist()
    label = id2label[pred_id]
    confidence = round(probs[pred_id], 3)
    all_scores = {id2label[i]: round(p, 3) for i, p in enumerate(probs)}
    return label, confidence, all_scores

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    base = os.path.dirname(__file__)
    model_base = os.path.join(base, "..", "models", "new_bert_finetuining")

    # Load encoders that are available locally
    print("Loading BERT age model...")
    bert_age_tok, bert_age_model, bert_age_labels = load_encoder(
        os.path.join(model_base, "bert_age_colab"))

    # HateBERT weights not saved locally — skip
    hatebert_available = False

    print("\n" + "="*70)
    print("PROBE: 'I am 88 years old...' — can models predict 66-?")
    print("="*70)

    for pid, text in PROBES:
        print(f"\n[{pid}] \"{text}\"")
        print("-" * 60)

        # BERT age
        label, conf, scores = predict_encoder(bert_age_tok, bert_age_model, bert_age_labels, text)
        print(f"  BERT age:          {label} (conf={conf}) | all: {scores}")

        # LLaMA
        if GROQ_API_KEY:
            try:
                raw = call_llm("llama-3.1-8b-instant", text)
                g, a = parse_gender(raw), parse_age(raw)
                print(f"  LLaMA-3.1 gender:  {g}   age: {a}   raw: {repr(raw[:80])}")
                time.sleep(2.5)
            except Exception as e:
                print(f"  LLaMA error: {e}")

            try:
                raw = call_llm("qwen/qwen3-32b", text, max_tokens=500)
                # strip <think>...</think> block if present
                clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                g, a = parse_gender(clean), parse_age(clean)
                print(f"  Qwen3-32B  gender: {g}   age: {a}   raw: {repr(clean[:80])}")
                time.sleep(2.5)
            except Exception as e:
                print(f"  Qwen error: {e}")
        else:
            print("  LLMs: GROQ_API_KEY not set — skipping")

    print("\n" + "="*70)
    print("NOTE: HateBERT, RoBERTa, RoBERTa-v2 weights not saved locally — not included.")
    print("="*70)

if __name__ == "__main__":
    main()

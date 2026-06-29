"""
Test whether BERT and Llama predict the 66+ age group.
Supervisor task (22.04): try artificial examples like "I am 88 years old..."
"""

import json
import re
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./age_distilbert_model"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3.2"

TEST_EXAMPLES = [
    {"text": "I am 88 years old. I hate all people.", "note": "explicit age statement"},
    {"text": "I am 75 years old and I remember the war. Young people today have no respect.", "note": "explicit age + nostalgia"},
    {"text": "Back in my day things were different. I am 70 years old and disgusted by this.", "note": "explicit age + nostalgia"},
    {"text": "As a senior citizen of 80 years, I find this deeply offensive.", "note": "senior citizen framing"},
    {"text": "I hate immigrants.", "note": "baseline (no age signal)"},
]

STYLE_PROMPT = """You are a linguistic analyst specialized in author profiling.
Analyze the writing style, vocabulary, sentence structure, and tone.
Determine the author's age group based ONLY on writing style cues.

Age groups: 0-25, 26-35, 36-65, 66-

Text: {text}

Respond ONLY in this format:
Age: <label>
Gender: <M or F>"""

def load_bert():
    with open(f"{MODEL_DIR}/label_mapping.json") as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model, id2label

def predict_bert(text, tokenizer, model, id2label):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    probs = torch.softmax(outputs.logits, dim=1)[0]
    all_probs = {id2label[i]: round(probs[i].item(), 3) for i in range(len(id2label))}
    return id2label[pred_id], all_probs

def predict_llama(text):
    prompt = STYLE_PROMPT.format(text=text)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": LLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=120,
        )
        output = response.json()["response"].strip()
        age_match = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", output)
        return age_match.group(1) if age_match else f"PARSE_FAIL: {output[:80]}"
    except Exception as e:
        return f"ERROR: {e}"

def main():
    print("Loading BERT model...")
    tokenizer, model, id2label = load_bert()
    print(f"Labels: {id2label}\n")

    print("=" * 70)
    print(f"{'Text':<45} {'BERT':>8} {'Llama':>8}")
    print("=" * 70)

    for ex in TEST_EXAMPLES:
        bert_pred, bert_probs = predict_bert(ex["text"], tokenizer, model, id2label)
        llama_pred = predict_llama(ex["text"])

        print(f"\n[{ex['note']}]")
        print(f"  Text : {ex['text'][:60]}")
        print(f"  BERT : {bert_pred}  (all probs: {bert_probs})")
        print(f"  Llama: {llama_pred}")

    print("\n" + "=" * 70)
    print("Done. Check if either model predicted '66-' for any example.")

if __name__ == "__main__":
    main()

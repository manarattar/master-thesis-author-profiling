import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_PATH = r"bert_pan14_outputs\bert_pan14_gender"
DATA_PATH = "hate_speech_only.tsv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

print("Model config label2id:", model.config.label2id)
print("Model config id2label:", model.config.id2label)

id2label = {0: "F", 1: "M"}

df = pd.read_csv(DATA_PATH, sep="\t")
df["gender"] = df["gender"].astype(str).str.strip().str.upper()

print("\nGold distribution:")
print(df["gender"].value_counts())

preds = []

for i, text in enumerate(df["text"].astype(str).tolist()):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze().tolist()
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    pred_label = id2label[pred_id]
    preds.append(pred_label)

    if i < 5:
        print(f"\nExample {i}")
        print("Logits:", logits)
        print("Pred:", pred_label)
        print("Gold:", df.iloc[i]["gender"])
        print("Text:", str(df.iloc[i]['text'])[:200])

print("\nPrediction distribution:")
print(pd.Series(preds).value_counts())

accuracy = accuracy_score(df["gender"], preds)
f1 = f1_score(df["gender"], preds, average="macro")

print("\nAccuracy:", accuracy)
print("Macro F1:", f1)
print(classification_report(df["gender"], preds, zero_division=0))
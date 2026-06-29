import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_PATH = r"bert_pan14_outputs\bert_pan14_gender"
DATA_PATH = "hate_speech_only.tsv"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Label mapping
id2label = {
    0: "F",
    1: "M"
}

# Load dataset
df = pd.read_csv(DATA_PATH, sep="\t")

# Normalize gold labels
df["gender"] = df["gender"].astype(str).str.strip().str.upper()

texts = df["text"].tolist()
gold = df["gender"].tolist()

preds = []

for text in texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    pred_label = id2label[pred_id]
    preds.append(pred_label)

accuracy = accuracy_score(gold, preds)
f1 = f1_score(gold, preds, average="macro")

print("Gender Accuracy:", accuracy)
print("Gender Macro F1:", f1)
print("\nClassification report:")
print(classification_report(gold, preds, zero_division=0))

# Save predictions
df["pred_gender"] = preds
df.to_csv("bert_gender_predictions.csv", index=False)
print("\nSaved to bert_gender_predictions.csv")
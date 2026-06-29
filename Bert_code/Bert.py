import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =========================
# settings
# =========================
PAN_PATH = r"pan14_dataset"   # folder containing XML files + truth.txt
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./bert_pan14_outputs"

# Choose task: "gender" or "age"
TASK = "age"

# Map PAN14 age groups to LiLaH age groups
AGE_MAP = {
    "18-24": "0-25",
    "25-34": "26-35",
    "35-49": "36-65",
    "50-64": "36-65",
    "65-xx": "66-",
}

# Label maps
GENDER_LABELS = {"FEMALE": 0, "MALE": 1}
AGE_LABELS = {"0-25": 0, "26-35": 1, "36-65": 2, "66-": 3}

# =========================
# load pan14
# =========================
def load_truth(truth_path: str) -> dict:
    truth = {}
    with open(truth_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            author_id, gender, age = line.split(":::")
            truth[author_id] = {
                "gender": gender.strip().upper(),
                "age_raw": age.strip(),
            }
    return truth

def read_author_xml(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    documents = []
    for doc in root.iter("document"):
        if doc.text:
            documents.append(doc.text.strip())

    return " ".join(documents)

def load_pan14_dataset(pan_path: str) -> pd.DataFrame:
    truth_path = os.path.join(pan_path, "truth.txt")
    truth = load_truth(truth_path)

    rows = []
    for filename in os.listdir(pan_path):
        if not filename.endswith(".xml"):
            continue

        author_id = filename[:-4]
        if author_id not in truth:
            continue

        xml_path = os.path.join(pan_path, filename)
        text = read_author_xml(xml_path)

        gender = truth[author_id]["gender"]
        age_raw = truth[author_id]["age_raw"]
        age_mapped = AGE_MAP.get(age_raw)

        rows.append(
            {
                "author_id": author_id,
                "text": text,
                "gender": gender,
                "age_raw": age_raw,
                "age": age_mapped,
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["text"].str.len() > 0].copy()
    return df

# =========================
# prepare data
# =========================
def prepare_dataset(df: pd.DataFrame, task: str) -> Dataset:
    if task == "gender":
        df = df.dropna(subset=["text", "gender"]).copy()
        df["label"] = df["gender"].map(GENDER_LABELS)
    elif task == "age":
        df = df.dropna(subset=["text", "age"]).copy()
        df["label"] = df["age"].map(AGE_LABELS)
    else:
        raise ValueError("TASK must be 'gender' or 'age'")

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

# =========================
# tokenization
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

# =========================
# metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

# =========================
# main
# =========================
def main():
    print("Loading PAN14...")
    df = load_pan14_dataset(PAN_PATH)

    print("Loaded rows:", len(df))
    print(df.head())

    dataset = prepare_dataset(df, TASK)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    num_labels = 2 if TASK == "gender" else 4

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining BERT for {TASK} classification...")
    trainer.train()

    print("\nEvaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    preds = trainer.predict(dataset["test"])
    pred_ids = np.argmax(preds.predictions, axis=1)
    true_ids = preds.label_ids

    print("\nDetailed report:")
    print(classification_report(true_ids, pred_ids, zero_division=0))

    save_path = os.path.join(OUTPUT_DIR, f"bert_pan14_{TASK}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nSaved model to: {save_path}")

if __name__ == "__main__":
    main()
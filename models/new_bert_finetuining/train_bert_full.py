"""
Fine-tune full BERT (bert-base-uncased) on pan14_prepared_with_reviews.csv.
Uses class weights to handle 66- imbalance.
Designed to run overnight — saves best model by macro F1.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

############################################################
# configuration
############################################################

CSV_FILE    = "../pan14_prepared_with_reviews.csv"
TASK        = "age"
MODEL_NAME  = "bert-base-uncased"
OUTPUT_DIR  = "../bert_reviews_model"

MAX_LENGTH        = 64
TRAIN_BATCH_SIZE  = 4
EVAL_BATCH_SIZE   = 4
NUM_EPOCHS        = 3
LEARNING_RATE     = 2e-5
WEIGHT_DECAY      = 0.01
RANDOM_STATE      = 42

############################################################
# weighted trainer
############################################################

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

############################################################
# dataset
############################################################

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

############################################################
# helpers
############################################################

def print_header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def elapsed(start):
    return round((time.time() - start) / 60, 2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }

############################################################
# main
############################################################

def main():
    total_start = time.time()

    print_header("ENVIRONMENT")
    print(f"Model: {MODEL_NAME}")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU — this will take several hours")

    print_header("LOAD DATA")
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["text", TASK]).copy()
    df["text"] = df["text"].astype(str)
    print(f"Rows: {len(df)}")

    labels_sorted = sorted(df[TASK].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels_sorted)}
    id2label  = {idx: label for label, idx in label2id.items()}
    df["label"] = df[TASK].map(label2id)

    print(f"Labels: {label2id}")
    print("Class distribution:")
    print(df[TASK].value_counts(dropna=False))

    # Class weights
    weights_array = compute_class_weight("balanced", classes=np.array(labels_sorted), y=df[TASK].tolist())
    class_weights = torch.tensor(weights_array, dtype=torch.float)
    print(f"Class weights: { {labels_sorted[i]: round(class_weights[i].item(), 3) for i in range(len(labels_sorted))} }")

    # Save label mapping
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w") as f:
        json.dump({"task": TASK, "label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    print_header("TRAIN / VAL SPLIT")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE, stratify=df["label"])
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    print_header("LOAD TOKENIZER")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print_header("TOKENIZE")
    train_dataset = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LENGTH)
    val_dataset   = TextDataset(val_df["text"].tolist(),   val_df["label"].tolist(),   tokenizer, MAX_LENGTH)

    print_header("LOAD MODEL")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels_sorted), id2label=id2label, label2id=label2id
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print_header("TRAINING ARGUMENTS")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print_header("TRAINING")
    trainer.train()

    print_header("SAVE MODEL")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")

    print_header("FINAL EVALUATION")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Detailed classification report on val set
    preds = trainer.predict(val_dataset)
    pred_ids = preds.predictions.argmax(axis=1)
    true_ids = preds.label_ids
    print("\nClassification report:")
    print(classification_report(true_ids, pred_ids, target_names=labels_sorted, zero_division=0))

    print_header("DONE")
    print(f"Total runtime: {elapsed(total_start)} minutes")

if __name__ == "__main__":
    main()

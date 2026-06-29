import os
import json
import time
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

class WeightedTrainer(Trainer):
    """Trainer that applies class weights to the loss to handle imbalanced classes."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

############################################################
# configuration
############################################################

CSV_FILE = "../pan14_prepared_with_reviews.csv"
TASK = "age"                    # change to "age" when needed
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = f"./{TASK}_distilbert_model"

MAX_LENGTH = 64
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 2
RANDOM_STATE = 42

############################################################
# helpers
############################################################

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def elapsed_minutes(start_time: float) -> float:
    return round((time.time() - start_time) / 60, 2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }

############################################################
# dataset
############################################################

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        print(f"Tokenization started for {len(texts)} texts...")
        start = time.time()

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length
        )
        self.labels = labels

        print(f"Tokenization finished in {round(time.time() - start, 2)} seconds.")

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

############################################################
# main
############################################################

def main():
    total_start = time.time()

    print_header("STEP 1 - ENVIRONMENT")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    print_header("STEP 2 - LOAD DATA")
    step_start = time.time()

    print(f"Reading CSV from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns found: {list(df.columns)}")

    required_columns = ["text", TASK]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["text", TASK]).copy()
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].apply(lambda x: " ".join(x.split()[:512]))

    print(f"Remaining rows after dropping missing values: {len(df)}")
    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 3 - INSPECT TEXT LENGTH")
    step_start = time.time()

    word_lengths = df["text"].apply(lambda x: len(x.split()))
    char_lengths = df["text"].apply(len)

    print(f"Average word count: {word_lengths.mean():.2f}")
    print(f"Median word count: {word_lengths.median():.2f}")
    print(f"Max word count: {word_lengths.max()}")
    print(f"Average character count: {char_lengths.mean():.2f}")
    print(f"Max character count: {char_lengths.max()}")

    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 4 - PREPARE LABELS")
    step_start = time.time()

    labels = sorted(df[TASK].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    df["label"] = df[TASK].map(label2id)

    print(f"Task: {TASK}")
    print(f"Labels: {labels}")
    print(f"label2id: {label2id}")
    print("Class distribution:")
    print(df[TASK].value_counts(dropna=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": TASK,
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()}
            },
            f,
            indent=2
        )

    print(f"Saved label mapping to: {os.path.join(OUTPUT_DIR, 'label_mapping.json')}")

    # Compute class weights to handle imbalance (especially 66- underrepresentation)
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array(labels),
        y=df[TASK].tolist()
    )
    class_weights = torch.tensor(class_weights_array, dtype=torch.float)
    print(f"Class weights: { {labels[i]: round(class_weights[i].item(), 3) for i in range(len(labels))} }")

    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 5 - TRAIN / VALIDATION SPLIT")
    step_start = time.time()

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 6 - LOAD TOKENIZER")
    step_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Tokenizer loaded: {MODEL_NAME}")
    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 7 - TOKENIZE DATA")
    step_start = time.time()

    print("Building training dataset...")
    train_dataset = TextDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    print("Building validation dataset...")
    val_dataset = TextDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    print(f"Tokenization step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 8 - LOAD MODEL")
    step_start = time.time()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    print(f"Model loaded: {MODEL_NAME}")
    print(f"Number of labels: {len(labels)}")
    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 9 - TRAINING ARGUMENTS")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Train batch size: {TRAIN_BATCH_SIZE}")
    print(f"Eval batch size: {EVAL_BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print_header("STEP 10 - START TRAINING")
    print("About to call trainer.train()")
    train_start = time.time()

    trainer.train()

    train_minutes = elapsed_minutes(train_start)
    print("trainer.train() has returned")
    print(f"Training finished in {train_minutes} minutes")

    print_header("STEP 11 - SAVE FINAL MODEL")
    save_start = time.time()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Model and tokenizer saved to: {OUTPUT_DIR}")
    print(f"Save step finished in {elapsed_minutes(save_start)} minutes")

    print_header("STEP 12 - FINAL EVALUATION")
    eval_start = time.time()

    metrics = trainer.evaluate()
    print("Final evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"Evaluation finished in {elapsed_minutes(eval_start)} minutes")

    print_header("DONE")
    print(f"Total runtime: {elapsed_minutes(total_start)} minutes")

if __name__ == "__main__":
    main()
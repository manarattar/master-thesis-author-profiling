import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DEFAULT_DATA_DIR = (
    r"pan14-training-corpora-truth\pan14-author-profiling-training-corpus-english-socialmedia-2014-04-16"
)
DEFAULT_OUTPUT_DIR = "bert_pan14_outputs"
DEFAULT_MODEL_NAME = "bert-base-uncased"

PAN14_AGE_TO_LILAH = {
    "18-24": "0-25",
    "25-34": "26-35",
    "35-49": "36-65",
    "50-64": "36-65",
    "65-xx": "66-",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT on the PAN14 English social media author profiling corpus."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Folder containing PAN14 XML files and truth.txt.",
    )
    parser.add_argument(
        "--task",
        choices=["gender", "age", "all"],
        default="all",
        help="Which classifier to train.",
    )
    parser.add_argument(
        "--age-scheme",
        choices=["pan14", "lilah"],
        default="lilah",
        help="Use original PAN14 age labels or the 4-class LiLaH mapping.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for trained models and metrics.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-authors",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    parser.add_argument(
        "--save-prepared-csv",
        action="store_true",
        help="Save the parsed author-level dataset as CSV.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer/model only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision on supported GPUs.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 on supported GPUs.",
    )
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_truth(truth_path: Path) -> dict:
    truth = {}
    with truth_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            author_id, gender, age = line.split(":::")
            truth[author_id] = {
                "gender": gender.strip().upper(),
                "age_pan14": age.strip(),
                "age_lilah": PAN14_AGE_TO_LILAH.get(age.strip()),
            }
    return truth

def read_author_xml(xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    texts = []
    for document in root.iter("document"):
        if document.text:
            cleaned = " ".join(document.text.split())
            if cleaned:
                texts.append(cleaned)
    return "\n".join(texts)

def load_author_dataframe(data_dir: Path, max_authors: int | None = None) -> pd.DataFrame:
    truth_path = data_dir / "truth.txt"
    truth = load_truth(truth_path)

    rows = []
    for xml_path in sorted(data_dir.glob("*.xml")):
        author_id = xml_path.stem
        if author_id not in truth:
            continue

        text = read_author_xml(xml_path)
        if not text:
            continue

        row = {
            "author_id": author_id,
            "text": text,
            "gender": truth[author_id]["gender"],
            "age_pan14": truth[author_id]["age_pan14"],
            "age_lilah": truth[author_id]["age_lilah"],
            "text_chars": len(text),
        }
        rows.append(row)

        if max_authors is not None and len(rows) >= max_authors:
            break

    if not rows:
        raise ValueError(f"No usable author files were found in {data_dir}")

    return pd.DataFrame(rows)

def build_task_dataframe(df: pd.DataFrame, task: str, age_scheme: str) -> tuple[pd.DataFrame, dict, dict]:
    if task == "gender":
        label_column = "gender"
        label_names = sorted(df[label_column].dropna().unique().tolist())
    elif task == "age":
        label_column = "age_pan14" if age_scheme == "pan14" else "age_lilah"
        label_names = sorted(df[label_column].dropna().unique().tolist())
    else:
        raise ValueError(f"Unsupported task: {task}")

    task_df = df.dropna(subset=["text", label_column]).copy()
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    task_df["label_name"] = task_df[label_column]
    task_df["label"] = task_df["label_name"].map(label2id).astype(int)
    return task_df, label2id, id2label

def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)
    return tokenized

def build_metrics_function(id2label: dict):
    ordered_labels = [id2label[idx] for idx in sorted(id2label)]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
        }

    compute_metrics.ordered_labels = ordered_labels
    return compute_metrics

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def split_task_dataframe(
    task_df: pd.DataFrame,
    eval_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_counts = task_df["label"].value_counts()
    min_class_size = int(label_counts.min())
    requested_eval_rows = max(1, int(round(len(task_df) * eval_size)))

    can_stratify = min_class_size >= 2 and requested_eval_rows >= len(label_counts)
    stratify = task_df["label"] if can_stratify else None
    if not can_stratify:
        print("Falling back to a non-stratified split because one or more classes are too small.")

    return train_test_split(
        task_df[["author_id", "text", "label", "label_name"]],
        test_size=eval_size,
        random_state=seed,
        stratify=stratify,
    )

def train_single_task(
    full_df: pd.DataFrame,
    task: str,
    age_scheme: str,
    model_name: str,
    output_dir: Path,
    epochs: float,
    learning_rate: float,
    weight_decay: float,
    train_batch_size: int,
    eval_batch_size: int,
    max_length: int,
    eval_size: float,
    seed: int,
    fp16: bool,
    bf16: bool,
    local_files_only: bool,
) -> None:
    task_df, label2id, id2label = build_task_dataframe(full_df, task=task, age_scheme=age_scheme)

    train_df, eval_df = split_task_dataframe(task_df, eval_size=eval_size, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df, preserve_index=False)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=max_length)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        local_files_only=local_files_only,
    )

    run_name = f"bert_pan14_{task}" if task == "gender" else f"bert_pan14_age_{age_scheme}"
    task_output_dir = output_dir / run_name
    ensure_dir(task_output_dir)

    training_args = TrainingArguments(
        output_dir=str(task_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        fp16=fp16,
        bf16=bf16,
    )

    compute_metrics = build_metrics_function(id2label)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\n=== Training task: {task} ===")
    print(f"Train size: {len(train_df)}")
    print(f"Eval size: {len(eval_df)}")
    print("Label counts:")
    print(task_df["label_name"].value_counts().sort_index())

    trainer.train()

    metrics = trainer.evaluate()
    predictions = trainer.predict(eval_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=1)
    true_ids = predictions.label_ids

    classification = classification_report(
        true_ids,
        pred_ids,
        target_names=compute_metrics.ordered_labels,
        zero_division=0,
        output_dict=True,
    )

    trainer.save_model(str(task_output_dir))
    tokenizer.save_pretrained(str(task_output_dir))

    metrics_payload = {
        "task": task,
        "age_scheme": age_scheme if task == "age" else None,
        "model_name": model_name,
        "train_size": int(len(train_df)),
        "eval_size": int(len(eval_df)),
        "labels": label2id,
        "metrics": metrics,
        "classification_report": classification,
    }
    metrics_path = task_output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    preview = eval_df[["author_id", "label_name"]].copy()
    preview["predicted_label"] = [id2label[int(idx)] for idx in pred_ids]
    preview["correct"] = preview["label_name"] == preview["predicted_label"]
    preview.to_csv(task_output_dir / "eval_predictions.csv", index=False)

    print(f"Saved model and metrics to: {task_output_dir}")

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    full_df = load_author_dataframe(data_dir, max_authors=args.max_authors)

    print(f"Loaded {len(full_df)} authors from {data_dir}")
    print("Gender distribution:")
    print(full_df["gender"].value_counts().sort_index())
    print("PAN14 age distribution:")
    print(full_df["age_pan14"].value_counts().sort_index())

    if args.save_prepared_csv:
        prepared_path = output_dir / "pan14_english_socialmedia_authors.csv"
        full_df.to_csv(prepared_path, index=False)
        print(f"Saved parsed dataset to: {prepared_path}")

    tasks = ["gender", "age"] if args.task == "all" else [args.task]
    for task in tasks:
        train_single_task(
            full_df=full_df,
            task=task,
            age_scheme=args.age_scheme,
            model_name=args.model_name,
            output_dir=output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            eval_size=args.eval_size,
            seed=args.seed,
            fp16=args.fp16,
            bf16=args.bf16,
            local_files_only=args.local_files_only,
        )

if __name__ == "__main__":
    main()

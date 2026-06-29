import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_TEST_PATH = "hate_speech_only.tsv"
DEFAULT_OUTPUT_PATH = "bert_hatespeech_predictions.csv"
DEFAULT_METRICS_PATH = "bert_hatespeech_metrics.json"
DEFAULT_GENDER_MODEL = r"bert_pan14_outputs\bert_pan14_gender"
DEFAULT_AGE_MODEL = r"bert_pan14_outputs\bert_pan14_age"
FALLBACK_TASK_LABELS = {
    "gender": {0: "F", 1: "M"},
    "age": {0: "0-25", 1: "26-35", 2: "36-65", 3: "66-"},
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained BERT author-profiling models on hate_speech_only.tsv."
    )
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH)
    parser.add_argument("--gender-model", default=DEFAULT_GENDER_MODEL)
    parser.add_argument("--age-model", default=DEFAULT_AGE_MODEL)
    parser.add_argument(
        "--task",
        choices=["gender", "age", "all"],
        default="all",
        help="Which predictions to generate.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()

def normalize_gender(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"FEMALE": "F", "MALE": "M"})
    )

def normalize_age(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()

def predict_labels(
    texts: list[str],
    model_dir: Path,
    max_length: int,
    batch_size: int,
    task: str,
) -> tuple[list[str], dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    config_id2label = model.config.id2label
    id2label = {int(idx): label for idx, label in config_id2label.items()}
    if all(label.startswith("LABEL_") for label in id2label.values()):
        id2label = FALLBACK_TASK_LABELS[task]

    predictions = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_pred_ids = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        predictions.extend(id2label[idx] for idx in batch_pred_ids)

    return predictions, id2label

def evaluate_task(
    df: pd.DataFrame,
    task: str,
    model_dir: Path,
    max_length: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    if task == "gender":
        gold_column = "gender"
        pred_column = "pred_gender"
        gold = normalize_gender(df[gold_column])
    elif task == "age":
        gold_column = "age"
        pred_column = "pred_age"
        gold = normalize_age(df[gold_column])
    else:
        raise ValueError(f"Unsupported task: {task}")

    preds, id2label = predict_labels(
        texts=df["text"].astype(str).tolist(),
        model_dir=model_dir,
        max_length=max_length,
        batch_size=batch_size,
        task=task,
    )

    # Align model outputs to the labels used in the TSV.
    if task == "gender":
        preds = normalize_gender(pd.Series(preds)).tolist()

    result_df = df.copy()
    result_df[pred_column] = preds

    ordered_labels = sorted(set(gold.tolist()) | set(preds))
    metrics = {
        "model_dir": str(model_dir),
        "accuracy": accuracy_score(gold, preds),
        "macro_f1": f1_score(gold, preds, average="macro"),
        "classification_report": classification_report(
            gold,
            preds,
            labels=ordered_labels,
            zero_division=0,
            output_dict=True,
        ),
        "id2label": {str(key): value for key, value in id2label.items()},
    }
    return result_df, metrics

def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.test_path, sep="\t")

    required_columns = {"text", "gender", "age"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.test_path}: {sorted(missing)}")

    output_df = df.copy()
    metrics = {}

    if args.task in {"gender", "all"}:
        gender_model = Path(args.gender_model)
        output_df, gender_metrics = evaluate_task(
            output_df,
            task="gender",
            model_dir=gender_model,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        metrics["gender"] = gender_metrics
        print(
            f"Gender Accuracy: {gender_metrics['accuracy']:.4f} | "
            f"Gender Macro F1: {gender_metrics['macro_f1']:.4f}"
        )

    if args.task in {"age", "all"}:
        age_model = Path(args.age_model)
        output_df, age_metrics = evaluate_task(
            output_df,
            task="age",
            model_dir=age_model,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        metrics["age"] = age_metrics
        print(
            f"Age Accuracy: {age_metrics['accuracy']:.4f} | "
            f"Age Macro F1: {age_metrics['macro_f1']:.4f}"
        )

    output_df.to_csv(args.output_path, index=False)
    with open(args.metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved predictions to: {args.output_path}")
    print(f"Saved metrics to: {args.metrics_path}")

if __name__ == "__main__":
    main()

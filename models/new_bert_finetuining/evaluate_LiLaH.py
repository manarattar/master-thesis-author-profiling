import os
import json
import time
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

############################################################
# configuration
############################################################

# Use the Lilah TSV directly
LILAH_FILE = "../hate_speech_only.tsv"

TASK = "age"   # change to "gender" to run gender
MODEL_DIR = "./bert_age_colab" if TASK == "age" else "./bert_gender_colab"
MAX_LENGTH = 64
PREDICTIONS_OUTPUT = f"lilah_{TASK}_predictions_bert.csv"

############################################################
# helpers
############################################################

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def elapsed_minutes(start_time: float) -> float:
    return round((time.time() - start_time) / 60, 2)

def normalize_lilah_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure Lilah labels match the label space used during training.

    Expected Lilah format observed:
    - gender already appears as: M / F
    - age already appears as: 0-25 / 26-35 / 36-65 / 66-

    This function also handles lowercase/full-word variants just in case.
    """

def normalize_lilah_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    gender_map = {
        "male": "M",
        "female": "F",
        "m": "M",
        "f": "F",
        "M": "M",
        "F": "F"
    }

    age_map = {
        "0-25": "0-25",
        "26-35": "26-35",
        "36-65": "36-65",
        "66-": "66-",
        "18-24": "0-25",
        "25-34": "26-35",
        "35-49": "36-65",
        "50-64": "36-65",
        "65-xx": "66-",
        "65+": "66-"
    }

    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip()
        df["gender"] = df["gender"].map(lambda x: gender_map.get(x, x))

    if "age" in df.columns:
        df["age"] = df["age"].astype(str).str.strip()
        df["age"] = df["age"].map(lambda x: age_map.get(x, x))

    return df

############################################################
# main
############################################################

def main():
    total_start = time.time()

    print_header("STEP 1 - ENVIRONMENT")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    print_header("STEP 2 - LOAD LABEL MAPPING")
    step_start = time.time()

    mapping_path = os.path.join(MODEL_DIR, "label_mapping.json")
    print(f"Reading label mapping from: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    task_from_mapping = mapping["task"]
    label2id = mapping["label2id"]
    id2label = {int(k): v for k, v in mapping["id2label"].items()}

    print(f"Task stored in model: {task_from_mapping}")
    print(f"label2id: {label2id}")
    print(f"id2label: {id2label}")

    if task_from_mapping != TASK:
        print(f"WARNING: script TASK='{TASK}' but model mapping says '{task_from_mapping}'")

    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 3 - LOAD MODEL AND TOKENIZER")
    step_start = time.time()

    print(f"Loading tokenizer from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print(f"Loading model from: {MODEL_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    print("Model and tokenizer loaded successfully")
    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 4 - LOAD LILAH TSV")
    step_start = time.time()

    print(f"Reading Lilah file from: {LILAH_FILE}")
    df = pd.read_csv(LILAH_FILE, sep="\t", encoding="utf-8")

    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns found: {list(df.columns)}")

    required_columns = ["text", TASK]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = normalize_lilah_labels(df)

    df = df.dropna(subset=["text", TASK]).copy()
    df["text"] = df["text"].astype(str)

    print(f"Remaining rows after dropping missing values: {len(df)}")
    print(f"Label distribution in Lilah for task '{TASK}':")
    print(df[TASK].value_counts(dropna=False))

    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 5 - KEEP ONLY LABELS SEEN DURING TRAINING")
    step_start = time.time()

    valid_labels = set(label2id.keys())
    before_count = len(df)

    df = df[df[TASK].isin(valid_labels)].copy()

    after_count = len(df)
    removed_count = before_count - after_count

    print(f"Valid labels from trained model: {sorted(valid_labels)}")
    print(f"Rows before filtering: {before_count}")
    print(f"Rows after filtering: {after_count}")
    print(f"Rows removed due to unseen labels: {removed_count}")

    if after_count == 0:
        raise ValueError("No evaluation rows remain after filtering to valid labels.")

    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 6 - INSPECT TEXT LENGTH")
    step_start = time.time()

    word_lengths = df["text"].apply(lambda x: len(x.split()))
    char_lengths = df["text"].apply(len)

    print(f"Average word count: {word_lengths.mean():.2f}")
    print(f"Median word count: {word_lengths.median():.2f}")
    print(f"Max word count: {word_lengths.max()}")
    print(f"Average character count: {char_lengths.mean():.2f}")
    print(f"Max character count: {char_lengths.max()}")

    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 7 - TOKENIZE TEXTS")
    step_start = time.time()

    texts = df["text"].tolist()
    true_labels = df[TASK].tolist()

    print(f"Tokenizing {len(texts)} texts with max_length={MAX_LENGTH}")

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    encodings = {k: v.to(device) for k, v in encodings.items()}

    print(f"Tokenization finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 8 - RUN PREDICTIONS")
    step_start = time.time()

    with torch.no_grad():
        outputs = model(**encodings)
        pred_ids = torch.argmax(outputs.logits, dim=1).cpu().tolist()

    pred_labels = [id2label[i] for i in pred_ids]

    print("Predictions completed")
    print(f"Step finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 9 - EVALUATION")
    step_start = time.time()

    print("Classification report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))

    print("Confusion matrix:")
    labels_in_order = sorted(list(set(true_labels) | set(pred_labels)))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_in_order)

    print("Labels order:", labels_in_order)
    print(cm)

    print(f"Evaluation finished in {elapsed_minutes(step_start)} minutes")

    print_header("STEP 10 - SAVE PREDICTIONS")
    step_start = time.time()

    output_df = df.copy()
    output_df["predicted_label"] = pred_labels

    output_df.to_csv(PREDICTIONS_OUTPUT, index=False)

    print(f"Saved predictions to: {PREDICTIONS_OUTPUT}")
    print(f"Save step finished in {elapsed_minutes(step_start)} minutes")

    print_header("DONE")
    print(f"Total runtime: {elapsed_minutes(total_start)} minutes")

if __name__ == "__main__":
    main()
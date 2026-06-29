import os
import pandas as pd
import xml.etree.ElementTree as ET

############################################
# label mapping between datasets
############################################

gender_map = {
    "MALE": "M",
    "FEMALE": "F"
}

# Updated age mapping:
# PAN 65-xx is mapped to Lilah's upper class 66-
age_map = {
    "18-24": "0-25",
    "25-34": "26-35",
    "35-49": "36-65",
    "50-64": "36-65",
    "65-xx": "66-"
}

############################################
# load pan truth file
############################################

def load_truth(truth_file):
    truth = {}

    with open(truth_file, "r", encoding="utf-8") as f:
        for line in f:
            author, gender, age = line.strip().split(":::")

            gender = gender_map.get(gender)
            age = age_map.get(age)

            truth[author] = (gender, age)

    return truth

############################################
# parse pan xml files
############################################

def parse_pan_dataset(pan_dir, truth):
    rows = []

    for file in os.listdir(pan_dir):
        if not file.endswith(".xml"):
            continue

        author_id = file.replace(".xml", "")

        if author_id not in truth:
            continue

        gender, age = truth[author_id]

        tree = ET.parse(os.path.join(pan_dir, file))
        root = tree.getroot()

        texts = []
        for doc in root.findall(".//document"):
            texts.append(doc.text if doc.text else "")

        full_text = " ".join(texts)

        rows.append({
            "author_id": author_id,
            "text": full_text,
            "gender": gender,
            "age": age
        })

    return pd.DataFrame(rows)

############################################
# load lilah dataset directly from tsv
############################################

def load_lilah(tsv_file):
    df = pd.read_csv(tsv_file, sep="\t", encoding="utf-8")

    # Keep normalization simple and aligned with the model label space
    df["gender"] = df["gender"].map({
        "male": "M",
        "female": "F",
        "M": "M",
        "F": "F"
    }).fillna(df["gender"])

    df["age"] = df["age"].map({
        "0-25": "0-25",
        "26-35": "26-35",
        "36-65": "36-65",
        "66-": "66-",
        "18-24": "0-25",
        "25-34": "26-35",
        "35-49": "36-65",
        "50-64": "36-65",
        "65-xx": "66-"
    }).fillna(df["age"])

    return df[["id", "text", "gender", "age"]]

############################################
# main
############################################

def main():
    pan_dir = "pan14-training-corpora-truth/pan14-author-profiling-training-corpus-english-socialmedia-2014-04-16"
    truth_file = "truth.txt"
    lilah_file = "hate_speech_only.tsv"

    truth = load_truth("pan14-training-corpora-truth/pan14-author-profiling-training-corpus-english-socialmedia-2014-04-16/truth.txt")

    pan_df = parse_pan_dataset(pan_dir, truth)
    lilah_df = load_lilah(lilah_file)

    pan_df.to_csv("pan14_prepared.csv", index=False)
    lilah_df.to_csv("lilah_prepared.csv", index=False)

    print("Datasets prepared successfully")
    print("PAN age labels:", sorted(pan_df["age"].dropna().unique()))
    print("Lilah age labels:", sorted(lilah_df["age"].dropna().unique()))

if __name__ == "__main__":
    main()
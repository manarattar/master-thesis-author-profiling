"""
Extract English-Reviews corpus from zip, merge with existing pan14_prepared.csv.
Adds 800 authors aged 65+ (vs 14 in current social media corpus).
"""

import io
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

ZIP_PATH = r"C:\Users\manar\Desktop\pan14-author-profiling-training-corpus-2014-04-16\pan14-author-profiling-training-corpus-2014-04-16\pan14-author-profiling-training-corpus-english-reviews-2014-04-16.zip"
EXISTING_CSV = "pan14_prepared.csv"
OUTPUT_CSV = "pan14_prepared_with_reviews.csv"

age_map = {
    "18-24": "0-25",
    "25-34": "26-35",
    "35-49": "36-65",
    "50-64": "36-65",
    "65-xx": "66-",
    "65-XX": "66-",
}

gender_map = {
    "MALE": "M",
    "FEMALE": "F",
}

def load_truth_from_zip(zf, truth_path):
    truth = {}
    with zf.open(truth_path) as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(":::")
            if len(parts) != 3:
                continue
            author, gender_raw, age_raw = parts
            gender = gender_map.get(gender_raw)
            age = age_map.get(age_raw)
            if gender and age:
                truth[author] = (gender, age)
    return truth

def parse_xml_from_zip(zf, xml_path, author_id, gender, age):
    with zf.open(xml_path) as f:
        try:
            tree = ET.parse(f)
            root = tree.getroot()
            texts = [doc.text.strip() for doc in root.findall(".//document") if doc.text]
            full_text = " ".join(texts)
            if full_text:
                return {"author_id": author_id, "text": full_text, "gender": gender, "age": age}
        except ET.ParseError:
            pass
    return None

def main():
    print("=" * 60)
    print("Step 1 — Loading existing pan14_prepared.csv")
    print("=" * 60)
    existing_df = pd.read_csv(EXISTING_CSV)
    print(f"Existing rows: {len(existing_df)}")
    print("Age distribution (existing):")
    print(existing_df["age"].value_counts(dropna=False))

    print("\n" + "=" * 60)
    print("Step 2 — Reading reviews zip")
    print("=" * 60)

    rows = []
    skipped = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        all_names = zf.namelist()

        # Find truth.txt inside the zip
        truth_path = next((n for n in all_names if n.endswith("truth.txt")), None)
        if not truth_path:
            raise FileNotFoundError("truth.txt not found inside zip")
        print(f"Found truth.txt at: {truth_path}")

        truth = load_truth_from_zip(zf, truth_path)
        print(f"Authors with valid age+gender mapping: {len(truth)}")

        # Find all XML files
        xml_files = [n for n in all_names if n.endswith(".xml")]
        print(f"XML files found: {len(xml_files)}")

        for xml_path in xml_files:
            filename = xml_path.split("/")[-1]
            author_id = filename.replace(".xml", "")

            if author_id not in truth:
                skipped += 1
                continue

            gender, age = truth[author_id]
            row = parse_xml_from_zip(zf, xml_path, author_id, gender, age)
            if row:
                rows.append(row)

    reviews_df = pd.DataFrame(rows)
    print(f"\nParsed {len(reviews_df)} authors from reviews corpus ({skipped} skipped)")
    print("Age distribution (reviews):")
    print(reviews_df["age"].value_counts(dropna=False))

    print("\n" + "=" * 60)
    print("Step 3 — Merging datasets")
    print("=" * 60)

    combined_df = pd.concat([existing_df, reviews_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["author_id"])

    print(f"Combined rows: {len(combined_df)}")
    print("\nAge distribution (combined):")
    print(combined_df["age"].value_counts(dropna=False))
    print("\nGender distribution (combined):")
    print(combined_df["gender"].value_counts(dropna=False))

    print("\n" + "=" * 60)
    print("Step 4 — Saving")
    print("=" * 60)

    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to: {OUTPUT_CSV}")

    # Summary
    old_66 = existing_df[existing_df["age"] == "66-"].shape[0]
    new_66 = combined_df[combined_df["age"] == "66-"].shape[0]
    old_pct = old_66 / len(existing_df) * 100
    new_pct = new_66 / len(combined_df) * 100
    print(f"\n66- authors: {old_66} ({old_pct:.1f}%) -> {new_66} ({new_pct:.1f}%)")
    print("Done.")

if __name__ == "__main__":
    main()

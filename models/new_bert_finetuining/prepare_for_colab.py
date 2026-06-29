"""
Creates pan14_colab.csv — a clean, small file for Colab training.
Keeps only text/age/gender, truncates texts to 2000 chars, drops bad rows.
Upload pan14_colab.csv to My Drive/thesis/ in Google Drive.
"""

import sys
import csv
import pandas as pd

CSV_IN  = "../pan14_prepared_with_reviews.csv"
CSV_OUT = "../pan14_colab.csv"
TEXT_LIMIT = 2000  # chars — BERT uses ~128 tokens ≈ 500 chars; 2000 is generous

csv.field_size_limit(min(sys.maxsize, 2147483647))

print("Reading source CSV...")
df = pd.read_csv(CSV_IN, engine="python", on_bad_lines="skip")
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

# Keep only what Colab needs
keep = [c for c in ["text", "age", "gender"] if c in df.columns]
df = df[keep].dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str).str[:TEXT_LIMIT]

print(f"  After filter: {len(df):,} rows")
print(f"  Avg text length: {df['text'].str.len().mean():.0f} chars")

if "age" in df.columns:
    print(f"  Age distribution:\n{df['age'].value_counts()}")
if "gender" in df.columns:
    print(f"  Gender distribution:\n{df['gender'].value_counts()}")

df.to_csv(CSV_OUT, index=False, quoting=csv.QUOTE_ALL)

size_mb = __import__("os").path.getsize(CSV_OUT) / 1e6
print(f"\nSaved to: {CSV_OUT}  ({size_mb:.1f} MB)")
print("Upload this file to My Drive/thesis/pan14_colab.csv in Google Drive.")

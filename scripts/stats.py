import pandas as pd

# Load the filtered TSV file
df = pd.read_csv("hate_speech_only.tsv", sep="\t")

# Basic dataset information
print("Dataset shape (rows, columns):", df.shape)
print()

print("Column names:")
print(df.columns.tolist())
print()

# Check label distribution if the column exists
if "binary_label" in df.columns:
    print("Binary label distribution:")
    print(df["binary_label"].value_counts())
    print()

if "label" in df.columns:
    print("Detailed label distribution:")
    print(df["label"].value_counts())
    print()

# Average text length
if "text" in df.columns:
    df["text_length"] = df["text"].astype(str).apply(len)
    print("Average text length:", df["text_length"].mean())
    print("Maximum text length:", df["text_length"].max())
    print("Minimum text length:", df["text_length"].min())
    print()

# Check missing values
print("Missing values per column:")
print(df.isnull().sum())
print()

# Show a few example rows
print("Sample rows:")
print(df.head())
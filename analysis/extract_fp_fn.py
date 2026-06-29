"""
Extract 10 FP + 10 FN examples for gender and age (BERT predictions).
Outputs a readable text report for the presentation.

Definitions:
  Gender (binary F/M):
    FP = predicted F, actually M  (model over-predicts female)
    FN = predicted M, actually F  (model misses female authors)

  Age (focus on 66- class — the most thesis-relevant):
    FP = predicted 66-, actually NOT 66-  (false alarm)
    FN = predicted NOT 66-, actually 66-  (missed older author)
"""

import os, sys, textwrap
import pandas as pd

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

base  = os.path.join(os.path.dirname(__file__), "..")
N     = 10
WRAP  = 120

g_df = pd.read_csv(os.path.join(base, "results", "BERT-gender_predictions.csv"))
a_df = pd.read_csv(os.path.join(base, "results", "BERT-age_predictions.csv"))

def show(df, gold_col, pred_col, condition, label, n=N):
    subset = df[condition].copy()
    subset["wc"] = subset["text"].str.split().str.len()
    # prefer medium-length texts (more interpretable than 2-word comments)
    subset = subset.sort_values("wc").iloc[len(subset)//4 : 3*len(subset)//4]
    sample = subset.sample(min(n, len(subset)), random_state=42)
    print(f"\n{'─'*70}")
    print(f"  {label}  (n available: {len(df[condition])}  |  showing {len(sample)})")
    print(f"{'─'*70}")
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        text = str(row["text"])
        wrapped = textwrap.fill(text[:300], width=WRAP, subsequent_indent="      ")
        print(f"\n  [{i:02d}] gold={row[gold_col]}  pred={row[pred_col]}  wc={row['wc']}")
        print(f"      {wrapped}")

# ── GENDER ────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  GENDER ERROR ANALYSIS — BERT")
print("=" * 70)

show(g_df, "gender", "predicted",
     (g_df["gender"] == "M") & (g_df["predicted"] == "F"),
     "FP — Predicted FEMALE, Actually MALE")

show(g_df, "gender", "predicted",
     (g_df["gender"] == "F") & (g_df["predicted"] == "M"),
     "FN — Predicted MALE, Actually FEMALE")

# ── AGE ───────────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  AGE ERROR ANALYSIS — BERT  (focus: 66- class)")
print("=" * 70)

show(a_df, "age", "predicted",
     (a_df["age"] != "66-") & (a_df["predicted"] == "66-"),
     "FP — Predicted 66-, Actually NOT 66-")

show(a_df, "age", "predicted",
     (a_df["age"] == "66-") & (a_df["predicted"] != "66-"),
     "FN — Predicted NOT 66-, Actually 66-",
     n=N)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  COUNTS SUMMARY")
print("=" * 70)
gfp = ((g_df["gender"]=="M") & (g_df["predicted"]=="F")).sum()
gfn = ((g_df["gender"]=="F") & (g_df["predicted"]=="M")).sum()
afp = ((a_df["age"]!="66-") & (a_df["predicted"]=="66-")).sum()
afn = ((a_df["age"]=="66-") & (a_df["predicted"]!="66-")).sum()
print(f"  Gender FP (pred F, true M): {gfp}")
print(f"  Gender FN (pred M, true F): {gfn}")
print(f"  Age FP (pred 66-, true ≠66-): {afp}")
print(f"  Age FN (pred ≠66-, true 66-): {afn}")

"""
Slovene LLaMA experiment — v2 with Slovene-morphology-aware prompt.
Compares against v1 (generic English prompt) to test whether explicit
Slovene morphological cues improve gender detection.

LLaMA-3.1-8b-instant via Groq API.
"""

import os, re, sys, time, requests
import pandas as pd
from sklearn.metrics import f1_score, classification_report

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    try:
        import winreg
        _k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        GROQ_API_KEY, _ = winreg.QueryValueEx(_k, "GROQ_API_KEY")
    except Exception:
        pass

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL    = "llama-3.1-8b-instant"

SAMPLE_N         = 619
SLEEP_PER_CALL   = 2.5
CHECKPOINT_EVERY = 50
MAX_TEXT_LENGTH  = 1500
RANDOM_SEED      = 42   # same seed → same sample as v1 for direct comparison

base = os.path.dirname(__file__)
DATA_PATH      = os.path.join(base, "..", "data", "LILAH_data", "merged-sl-meta-lilah.tsv")
OUTPUT_CSV     = os.path.join(base, "llm_slovene_results_llama_v2.csv")
CHECKPOINT_CSV = os.path.join(base, "llm_slovene_checkpoint_llama_v2.csv")

# ── Slovene prompt with Slovene morphological instructions ───────────────────
SLOVENE_PROMPT = """Ste jezikovni analitik, specializiran za demografsko profiliranje avtorjev v slovenščini.

Spodnje besedilo je napisano v slovenščini. Uporabite slovnične in slogovne značilnosti slovenščine za napovedovanje spola in starostne skupine avtorja.

SPOL — prednostni znaki v slovenščini:

MOŠKI (M):
- Glagolski pridevnik na -l (moška oblika): "sem šel", "sem rekel", "sem bil", "sem naredil", "sem vedel", "sem mislil"
- Pridevnik v 1. osebi (moška oblika): "sem vesel", "sem zadovoljen", "sem prepričan", "sem utrujen", "sem jezen"
- Tudi: neposreden, odločen slog, manj omiljevanja

ŽENSKI (F):
- Glagolski pridevnik na -la (ženska oblika): "sem šla", "sem rekla", "sem bila", "sem naredila", "sem vedela", "sem mislila"
- Pridevnik v 1. osebi (ženska oblika, na -a): "sem vesela", "sem zadovoljna", "sem prepričana", "sem utrujena", "sem jezna"
- Tudi: omiljevanje, čustveni izrazi, kvalifikatorji

Če besedilo vsebuje glagolski pridevnik ali pridevniško obliko v 1. osebi, uporabite to kot primarni znak spola — je slovnično kodiran in zelo zanesljiv.

STAROST — slogovni znaki v slovenščini:
- 0-25:  spletni sleng, okrajšave (tko, tk, kr, lol), neformalni zapis, kratke povedi, malo ločil
- 26-35: mešanica formalnega in neformalnega, polne povedi, reference na delo ali odraslo življenje
- 36-65: formalno besedišče, daljše povedi, tradicionalni izrazi, malo slenga
- 66-:   zelo formalen slog, zastareli izrazi (lepo prosim, spoštovano, cenjeni), brez okrajšav, dolge in zapletene povedi

Dovoljene oznake:
Spol: M ali F
Starost: 0-25 | 26-35 | 36-65 | 66-

Pravila:
- Izberite natanko en spol in eno starostno skupino.
- Če niste prepričani, izberite najverjetnejšo možnost.
- Ne razlagajte svojega odgovora.
- Izpišite samo zahtevano obliko, nič drugega.

Oblika odgovora:
Gender: <M ali F>
Age: <0-25 | 26-35 | 36-65 | 66->

Besedilo:
\"\"\"{TEXT}\"\"\"
"""

GENDER_LABELS = ["F", "M"]
AGE_LABELS    = ["0-25", "26-35", "36-65", "66-"]

def query_llm(text: str) -> str:
    prompt = SLOVENE_PROMPT.replace("{TEXT}", text)
    error_count = 0
    while True:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 20},
            timeout=60,
        )
        if r.status_code == 429:
            wait = int(r.headers.get("retry-after", 30))
            print(f"    Rate limited — waiting {wait}s")
            time.sleep(wait)
            continue
        try:
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            error_count += 1
            if error_count >= 5:
                raise
            time.sleep(5 * error_count)

def parse_gender(t): m = re.search(r"Gender:\s*(M|F)\b", t, re.I); return m.group(1).upper() if m else None
def parse_age(t):    m = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", t);    return m.group(1) if m else None

def print_metrics(gold, pred, labels, task):
    df = pd.DataFrame({"gold": gold, "pred": pred}).dropna()
    valid = len(df)
    print(f"\n{'='*60}")
    print(f"{task.upper()} — SL v2 / LLaMA (n={valid}/{len(gold)} valid)")
    print(f"{'='*60}")
    if valid == 0:
        print("No valid predictions."); return
    print(f"Parse failure rate: {(len(gold)-valid)/len(gold):.3f}")
    print(f"Macro F1:    {f1_score(df['gold'], df['pred'], average='macro',    labels=labels, zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(df['gold'], df['pred'], average='weighted', labels=labels, zero_division=0):.4f}")
    print(classification_report(df["gold"], df["pred"], labels=labels, zero_division=0))

def main():
    df = pd.read_csv(DATA_PATH, sep="\t")
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["age"]    = df["age"].astype(str).str.strip()
    df = df.dropna(subset=["text","gender","age"])
    df = df[df["text"].str.strip() != ""]
    df = df[df["gender"].isin(["M","F"])].reset_index(drop=True)

    print(f"Slovene data: {len(df)} rows")

    # Same stratified sample as v1
    frac  = SAMPLE_N / len(df)
    parts = [grp.sample(frac=frac, random_state=RANDOM_SEED) for _, grp in df.groupby("gender")]
    sample = pd.concat(parts).sample(n=min(SAMPLE_N, sum(len(p) for p in parts)),
                                     random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Sample: {len(sample)} rows | gender: {dict(sample['gender'].value_counts())}")

    start_idx   = 0
    pred_gender = [None] * len(sample)
    pred_age    = [None] * len(sample)
    raw_outputs = [None] * len(sample)

    if os.path.exists(CHECKPOINT_CSV):
        ckpt = pd.read_csv(CHECKPOINT_CSV)
        done = int(ckpt["pred_gender"].notna().sum())
        if done > 0:
            start_idx = done
            pred_gender[:done] = ckpt["pred_gender"].tolist()[:done]
            pred_age[:done]    = ckpt["pred_age"].tolist()[:done]
            raw_outputs[:done] = ckpt["raw_output"].tolist()[:done]
            print(f"Resuming from row {start_idx}")

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set"); return

    est_min = (len(sample) - start_idx) * SLEEP_PER_CALL / 60
    print(f"\nRunning {len(sample) - start_idx} calls (~{est_min:.0f} min estimated)...")

    for i in range(start_idx, len(sample)):
        text = str(sample.iloc[i]["text"])[:MAX_TEXT_LENGTH]
        try:
            raw  = query_llm(text)
            g, a = parse_gender(raw), parse_age(raw)
        except Exception as e:
            print(f"  Error row {i}: {e}")
            raw, g, a = None, None, None

        pred_gender[i] = g
        pred_age[i]    = a
        raw_outputs[i] = raw

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(sample) - 1:
            ckpt_df = sample.copy()
            ckpt_df["pred_gender"] = pred_gender
            ckpt_df["pred_age"]    = pred_age
            ckpt_df["raw_output"]  = raw_outputs
            ckpt_df.to_csv(CHECKPOINT_CSV, index=False)
            print(f"  {i+1}/{len(sample)} — checkpoint saved")

        time.sleep(SLEEP_PER_CALL)

    sample["pred_gender"] = pred_gender
    sample["pred_age"]    = pred_age
    sample["raw_output"]  = raw_outputs
    sample.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinal results saved to: {OUTPUT_CSV}")

    print_metrics(sample["gender"].tolist(), pred_gender, GENDER_LABELS, "gender")
    print_metrics(sample["age"].tolist(),    pred_age,    AGE_LABELS,    "age")

    # Comparison table
    v1 = pd.read_csv(os.path.join(base, "llm_slovene_results.csv"))
    print("\n" + "="*60)
    print("LLaMA GENDER — v1 (generic) vs v2 (Slovene-morphology-aware)")
    print("="*60)
    for label, v1f, v2f in [
        ("Macro F1", 0.537, None),
        ("F1-F",     0.454, None),
        ("F1-M",     0.621, None),
    ]:
        v2_valid = pd.DataFrame({"gold": sample["gender"], "pred": pred_gender}).dropna()
        v2_macro = f1_score(v2_valid["gold"], v2_valid["pred"], average="macro",    labels=GENDER_LABELS, zero_division=0)
        v2_per   = f1_score(v2_valid["gold"], v2_valid["pred"], average=None,       labels=GENDER_LABELS, zero_division=0)
        print(f"  v1 Macro F1: 0.537  F1-F: 0.454  F1-M: 0.621")
        print(f"  v2 Macro F1: {v2_macro:.3f}  F1-F: {v2_per[0]:.3f}  F1-M: {v2_per[1]:.3f}")
        break

if __name__ == "__main__":
    main()

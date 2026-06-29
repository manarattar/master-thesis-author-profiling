"""
Slovene Qwen3-32B experiment — Slovene-morphology-aware prompt.
Compares against LLaMA v1/v2 on the same 619-row stratified sample.
Uses the same random seed as run_slovene_llama.py for direct comparison.

Fixes applied vs original:
  1. /nothink appended to prompt — disables extended thinking, response stays tiny
  2. max_tokens=50 — sufficient for Gender/Age answer only
  3. Retry on parse failure (up to 3 attempts per row)
  4. Retry on network/connection errors with backoff (not just 429)
"""

import os, re, time, requests
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
MODEL    = "qwen/qwen3-32b"

SAMPLE_N         = 619
SLEEP_PER_CALL   = 2.5
CHECKPOINT_EVERY = 50
MAX_TEXT_LENGTH  = 1500
RANDOM_SEED      = 42

base = os.path.dirname(__file__)
DATA_PATH      = os.path.join(base, "..", "data", "LILAH_data", "merged-sl-meta-lilah.tsv")
OUTPUT_CSV     = os.path.join(base, "llm_slovene_results_qwen.csv")
CHECKPOINT_CSV = os.path.join(base, "llm_slovene_checkpoint_qwen.csv")

# ── Prompt — /nothink at end disables Qwen3 extended thinking ─────────────────
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

/nothink"""

GENDER_LABELS = ["F", "M"]
AGE_LABELS    = ["0-25", "26-35", "36-65", "66-"]

def strip_thinking(text: str) -> str:
    m = re.search(r"</think>(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text

def query_llm(text: str) -> str:
    """Call Groq API. Retries on 429 and connection errors."""
    prompt = SLOVENE_PROMPT.replace("{TEXT}", text)
    error_count = 0
    while True:
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.1,
                      "max_tokens": 50},   # tiny — no thinking block needed
                timeout=60,
            )
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 30))
                print(f"    Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.ConnectionError:
            error_count += 1
            wait = min(60 * error_count, 300)
            print(f"    Connection error — waiting {wait}s (attempt {error_count})")
            time.sleep(wait)
            if error_count >= 10:
                raise

        except Exception as e:
            error_count += 1
            if error_count >= 5:
                raise
            time.sleep(5 * error_count)

def parse_gender(t):
    t = strip_thinking(t)
    m = re.search(r"Gender:\s*(M|F)\b", t, re.I)
    return m.group(1).upper() if m else None

def parse_age(t):
    t = strip_thinking(t)
    m = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", t)
    return m.group(1) if m else None

def query_with_retry(text: str, max_parse_attempts: int = 3):
    """Call LLM and retry up to max_parse_attempts times if parse fails."""
    for attempt in range(max_parse_attempts):
        raw = query_llm(text)
        g   = parse_gender(raw)
        a   = parse_age(raw)
        if g is not None and a is not None:
            return raw, g, a
        if attempt < max_parse_attempts - 1:
            print(f"    Parse failed (attempt {attempt+1}) — retrying")
            time.sleep(2)
    return raw, g, a   # return best effort after all attempts

def print_metrics(gold, pred, labels, task):
    df = pd.DataFrame({"gold": gold, "pred": pred}).dropna()
    valid = len(df)
    print(f"\n{'='*60}")
    print(f"{task.upper()} — SL / Qwen3-32B (n={valid}/{len(gold)} valid)")
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
    df = df.dropna(subset=["text", "gender", "age"])
    df = df[df["text"].str.strip() != ""]
    df = df[df["gender"].isin(["M", "F"])].reset_index(drop=True)
    print(f"Slovene data: {len(df)} rows")

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
            raw, g, a = query_with_retry(text)
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

    print("\n" + "="*60)
    print("GENDER COMPARISON — all Slovene runs")
    print("="*60)
    valid_df = pd.DataFrame({"gold": sample["gender"], "pred": pred_gender}).dropna()
    mac = f1_score(valid_df["gold"], valid_df["pred"], average="macro", labels=GENDER_LABELS, zero_division=0)
    per = f1_score(valid_df["gold"], valid_df["pred"], average=None,    labels=GENDER_LABELS, zero_division=0)
    print(f"  LLaMA v1 (EN prompt):  Macro=0.537  F1-F=0.454  F1-M=0.621")
    print(f"  LLaMA v2 (SL prompt):  Macro=0.583  F1-F=0.484  F1-M=0.682")
    print(f"  Qwen3-32B (SL prompt): Macro={mac:.3f}  F1-F={per[0]:.3f}  F1-M={per[1]:.3f}")

if __name__ == "__main__":
    main()

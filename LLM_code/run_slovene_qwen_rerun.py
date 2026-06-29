"""
Re-run failed Qwen3-32B SL rows only.
Fixes the max_tokens=500 issue that cut off thinking blocks before </think>.
Solution: append /nothink to disable thinking → tiny output, no truncation.
Loads llm_slovene_results_qwen.csv, fills in rows where pred_gender is None.
"""

import os, re, time, requests
import pandas as pd
from sklearn.metrics import f1_score, classification_report

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

SLEEP_PER_CALL  = 3.0
MAX_TEXT_LENGTH = 1500

base = os.path.dirname(__file__)
RESULTS_CSV = os.path.join(base, "llm_slovene_results_qwen.csv")

GENDER_LABELS = ["F", "M"]
AGE_LABELS    = ["0-25", "26-35", "36-65", "66-"]

# Same Slovene prompt — /nothink appended to disable thinking mode
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

def query_llm(text: str) -> str:
    prompt = SLOVENE_PROMPT.replace("{TEXT}", text)
    error_count = 0
    while True:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1,
                  "max_tokens": 50},   # tiny — no thinking block
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

def parse_gender(t):
    m = re.search(r"Gender:\s*(M|F)\b", t, re.I)
    return m.group(1).upper() if m else None

def parse_age(t):
    m = re.search(r"Age:\s*(0-25|26-35|36-65|66-)", t)
    return m.group(1) if m else None

def print_metrics(gold, pred, labels, task):
    df = pd.DataFrame({"gold": gold, "pred": pred}).dropna()
    valid = len(df)
    print(f"\n{'='*60}")
    print(f"{task.upper()} — SL / Qwen3-32B rerun (n={valid}/{len(gold)} valid)")
    print(f"{'='*60}")
    if valid == 0:
        print("No valid predictions."); return
    print(f"Parse failure rate: {(len(gold)-valid)/len(gold):.3f}")
    print(f"Macro F1: {f1_score(df['gold'], df['pred'], average='macro', labels=labels, zero_division=0):.4f}")
    print(classification_report(df["gold"], df["pred"], labels=labels, zero_division=0))

def main():
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set"); return

    df = pd.read_csv(RESULTS_CSV)
    failed_idx = df[df["pred_gender"].isna()].index.tolist()
    print(f"Loaded {len(df)} rows — {len(failed_idx)} need re-running")

    if not failed_idx:
        print("All rows already have predictions — nothing to do.")
        return

    est_min = len(failed_idx) * SLEEP_PER_CALL / 60
    print(f"Estimated time: ~{est_min:.0f} min\n")

    for count, i in enumerate(failed_idx):
        text = str(df.at[i, "text"])[:MAX_TEXT_LENGTH]
        try:
            raw  = query_llm(text)
            g, a = parse_gender(raw), parse_age(raw)
        except Exception as e:
            print(f"  Error row {i}: {e}")
            raw, g, a = None, None, None

        df.at[i, "pred_gender"] = g
        df.at[i, "pred_age"]    = a
        df.at[i, "raw_output"]  = raw

        if (count + 1) % 50 == 0 or count == len(failed_idx) - 1:
            df.to_csv(RESULTS_CSV, index=False)
            print(f"  {count+1}/{len(failed_idx)} re-run — checkpoint saved")

        time.sleep(SLEEP_PER_CALL)

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nFinal results saved to: {RESULTS_CSV}")

    print_metrics(df["gender"].tolist(), df["pred_gender"].tolist(), GENDER_LABELS, "gender")
    print_metrics(df["age"].tolist(),    df["pred_age"].tolist(),    AGE_LABELS,    "age")

    print("\n" + "="*60)
    print("GENDER COMPARISON — all Slovene runs")
    print("="*60)
    valid_df = pd.DataFrame({"gold": df["gender"], "pred": df["pred_gender"]}).dropna()
    mac  = f1_score(valid_df["gold"], valid_df["pred"], average="macro",    labels=GENDER_LABELS, zero_division=0)
    per  = f1_score(valid_df["gold"], valid_df["pred"], average=None,       labels=GENDER_LABELS, zero_division=0)
    print(f"  LLaMA v1 (EN prompt):  Macro=0.537  F1-F=0.454  F1-M=0.621")
    print(f"  LLaMA v2 (SL prompt):  Macro=0.583  F1-F=0.484  F1-M=0.682")
    print(f"  Qwen3-32B (SL prompt): Macro={mac:.3f}  F1-F={per[0]:.3f}  F1-M={per[1]:.3f}")

if __name__ == "__main__":
    main()

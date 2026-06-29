"""
Downloads Colab-trained BERT models from Google Drive using gdown.
Run: python download_colab_models.py
"""

import os, subprocess, sys

def install_gdown():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])

def dl(file_id, dest_path):
    import gdown
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        print(f"  Already exists ({size/1e6:.1f} MB): {dest_path}")
        return
    print(f"  Downloading -> {dest_path}")
    gdown.download(id=file_id, output=dest_path, quiet=False)

def main():
    print("Installing gdown...")
    install_gdown()

    # ── AGE model: best checkpoint is checkpoint-1340 (root missing safetensors) ──
    age_dir = "./bert_age_colab"
    os.makedirs(age_dir, exist_ok=True)
    print("\n=== BERT age model (checkpoint-1340) ===")
    dl("1gWcDUxFGy9RJA9bEt_4W3tS1kBFhUXKw", f"{age_dir}/model.safetensors")   # 418 MB
    dl("1godD6MoC_jRWchrqIxi7sQ0TdMD4b7wK", f"{age_dir}/config.json")
    dl("1YmSdlbILtrtuiugolxMT0uCBQLYTSdmD", f"{age_dir}/label_mapping.json")

    # ── GENDER model: root folder is complete ─────────────────────────────────────
    gender_dir = "./bert_gender_colab"
    os.makedirs(gender_dir, exist_ok=True)
    print("\n=== BERT gender model ===")
    dl("1owt2B84vuwg6AMfIZIJJLYP6Vu-Qbcs8", f"{gender_dir}/model.safetensors")  # 418 MB
    dl("1w4V2fztM3HwSIglpE2TT08cC1OJ0yZCn", f"{gender_dir}/config.json")
    dl("1T51bLavR509jV__I3cdkEkn6m1bdKf8U", f"{gender_dir}/tokenizer.json")
    dl("1XZFCXPoRJ5g-NfDh9GfwfLDTOYafH35W", f"{gender_dir}/tokenizer_config.json")
    dl("1nBZW1iZg6IHt6uofiAkrM7q7LxxYWvmR", f"{gender_dir}/label_mapping.json")

    print("\n=== Summary ===")
    for d in [age_dir, gender_dir]:
        print(f"\n{d}/")
        for f in sorted(os.listdir(d)):
            size = os.path.getsize(os.path.join(d, f))
            print(f"  {f}  ({size/1e6:.1f} MB)")

    print("\nDone. Now run:")
    print("  python evaluate_LiLaH.py              # age (TASK='age', MODEL_DIR='./bert_age_colab')")
    print("  # then change TASK='gender', MODEL_DIR='./bert_gender_colab' and run again")

if __name__ == "__main__":
    main()

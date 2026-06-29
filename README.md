# Age and Gender Identification of Hate Speech Authors

Master's thesis project — Vrije Universiteit Amsterdam, 2026.

## Overview

This repository contains code and figures for a cross-domain author profiling study on hate speech data. The goal is to predict the **gender** (M/F) and **age group** (0–25, 26–35, 36–65, 66+) of hate speech authors by comparing two approaches:

- **Fine-tuned encoder models** (BERT, DistilBERT, HateBERT, RoBERTa) trained on PAN14 social media data
- **Zero-shot LLMs** (LLaMA-3.1-8B, Qwen) prompted without any task-specific training

Evaluation is performed on the English subset of the [LiLaH-HAG](https://github.com/clarinsi/LiLaH) dataset — a multilingual hate speech corpus annotated with author demographics.

## Repository Structure

```
Bert_code/          BERT/DistilBERT training and prediction scripts
LLM_code/           LLM experiment scripts, prompt variants, and result figures
analysis/           Error analysis, confusion matrices, TF-IDF and stylometric analysis
  figures/          All generated figures (confusion matrices, wordclouds, distributions)
models/             Model configs and tokenizer files (no weights — download separately)
notebooks/          Google Colab notebooks for training and evaluation
results/            Result figures from experiments
scripts/            Data preparation and evaluation utilities
```

## Key Results

| Task   | Best Model       | Macro F1 |
|--------|-----------------|----------|
| Gender | LLaMA-3.1-8B    | 0.515    |
| Age    | BERT (PAN14)    | 0.310    |

Fine-tuned BERT outperforms zero-shot LLMs on age prediction. LLMs show competitive gender prediction but exhibit systematic male-prediction bias. Neither approach reliably identifies the 66+ age group, reflecting severe underrepresentation in PAN14 training data (0.2% of training vs. 10.3% of test).

## Datasets

- **Training**: PAN14 Author Profiling dataset (not included — available via [PAN @ CLEF](https://pan.webis.de/))
- **Evaluation**: LiLaH-HAG English subset (not included — available via [CLARIN.SI](https://www.clarin.si/))

Data files are excluded from this repository.

## Setup

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib wordcloud
```

For LLM experiments, install [Ollama](https://ollama.com/) and pull the required models:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

Set your Groq API key as an environment variable if using Groq-hosted models:

```bash
export GROQ_API_KEY=your_key_here
```

## Usage

**Fine-tune BERT on PAN14:**
```bash
python Bert_code/train_pan14_bert.py
```

**Run LLM zero-shot experiments:**
```bash
python LLM_code/run_best_full.py
```

**Run analysis and generate figures:**
```bash
python analysis/error_analysis.py
python analysis/confusion_matrices.py
```

## Citation

If you use LiLaH-HAG, please cite:

```
Markov et al. (2022). LiLaH: A Multilingual Hate Speech Dataset for Linguistic Analysis.
Proceedings of LREC 2022.
```

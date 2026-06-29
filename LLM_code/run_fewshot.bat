@echo off
rem Set your Groq API key: set GROQ_API_KEY=your_key_here
set PYTHONUNBUFFERED=1
cd /d "E:\Manar\Thesis_master\LiLaH-dataset-profiling\LLM_code"
python -u run_fewshot.py > fewshot_run.log 2>&1


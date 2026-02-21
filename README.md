# Hindi ASR Fine-tuning (Whisper-small)

This workspace now has an end-to-end pipeline for:

1.  Preprocessing JoshTalk Hindi CSV into a Hugging Face dataset.
2.  Fine-tuning `openai/whisper-small`.
3.  Evaluating baseline vs fine-tuned model on FLEURS Hindi (`google/fleurs`, `hi_in`, `test`).

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Preprocess your CSV

```bash
python scripts/prepare_joshtalk_hindi.py ^  --csv_path "FT_Data - data.csv" ^  --output_dir data/joshtalk_hindi_hf ^  --audio_dir data/raw_audio
```

What this script does:

-   Filters `language == "hi"`.
-   Fixes broken links by constructing:
    -   `https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_audio.wav`
    -   `https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_transcription.json`
    -   `https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_metadata.json`
-   Downloads audio + transcription.
-   Creates short segment clips from transcription timestamps.
-   Creates `train`/`validation` split and saves HF dataset at `data/joshtalk_hindi_hf`.
-   Writes summary to `data/joshtalk_hindi_hf/prepare_summary.json`.

Note:
-   Re-run preprocessing if you prepared data earlier with the old script. The current script builds segment-level samples required for stable Whisper training.

## 3) Fine-tune Whisper-small

Use a GPU for practical runtime.

```bash
python scripts/train_whisper_hindi.py ^  --dataset_dir data/joshtalk_hindi_hf ^  --output_dir outputs/whisper-small-hi ^  --num_train_epochs 8 ^  --per_device_train_batch_size 8 ^  --gradient_accumulation_steps 2
```

Artifacts:

-   Model + processor: `outputs/whisper-small-hi`
-   Training/eval metrics: `outputs/whisper-small-hi/train_metrics.json`

## 4) Evaluate on FLEURS Hindi test

```bash
python scripts/evaluate_fleurs_hi.py ^  --finetuned_model outputs/whisper-small-hi ^  --output_json outputs/fleurs_hi_eval.json
```

Output:

-   `outputs/fleurs_hi_eval.json` with side-by-side metrics:
    -   Baseline `openai/whisper-small` WER/CER
    -   Fine-tuned model WER/CER

Offline metrics:
-   WER/CER are computed with local `jiwer` (no `evaluate.load("cer")` download dependency).

## What to submit for your assignment

1.  **Preprocessing report (part a)**:

-   Data size before/after filtering.
-   URL correction logic.
-   Any dropped samples and why.
-   Final train/validation counts.
-   Text normalization choices.

2.  **Training + evaluation report (part b)**:

-   Fine-tuning config (epochs, batch size, LR, hardware).
-   Baseline FLEURS Hindi test WER/CER.
-   Fine-tuned FLEURS Hindi test WER/CER.
-   Relative improvement and 3-5 qualitative prediction examples.

3.  **Reproducibility**:

-   Commands run.
-   Random seed.
-   Script versions from this repo

## Repository Scope (Important)

This GitHub repository tracks **code and lightweight reports only**.

Not pushed to GitHub:
- Raw/processed datasets
- Audio clips and segment folders
- Large training artifacts/checkpoints
- Large generated CSVs/outputs

Data is intentionally excluded via `.gitignore` to keep the repository small and clean.

Included submission docs:
- `outputs/submission_q2_to_q4/Assignment_Q1_to_Q4_Report.md`
- `outputs/submission_q2_to_q4/submission_manifest.csv`

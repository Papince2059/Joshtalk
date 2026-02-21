# Question 1 Report

## Dataset
- Source metadata CSV: `FT_Data - data.csv`
- Language filtered: `hi`
- URL correction used:
  - `https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_audio.wav`
  - `https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_transcription.json`
  - `https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_metadata.json`

## Preprocessing Done
- Parsed metadata and filtered Hindi rows.
- Downloaded full audio from corrected `upload_goai` URLs.
- Downloaded segment-level transcription JSON per recording.
- Cut segment clips using `start/end` timestamps.
- Normalized transcript text (trim + whitespace normalization).
- Removed empty transcripts and invalid-duration segments.
- Exported Hugging Face dataset split for training.

## Preprocessing Summary
- Input Hindi recordings: `104`
- Prepared segment samples: `5794`
- Train split: `5214`
- Validation split: `580`
- Failures: `0`
- Source: `data/joshtalk_hindi_hf/prepare_summary.json`

## Fine-tuning Status
- Model: `openai/whisper-small`
- Training run was interrupted due to GPU limitation.
- Progress at interruption: `[401/800], Epoch 1.23/3`
- Logged validation snapshot (step `200`):
  - Training loss: `0.421100`
  - Validation loss: `0.391110`
  - WER: `41.885734`
  - CER: `20.908851`

## WER Table (Structured)
See `outputs/q1_wer_table.csv`.

| Model | Eval Dataset | Split | WER | CER | Status | Notes |
|---|---|---|---:|---:|---|---|
| `openai/whisper-small` | `google/fleurs hi_in` | `test` | `0.83` | - | `reported` | user-reported baseline |
| `whisper-small fine-tune (partial)` | JoshTalk validation | `step=200` | `0.41885734` | `0.20908851` | `partial_training` | interrupted run |

## Commands Used / Reproducibility
Preprocess:
```powershell
python scripts/prepare_joshtalk_hindi.py --csv_path "FT_Data - data.csv" --output_dir data/joshtalk_hindi_hf --audio_dir data/raw_audio --clips_dir data/clips --num_workers 8
```

Fine-tune:
```powershell
python scripts/train_whisper_hindi.py --dataset_dir data/joshtalk_hindi_hf --output_dir outputs/whisper-small-hi --num_train_epochs 3 --max_steps 800 --per_device_train_batch_size 8 --gradient_accumulation_steps 2
```

FLEURS evaluation (baseline + fine-tuned) when checkpoint is available:
```powershell
python scripts/evaluate_fleurs_hi.py --finetuned_model outputs/whisper-small-hi --output_json outputs/fleurs_hi_eval.json
```

## Note
The assignment asks baseline vs fine-tuned evaluation on FLEURS Hindi test. Since the fine-tune run was incomplete, this report includes the partial training result and a reproducible path to finish final evaluation once GPU runtime is available.

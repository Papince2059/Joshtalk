# Final Report: Questions 1 to 4

## Question 1: Hindi ASR Fine-tuning (Whisper-small)
### Method
- Preprocessed Hindi subset into segment-level dataset (`data/joshtalk_hindi_hf`).
- Trained `openai/whisper-small` and evaluated during training.
- Training was interrupted due to GPU limits (partial run only).

### Data Preparation Summary
- Input Hindi recordings: 104
- Prepared segment samples: 5794
- Train rows: 5214
- Validation rows: 580
- Failures: 0

### Partial Training Result (User-provided run log)
- Training progress at interruption: `[401/800], Epoch 1.23/3`
- Baseline WER: `0.830` (83.00%)
- Validation WER at step 200: `0.418857` (41.89%)
- Validation CER at step 200: `0.209089` (20.91%)
- Relative WER improvement vs baseline: `49.54%`
- Note: this is **not** final model performance because training did not finish.

### Q1 Artifacts
- Chart: `outputs/final_report/charts/q1_wer_comparison.png`
- Preprocess summary: `data/joshtalk_hindi_hf/prepare_summary.json`

## Question 2: Disfluency Detection and Segmentation
### Method
- Hybrid rule-based detection using:
  - Disfluency lexicon matching from provided sheet.
  - Regex rules for repetition/prolongation/false-start patterns.
- For each disfluency-positive segment, clipped audio from full recordings using transcript timestamps.
- Exported one row per occurrence in CSV + saved corresponding segmented clips.

### Results
- Input recordings: 104
- Detected disfluency rows: 7926
- Recordings with hits: 104
- Failures: 0

### Q2 Deliverables
- Output sheet: `outputs/q2_disfluency_segments.csv`
- Segmented clips: `data/q2_disfluency_clips`
- Methodology summary: `outputs/q2_methodology.md`
- Chart: `outputs/final_report/charts/q2_disfluency_distribution.png`

## Question 3: Correct vs Incorrect Spelling in Unique Words
### Method
- Input unique-word list was processed from `Unique Words Data - Sheet1.csv`.
- Rule-based orthography checks:
  - Non-Devanagari content detection.
  - Invalid Hindi character-sequence constraints.
  - Obvious noisy repetition/sign misuse checks.
- Classified each unique word into `correct spelling` or `incorrect spelling`.

### Results
- Total unique words: 175,780
- Correct spelling words: 148,396
- Incorrect spelling words: 27,384

### Q3 Deliverables
- Output sheet: `outputs/q3_word_spelling_labels.csv`
- Summary: `outputs/q3_spelling_summary.json`
- Methodology summary: `outputs/q3_methodology.md`
- Chart: `outputs/final_report/charts/q3_spelling_counts.png`

## Question 4: Multi-model Transcript Evaluation
### Method
- Used `Human` as reference and computed normalized-text WER/CER for each model (`Model H/i/k/l/m/n`).
- Ranked models by aggregate WER then CER.
- Added per-segment analysis file for deeper review.

### Results
- Input segments: 46
- Best model: `Model i`
- Best model WER: `0.001222`
- Best model CER: `0.000829`

### Q4 Deliverables
- Model metrics: `outputs/q4_model_metrics.csv`
- Segment analysis: `outputs/q4_segment_analysis.csv`
- Summary: `outputs/q4_summary.json`
- Short report: `outputs/q4_report.md`
- Chart: `outputs/final_report/charts/q4_model_metrics.png`

## Reproducibility
- Q2 script: `scripts/prepare_disfluency_dataset.py`
- Q3 script: `scripts/classify_q3_spelling.py`
- Q4 script: `scripts/evaluate_q4_models.py`
- Consolidated report script: `scripts/build_final_report.py`

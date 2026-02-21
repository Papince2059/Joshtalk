# Assignment Submission Report (Questions 2 to 4)

## Scope
This report covers only:
- Question 2: Disfluency detection and segmented dataset creation
- Question 3: Correct vs incorrect spelling classification of unique words
- Question 4: Multi-model transcript evaluation

---

## Question 2
### Objective
Detect target Hindi speech disfluencies from segment transcripts, extract corresponding audio clips, and create a structured occurrence-level sheet.

### Methodology (Short)
- Loaded Hindi metadata and resolved corrected `upload_goai` URLs for transcript/audio.
- Used hybrid disfluency detection:
  - Lexicon matching from provided disfluency list.
  - Regex rules for repetition, prolongation, and false-start patterns.
- For each segment with a detected disfluency, clipped audio from full recording using segment `start/end` timestamps.
- Saved one row per disfluency occurrence in CSV with clip path and metadata.

### Output Summary
- Input recordings processed: `104`
- Disfluency occurrences detected: `7926`
- Recordings with at least one hit: `104`
- Failures: `0`

### Deliverables
- Sheet (occurrence-level): `outputs/q2_disfluency_segments.csv`
- Segmented clips directory: `data/q2_disfluency_clips`
- Summary JSON: `outputs/q2_disfluency_summary.json`
- Methodology note: `outputs/q2_methodology.md`

---

## Question 3
### Objective
Classify unique words into:
- `correct spelling`
- `incorrect spelling`

### Methodology (Short)
- Used the unique-word file (`Unique Words Data - Sheet1.csv`).
- Applied Unicode normalization and Hindi orthography checks:
  - Non-Devanagari/script-noise detection.
  - Invalid sequence/sign rules (obvious spelling/character errors).
- Produced two-column output: `word`, `spelling_label`.

### Output Summary
- Total unique words processed: `175,780`
- Correct spelling: `148,396`
- Incorrect spelling: `27,384`

### Deliverables
- Output sheet: `outputs/q3_word_spelling_labels.csv`
- Summary JSON: `outputs/q3_spelling_summary.json`
- Methodology note: `outputs/q3_methodology.md`

---

## Question 4
### Objective
Evaluate model transcripts against human reference transcripts and select best-performing model using WER/CER.

### Methodology (Short)
- Used `Human` as reference in `Question 4 - Task.csv`.
- Normalized text and computed aggregate WER/CER for:
  - `Model H`, `Model i`, `Model k`, `Model l`, `Model m`, `Model n`
- Ranked models by WER (primary) then CER.
- Generated per-segment analysis for deeper review.

### Output Summary
- Input segments: `46`
- Best model: `Model i`
- Best WER: `0.001222`
- Best CER: `0.000829`

### Deliverables
- Model metrics table: `outputs/q4_model_metrics.csv`
- Segment-level analysis: `outputs/q4_segment_analysis.csv`
- Summary JSON: `outputs/q4_summary.json`
- Short report: `outputs/q4_report.md`

---

## Reproducibility (Scripts Used)
- Q2: `scripts/prepare_disfluency_dataset.py`
- Q3: `scripts/classify_q3_spelling.py`
- Q4: `scripts/evaluate_q4_models.py`


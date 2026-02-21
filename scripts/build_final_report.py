import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build consolidated Q1-Q4 report with charts.")
    parser.add_argument("--q1_baseline_wer", type=float, default=0.83)
    parser.add_argument("--q1_step200_val_wer", type=float, default=41.885734)
    parser.add_argument("--q1_step200_val_cer", type=float, default=20.908851)
    parser.add_argument("--q1_partial_step", type=int, default=200)
    parser.add_argument("--q1_progress_text", type=str, default="[401/800], Epoch 1.23/3")
    parser.add_argument("--out_dir", type=str, default="outputs/final_report")
    return parser.parse_args()


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    q1_prep = json.loads(Path("data/joshtalk_hindi_hf/prepare_summary.json").read_text(encoding="utf-8"))
    q2 = json.loads(Path("outputs/q2_disfluency_summary.json").read_text(encoding="utf-8"))
    q3 = json.loads(Path("outputs/q3_spelling_summary.json").read_text(encoding="utf-8"))
    q4 = json.loads(Path("outputs/q4_summary.json").read_text(encoding="utf-8"))
    q4_metrics = pd.read_csv("outputs/q4_model_metrics.csv", encoding="utf-8-sig")

    q1_val_wer_decimal = args.q1_step200_val_wer / 100.0
    q1_val_cer_decimal = args.q1_step200_val_cer / 100.0
    q1_rel_improvement = (args.q1_baseline_wer - q1_val_wer_decimal) / args.q1_baseline_wer

    # Q1 chart
    plt.figure(figsize=(7, 4))
    plt.bar(["Baseline WER", "Partial FT WER\n(step 200)"], [args.q1_baseline_wer, q1_val_wer_decimal], color=["#888888", "#2A9D8F"])
    plt.ylabel("WER (lower is better)")
    plt.title("Q1 ASR WER Comparison")
    plt.ylim(0, max(args.q1_baseline_wer, q1_val_wer_decimal) * 1.2)
    for i, v in enumerate([args.q1_baseline_wer, q1_val_wer_decimal]):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    q1_chart = charts_dir / "q1_wer_comparison.png"
    plt.savefig(q1_chart, dpi=160)
    plt.close()

    # Q2 chart
    q2_counts = pd.Series(q2["disfluency_type_counts"]).sort_values(ascending=False)
    plt.figure(figsize=(8, 4.5))
    q2_counts.plot(kind="bar", color="#E76F51")
    plt.ylabel("Detected occurrences")
    plt.title("Q2 Disfluency Type Distribution")
    plt.tight_layout()
    q2_chart = charts_dir / "q2_disfluency_distribution.png"
    plt.savefig(q2_chart, dpi=160)
    plt.close()

    # Q3 chart
    plt.figure(figsize=(6.5, 4))
    vals = [q3["correct_spelling_words"], q3["incorrect_spelling_words"]]
    labels = ["Correct spelling", "Incorrect spelling"]
    colors = ["#2A9D8F", "#E76F51"]
    plt.bar(labels, vals, color=colors)
    plt.ylabel("Unique words")
    plt.title("Q3 Spelling Classification")
    for i, v in enumerate(vals):
        plt.text(i, v + max(vals) * 0.01, f"{v:,}", ha="center", va="bottom")
    plt.tight_layout()
    q3_chart = charts_dir / "q3_spelling_counts.png"
    plt.savefig(q3_chart, dpi=160)
    plt.close()

    # Q4 chart
    plt.figure(figsize=(8, 4.5))
    x = range(len(q4_metrics))
    width = 0.38
    plt.bar([i - width / 2 for i in x], q4_metrics["wer"], width=width, label="WER", color="#264653")
    plt.bar([i + width / 2 for i in x], q4_metrics["cer"], width=width, label="CER", color="#F4A261")
    plt.xticks(list(x), q4_metrics["model"], rotation=0)
    plt.ylabel("Error rate")
    plt.title("Q4 Model-wise WER/CER")
    plt.legend()
    plt.tight_layout()
    q4_chart = charts_dir / "q4_model_metrics.png"
    plt.savefig(q4_chart, dpi=160)
    plt.close()

    report_md = out_dir / "final_report_q1_q4.md"
    report_text = f"""# Final Report: Questions 1 to 4

## Question 1: Hindi ASR Fine-tuning (Whisper-small)
### Method
- Preprocessed Hindi subset into segment-level dataset (`data/joshtalk_hindi_hf`).
- Trained `openai/whisper-small` and evaluated during training.
- Training was interrupted due to GPU limits (partial run only).

### Data Preparation Summary
- Input Hindi recordings: {q1_prep["input_rows_hi"]}
- Prepared segment samples: {q1_prep["prepared_rows"]}
- Train rows: {q1_prep["train_rows"]}
- Validation rows: {q1_prep["validation_rows"]}
- Failures: {q1_prep["failures"]}

### Partial Training Result (User-provided run log)
- Training progress at interruption: `{args.q1_progress_text}`
- Baseline WER: `{args.q1_baseline_wer:.3f}` ({pct(args.q1_baseline_wer)})
- Validation WER at step {args.q1_partial_step}: `{q1_val_wer_decimal:.6f}` ({pct(q1_val_wer_decimal)})
- Validation CER at step {args.q1_partial_step}: `{q1_val_cer_decimal:.6f}` ({pct(q1_val_cer_decimal)})
- Relative WER improvement vs baseline: `{q1_rel_improvement * 100:.2f}%`
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
- Input recordings: {q2["input_rows_hi"]}
- Detected disfluency rows: {q2["detected_disfluency_rows"]}
- Recordings with hits: {q2["unique_recordings_with_hits"]}
- Failures: {q2["failures"]}

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
- Total unique words: {q3["total_unique_words"]:,}
- Correct spelling words: {q3["correct_spelling_words"]:,}
- Incorrect spelling words: {q3["incorrect_spelling_words"]:,}

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
- Input segments: {q4["input_rows"]}
- Best model: `{q4["best_model"]}`
- Best model WER: `{q4["best_model_wer"]:.6f}`
- Best model CER: `{q4["best_model_cer"]:.6f}`

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
"""
    report_md.write_text(report_text, encoding="utf-8")

    summary = {
        "report_path": str(report_md.resolve()),
        "charts": [
            str(q1_chart.resolve()),
            str(q2_chart.resolve()),
            str(q3_chart.resolve()),
            str(q4_chart.resolve()),
        ],
    }
    (out_dir / "report_assets.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

import pandas as pd
from jiwer import cer, wer


MODEL_COLUMNS = ["Model H", "Model i", "Model k", "Model l", "Model m", "Model n"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Q4 model transcriptions against Human references."
    )
    parser.add_argument("--input_csv", required=True, help="Path to Question 4 CSV")
    parser.add_argument("--metrics_csv", required=True, help="Output CSV for model-level metrics")
    parser.add_argument("--segment_csv", required=True, help="Output CSV for per-segment error analysis")
    parser.add_argument("--summary_json", required=True, help="Output summary JSON")
    parser.add_argument("--report_md", default=None, help="Optional markdown report path")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFC", str(text or ""))
    t = t.replace("\n", " ").replace("–", "-").replace("—", "-")
    t = re.sub(r"[.,!?।;:'\"()\[\]{}]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    refs = [normalize_text(x) for x in df["Human"].tolist()]
    rows = []
    for model_col in MODEL_COLUMNS:
        hyps = [normalize_text(x) for x in df[model_col].tolist()]
        rows.append(
            {
                "model": model_col,
                "wer": float(wer(refs, hyps)),
                "cer": float(cer(refs, hyps)),
            }
        )
    out = pd.DataFrame(rows).sort_values(["wer", "cer"], ascending=True).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out[["rank", "model", "wer", "cer"]]


def build_segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for idx, row in df.iterrows():
        ref = normalize_text(row["Human"])
        per_model = []
        for model_col in MODEL_COLUMNS:
            hyp = normalize_text(row[model_col])
            per_model.append((model_col, wer([ref], [hyp]), cer([ref], [hyp]), hyp))

        best = sorted(per_model, key=lambda x: (x[1], x[2]))[0]
        min_wer = min(x[1] for x in per_model)
        tied = sorted([x[0] for x in per_model if abs(x[1] - min_wer) < 1e-12])
        rows.append(
            {
                "segment_idx": int(idx),
                "segment_url_link": row["segment_url_link"],
                "human_text": row["Human"],
                "best_model": best[0],
                "best_model_wer": float(best[1]),
                "best_model_cer": float(best[2]),
                "best_tie_models": "|".join(tied),
                "best_tie_count": int(len(tied)),
                "model_h_wer": float(per_model[0][1]),
                "model_i_wer": float(per_model[1][1]),
                "model_k_wer": float(per_model[2][1]),
                "model_l_wer": float(per_model[3][1]),
                "model_m_wer": float(per_model[4][1]),
                "model_n_wer": float(per_model[5][1]),
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, metrics_df: pd.DataFrame, segment_df: pd.DataFrame):
    best = metrics_df.iloc[0]
    best_model = str(best["model"])
    win_counts = segment_df["best_model"].value_counts().to_dict()
    tie_free = segment_df[segment_df["best_tie_count"] == 1]["best_model"].value_counts().to_dict()

    lines = [
        "# Question-4 Evaluation Report",
        "",
        "## Approach",
        "Compared each model transcript against Human transcript using normalized-text WER and CER.",
        "",
        "## Result",
        f"- Best model by WER/CER: **{best_model}**",
        f"- WER: **{best['wer']:.4f}**",
        f"- CER: **{best['cer']:.4f}**",
        "",
        "## Per-segment wins",
        "(`best_model` uses deterministic tie-break by model-column order.)",
    ]
    for m in MODEL_COLUMNS:
        lines.append(f"- {m}: {int(win_counts.get(m, 0))}")
    lines.extend(["", "## Per-segment unique wins (no ties)"])
    for m in MODEL_COLUMNS:
        lines.append(f"- {m}: {int(tie_free.get(m, 0))}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    metrics_csv = Path(args.metrics_csv)
    segment_csv = Path(args.segment_csv)
    summary_json = Path(args.summary_json)

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    required = {"segment_url_link", "Human", *MODEL_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    metrics_df = build_metrics(df)
    segment_df = build_segment_analysis(df)

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    segment_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    segment_df.to_csv(segment_csv, index=False, encoding="utf-8-sig")

    best = metrics_df.iloc[0]
    summary = {
        "input_rows": int(len(df)),
        "best_model": str(best["model"]),
        "best_model_wer": float(best["wer"]),
        "best_model_cer": float(best["cer"]),
        "model_ranking": metrics_df.to_dict(orient="records"),
        "per_segment_best_model_counts": segment_df["best_model"].value_counts().to_dict(),
        "per_segment_unique_best_model_counts": segment_df[segment_df["best_tie_count"] == 1]["best_model"]
        .value_counts()
        .to_dict(),
        "metrics_csv": str(metrics_csv.resolve()),
        "segment_csv": str(segment_csv.resolve()),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.report_md:
        write_report(Path(args.report_md), metrics_df, segment_df)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

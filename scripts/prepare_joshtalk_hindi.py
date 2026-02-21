import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import pandas as pd
import requests
import soundfile as sf
from datasets import Dataset, DatasetDict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare JoshTalk Hindi ASR dataset for Whisper.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to source CSV.")
    parser.add_argument("--output_dir", type=str, default="data/joshtalk_hindi_hf", help="HF dataset output path.")
    parser.add_argument("--audio_dir", type=str, default="data/raw_audio", help="Downloaded full audio directory.")
    parser.add_argument("--clips_dir", type=str, default="data/clips", help="Segment clip directory.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel download workers.")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--min_segment_sec", type=float, default=0.4, help="Drop very short segments.")
    parser.add_argument("--max_segment_sec", type=float, default=30.0, help="Drop segments longer than Whisper window.")
    return parser.parse_args()


def extract_folder_id(row):
    for col in ["rec_url_gcp", "transcription_url_gcp", "metadata_url_gcp"]:
        url = str(row.get(col, ""))
        m = re.search(r"/hi/([^/]+)/", url)
        if m:
            return m.group(1)
        m2 = re.search(r"upload_goai/([^/]+)/", url)
        if m2:
            return m2.group(1)
    return None


def build_new_urls(folder_id, recording_id):
    base = f"https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}"
    return {
        "audio_url": f"{base}_audio.wav",
        "transcription_url": f"{base}_transcription.json",
        "metadata_url": f"{base}_metadata.json",
    }


def normalize_text(text):
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def fetch_json(url, timeout=45):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def download_file(url, path, timeout=120):
    if path.exists() and path.stat().st_size > 0:
        return
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)


def build_segment_records(row, audio_root, clips_root, min_segment_sec, max_segment_sec):
    recording_id = str(row["recording_id"])
    user_id = str(row["user_id"])
    language = str(row["language"])

    folder_id = extract_folder_id(row)
    if not folder_id:
        raise ValueError(f"Could not infer folder id for recording_id={recording_id}")

    urls = build_new_urls(folder_id, recording_id)
    segments = fetch_json(urls["transcription_url"])

    full_audio_path = Path(audio_root) / f"{recording_id}.wav"
    download_file(urls["audio_url"], full_audio_path)

    wav, sr = librosa.load(str(full_audio_path), sr=16000, mono=True)

    rec_dir = Path(clips_root) / recording_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    out = []
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = normalize_text(seg.get("text", ""))

        if not text:
            continue
        if end <= start:
            continue
        seg_dur = end - start
        if seg_dur < min_segment_sec or seg_dur > max_segment_sec:
            continue

        s = max(0, int(start * sr))
        e = min(len(wav), int(end * sr))
        if e <= s:
            continue

        clip = wav[s:e]
        clip_name = f"{recording_id}_{idx:04d}.wav"
        clip_path = rec_dir / clip_name
        if not clip_path.exists():
            sf.write(str(clip_path), clip, sr)

        out.append(
            {
                "id": f"{recording_id}_{idx:04d}",
                "recording_id": recording_id,
                "user_id": user_id,
                "language": language,
                "audio": str(clip_path.resolve()),
                "sentence": text,
                "start": start,
                "end": end,
                "duration_sec": round(seg_dur, 3),
                "audio_url": urls["audio_url"],
                "transcription_url": urls["transcription_url"],
            }
        )

    return out


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    required_cols = {"user_id", "recording_id", "language"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["language"] == "hi"].copy()
    if args.max_rows:
        df = df.head(args.max_rows)

    rows = df.to_dict(orient="records")
    records = []
    failures = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [
            ex.submit(
                build_segment_records,
                r,
                args.audio_dir,
                args.clips_dir,
                args.min_segment_sec,
                args.max_segment_sec,
            )
            for r in rows
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading + segmenting"):
            try:
                records.extend(fut.result())
            except Exception as e:
                failures.append(str(e))

    if not records:
        raise RuntimeError("No valid segment records were prepared.")

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    dataset_dict = DatasetDict({"train": split["train"], "validation": split["test"]})

    out_dir = Path(args.output_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(out_dir))

    summary = {
        "input_rows_hi": len(rows),
        "prepared_rows": len(records),
        "train_rows": len(dataset_dict["train"]),
        "validation_rows": len(dataset_dict["validation"]),
        "failures": len(failures),
        "failure_examples": failures[:10],
        "notes": "Prepared at segment-level for Whisper (short clips with short text).",
    }
    with open(out_dir / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

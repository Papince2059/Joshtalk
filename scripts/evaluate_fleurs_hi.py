import argparse
import json
import io
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper models on FLEURS Hindi test set.")
    parser.add_argument("--baseline_model", type=str, default="openai/whisper-small")
    parser.add_argument("--finetuned_model", type=str, required=True, help="Path to fine-tuned model directory.")
    parser.add_argument("--language", type=str, default="hindi")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default="outputs/fleurs_hi_eval.json")
    return parser.parse_args()


def run_model(model_name_or_path, dataset, language, task):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

    predictions = []
    references = []

    def load_audio_16k(audio_obj):
        if isinstance(audio_obj, dict):
            if "array" in audio_obj and audio_obj["array"] is not None:
                arr = np.asarray(audio_obj["array"], dtype=np.float32)
                sr = int(audio_obj.get("sampling_rate", 16000))
                if sr != 16000:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                return arr
            path = audio_obj.get("path")
            if path:
                arr, _ = librosa.load(path, sr=16000, mono=True)
                return arr
            raw = audio_obj.get("bytes")
            if raw is not None:
                arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                if sr != 16000:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                return arr
        raise ValueError("Unsupported audio format in dataset.")

    for ex in tqdm(dataset, desc=f"Evaluating {model_name_or_path}"):
        audio_arr = load_audio_16k(ex["audio"])
        ref = ex["transcription"].strip()
        if not ref:
            continue

        input_features = processor.feature_extractor(
            audio_arr, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            pred_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=225)
        pred = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        predictions.append(pred)
        references.append(ref)

    wer = jiwer.wer(references, predictions) * 100
    cer = jiwer.cer(references, predictions) * 100

    return {
        "samples_evaluated": len(predictions),
        "wer": wer,
        "cer": cer,
    }


def main():
    args = parse_args()

    ds = load_dataset("google/fleurs", "hi_in", split=args.split)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    baseline_metrics = run_model(args.baseline_model, ds, args.language, args.task)
    finetuned_metrics = run_model(args.finetuned_model, ds, args.language, args.task)

    out = {
        "dataset": "google/fleurs",
        "config": "hi_in",
        "split": args.split,
        "baseline_model": args.baseline_model,
        "finetuned_model": args.finetuned_model,
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

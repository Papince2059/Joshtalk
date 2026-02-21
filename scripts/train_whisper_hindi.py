import argparse
import json
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import jiwer
import librosa
import numpy as np
import torch
from datasets import Audio, load_from_disk
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on prepared Hindi dataset.")
    parser.add_argument("--dataset_dir", type=str, default="data/joshtalk_hindi_hf")
    parser.add_argument("--model_name", type=str, default="openai/whisper-small")
    parser.add_argument("--output_dir", type=str, default="outputs/whisper-small-hi")
    parser.add_argument("--language", type=str, default="hindi")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--num_train_epochs", type=float, default=8.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_label_length", type=int, default=448)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
            return_attention_mask=True,
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    args = parse_args()
    dataset = load_from_disk(args.dataset_dir)
    if "audio" in dataset["train"].column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    train_ds = dataset["train"]
    eval_ds = dataset["validation"]

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # Avoid forced_decoder_ids/task conflict warnings during generation.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    def load_audio_16k(audio_obj):
        if isinstance(audio_obj, str):
            arr, _ = librosa.load(audio_obj, sr=16000, mono=True)
            return arr
        if isinstance(audio_obj, dict):
            path = audio_obj.get("path")
            if path:
                arr, _ = librosa.load(path, sr=16000, mono=True)
                return arr
        raise ValueError(f"Unsupported audio object format: {type(audio_obj)}")

    def prepare_batch(batch):
        audio_arr = load_audio_16k(batch["audio"])
        batch["input_features"] = feature_extractor(
            audio_arr, sampling_rate=16000
        ).input_features[0]
        batch["labels"] = tokenizer(
            batch["sentence"], truncation=True, max_length=args.max_label_length
        ).input_ids
        return batch

    train_ds = train_ds.map(
        prepare_batch,
        remove_columns=train_ds.column_names,
        num_proc=1,
        desc="Preparing train features",
    )
    eval_ds = eval_ds.map(
        prepare_batch,
        remove_columns=eval_ds.column_names,
        num_proc=1,
        desc="Preparing eval features",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * jiwer.wer(label_str, pred_str)
        cer = 100 * jiwer.cer(label_str, pred_str)
        return {"wer": wer, "cer": cer}

    fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        fp16=fp16,
        gradient_checkpointing=True,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # `tokenizer` is deprecated in newer Transformers; prefer `processing_class` when available.
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is None:
        resume_ckpt = get_last_checkpoint(args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    eval_metrics = trainer.evaluate()
    with open(f"{args.output_dir}/train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps({"train_metrics": train_result.metrics, "eval_metrics": eval_metrics}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fine-tune badrex/Ethio-ASR-amharic on local Amharic audio.

Expected CSV columns:
    audio_path,text

Optional columns such as speaker_id, dialect, gender, and duration are allowed
and ignored by this training script. Paths may be absolute or relative to the
CSV file.

Example:
    python train_ethio_asr.py \
      --train-csv manifests/train.csv \
      --validation-csv manifests/validation.csv \
      --test-csv manifests/test.csv \
      --output-dir outputs/ethio-asr-leyu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


WHITESPACE_RE = re.compile(r"\s+")


def normalize_amharic_text(text: str) -> str:
    """Apply conservative normalization without changing Amharic spelling."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return WHITESPACE_RE.sub(" ", text).strip()


def read_manifest(csv_path: str) -> list[dict[str, str]]:
    manifest_path = Path(csv_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_path", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{manifest_path} is missing CSV columns: {sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            audio_value = (row.get("audio_path") or "").strip()
            text = normalize_amharic_text(row.get("text") or "")
            if not audio_value or not text:
                raise ValueError(
                    f"Empty audio_path or text in {manifest_path}, line {line_number}"
                )

            audio_path = Path(audio_value).expanduser()
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio not found in {manifest_path}, line {line_number}: "
                    f"{audio_path}"
                )

            rows.append({"audio_path": str(audio_path), "text": text})

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


class AmharicAudioDataset(Dataset):
    """Loads and resamples audio lazily so 500 hours are not held in RAM."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        processor: Any,
        sample_rate: int,
        min_duration_seconds: float,
        max_duration_seconds: float,
        validate_vocabulary: bool,
    ) -> None:
        self.rows = rows
        self.processor = processor
        self.sample_rate = sample_rate
        self.min_samples = int(min_duration_seconds * sample_rate)
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.input_name = processor.model_input_names[0]

        if validate_vocabulary:
            self._validate_vocabulary()

    def _validate_vocabulary(self) -> None:
        tokenizer = self.processor.tokenizer
        unk_id = tokenizer.unk_token_id
        if unk_id is None:
            return

        bad_examples: list[str] = []
        for row in self.rows:
            token_ids = tokenizer(row["text"], add_special_tokens=False).input_ids
            if unk_id in token_ids:
                bad_examples.append(row["text"])
                if len(bad_examples) == 5:
                    break
        if bad_examples:
            examples = "\n  - ".join(bad_examples)
            raise ValueError(
                "The existing Ethio-ASR tokenizer produced <unk> for some "
                f"transcriptions. Fix text normalization or vocabulary first:\n  - {examples}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, source_rate = torchaudio.load(row["audio_path"])

        if waveform.ndim != 2 or waveform.shape[0] < 1:
            raise ValueError(f"Invalid audio shape for {row['audio_path']}: {waveform.shape}")
        waveform = waveform.mean(dim=0)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_rate, self.sample_rate
            )

        sample_count = waveform.numel()
        if sample_count < self.min_samples or sample_count > self.max_samples:
            duration = sample_count / self.sample_rate
            raise ValueError(
                f"Audio duration {duration:.2f}s is outside the configured range "
                f"for {row['audio_path']}. Segment or filter it without truncating "
                "the corresponding transcript."
            )

        processed = self.processor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
        )
        labels = self.processor.tokenizer(
            row["text"], add_special_tokens=False
        ).input_ids

        return {
            self.input_name: processed[self.input_name][0],
            "labels": labels,
        }


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    input_name: str

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {self.input_name: feature[self.input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv")
    parser.add_argument("--output-dir", default="outputs/ethio-asr-leyu")
    parser.add_argument("--model-name", default="badrex/Ethio-ASR-amharic")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--min-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--dataloader-workers", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 instead of the default BF16.",
    )
    parser.add_argument(
        "--unfreeze-feature-encoder",
        action="store_true",
        help="Train the feature encoder from the beginning.",
    )
    parser.add_argument(
        "--skip-vocabulary-check",
        action="store_true",
        help="Do not fail when the tokenizer emits an unknown token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCTC.from_pretrained(args.model_name)
    model.config.ctc_loss_reduction = "mean"
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if not args.unfreeze_feature_encoder:
        freeze_method = getattr(model, "freeze_feature_encoder", None)
        if freeze_method is None:
            raise AttributeError(
                "This Transformers version does not expose "
                "model.freeze_feature_encoder(). Upgrade Transformers or use "
                "--unfreeze-feature-encoder after verifying the model implementation."
            )
        freeze_method()

    train_rows = read_manifest(args.train_csv)
    validation_rows = read_manifest(args.validation_csv)
    test_rows = read_manifest(args.test_csv) if args.test_csv else None

    dataset_kwargs = {
        "processor": processor,
        "sample_rate": args.sample_rate,
        "min_duration_seconds": args.min_duration_seconds,
        "max_duration_seconds": args.max_duration_seconds,
        "validate_vocabulary": not args.skip_vocabulary_check,
    }
    train_dataset = AmharicAudioDataset(train_rows, **dataset_kwargs)
    validation_dataset = AmharicAudioDataset(validation_rows, **dataset_kwargs)
    test_dataset = (
        AmharicAudioDataset(test_rows, **dataset_kwargs) if test_rows else None
    )

    input_name = processor.model_input_names[0]
    collator = DataCollatorCTCWithPadding(processor, input_name)

    def preprocess_logits_for_metrics(logits: Any, labels: Any) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

    def compute_metrics(prediction: Any) -> dict[str, float]:
        prediction_ids = prediction.predictions
        label_ids = np.array(prediction.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        predicted_text = processor.batch_decode(prediction_ids)
        reference_text = processor.batch_decode(label_ids, group_tokens=False)
        return {
            "wer": float(jiwer.wer(reference_text, predicted_text)),
            "cer": float(jiwer.cer(reference_text, predicted_text)),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        bf16=not args.fp16,
        fp16=args.fp16,
        tf32=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        eval_accumulation_steps=8,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        save_safetensors=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint or None
    )
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    validation_metrics = trainer.evaluate(
        validation_dataset, metric_key_prefix="validation"
    )
    trainer.save_metrics("validation", validation_metrics)

    if test_dataset is not None:
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        trainer.save_metrics("test", test_metrics)

    summary = {
        "model": args.model_name,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_wer": trainer.state.best_metric,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples": len(test_dataset) if test_dataset else 0,
    }
    summary_path = Path(args.output_dir) / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

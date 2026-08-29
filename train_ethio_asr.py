#!/usr/bin/env python3
"""Stream all Leyu Amharic dialect datasets and fine-tune Ethio-ASR.

The Leyu repositories expose only a ``train`` split. This script performs a
metadata-only first pass, creates deterministic speaker-disjoint splits inside
each dialect, and streams audio during training without extracting the complete
dataset locally.

Example:
    python train_ethio_asr.py --output-dir outputs/chaka-asr

For gated/private data, run ``hf auth login`` or set ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fsspec
import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_HF_DATASETS = (
    "leyu-amharic/leyu-amharic-addis-ababa-dialect",
    "leyu-amharic/leyu-amharic-gonder-dialect",
    "leyu-amharic/leyu-amharic-wello-dialect",
    "leyu-amharic/leyu-amharic-gojjam-dialect",
    "leyu-amharic/leyu-amharic-shewa-dialect",
)
REQUIRED_COLUMNS = {"audio", "text", "dialect", "speaker_id", "gender"}
WHITESPACE_RE = re.compile(r"\s+")


def normalize_amharic_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return WHITESPACE_RE.sub(" ", text).strip()


def repo_slug(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1].replace("leyu-amharic-", "").replace(
        "-dialect", ""
    )


def stable_speaker_order(repo_id: str, speaker_id: str, seed: int) -> str:
    return hashlib.sha256(
        f"{seed}:{repo_id}:{speaker_id}".encode("utf-8")
    ).hexdigest()


def load_streaming_repository(
    repo_id: str, revision: str, token: str | None
) -> Any:
    dataset = load_dataset(
        repo_id,
        split="train",
        streaming=True,
        revision=revision,
        token=token,
    )
    columns = set(dataset.features or {})
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(
            f"{repo_id} is missing {sorted(missing)}; available={sorted(columns)}"
        )
    return dataset.cast_column("audio", Audio(decode=False))


def assign_speaker_splits(
    repo_id: str,
    speakers: Iterable[str],
    validation_percent: int,
    test_percent: int,
    seed: int,
) -> dict[str, str]:
    ordered = sorted(
        {str(speaker) for speaker in speakers},
        key=lambda speaker: stable_speaker_order(repo_id, speaker, seed),
    )
    if len(ordered) < 3:
        raise ValueError(f"{repo_id} needs at least 3 speakers")

    validation_count = max(1, round(len(ordered) * validation_percent / 100))
    test_count = max(1, round(len(ordered) * test_percent / 100))
    if validation_count + test_count >= len(ordered):
        raise ValueError(f"Split percentages leave no training speakers in {repo_id}")

    test_speakers = set(ordered[:test_count])
    validation_speakers = set(
        ordered[test_count : test_count + validation_count]
    )
    return {
        speaker: (
            "test"
            if speaker in test_speakers
            else "validation"
            if speaker in validation_speakers
            else "train"
        )
        for speaker in ordered
    }


def read_streamed_audio(audio_value: Any) -> tuple[torch.Tensor, int, int]:
    """Decode Audio(decode=False) bytes/path through SoundFile, without FFmpeg."""
    if isinstance(audio_value, dict) and audio_value.get("array") is not None:
        array = np.asarray(audio_value["array"], dtype=np.float32)
        sample_rate = int(audio_value["sampling_rate"])
        if array.ndim == 1:
            return torch.from_numpy(array), sample_rate, 1
        if array.ndim == 2:
            if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
                return torch.from_numpy(array).mean(0), sample_rate, array.shape[0]
            return torch.from_numpy(array).mean(1), sample_rate, array.shape[1]
        raise ValueError(f"Unsupported audio shape: {array.shape}")

    payload: bytes | None = None
    path: str | None = None
    if isinstance(audio_value, dict):
        if audio_value.get("bytes") is not None:
            payload = bytes(audio_value["bytes"])
        if audio_value.get("path"):
            path = str(audio_value["path"])
    elif isinstance(audio_value, (str, os.PathLike)):
        path = os.fspath(audio_value)
    else:
        raise TypeError(f"Unsupported audio value: {type(audio_value)!r}")

    if payload is not None:
        audio, sample_rate = sf.read(
            io.BytesIO(payload), dtype="float32", always_2d=True
        )
    elif path and Path(path).is_file():
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    elif path:
        with fsspec.open(path, "rb") as handle:
            audio, sample_rate = sf.read(
                io.BytesIO(handle.read()), dtype="float32", always_2d=True
            )
    else:
        raise ValueError("Streamed audio has neither bytes nor path")

    channels = int(audio.shape[1])
    return torch.from_numpy(audio).mean(1), int(sample_rate), channels


def inspect_repositories_and_build_plan(
    repo_ids: list[str],
    revision: str,
    token: str | None,
    validation_percent: int,
    test_percent: int,
    seed: int,
    audio_samples_per_repo: int,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Scan projected metadata and construct balanced speaker-level splits."""
    plan: dict[str, dict[str, str]] = {}
    report: dict[str, Any] = {
        "split_policy": {
            "unit": "speaker_id namespaced by repository",
            "train_percent": 100 - validation_percent - test_percent,
            "validation_percent": validation_percent,
            "test_percent": test_percent,
            "seed": seed,
        },
        "repositories": {},
    }

    for repo_id in repo_ids:
        print(f"Scanning metadata: {repo_id}", flush=True)
        dataset = load_streaming_repository(repo_id, revision, token)
        metadata = dataset.select_columns(["speaker_id", "gender", "dialect"])
        speaker_rows: Counter[str] = Counter()
        gender_rows: Counter[str] = Counter()
        dialect_rows: Counter[str] = Counter()
        speaker_gender_rows: dict[str, Counter[str]] = defaultdict(Counter)

        for example in metadata:
            speaker = str(example["speaker_id"])
            gender = str(example.get("gender") or "unknown")
            dialect = str(example.get("dialect") or repo_slug(repo_id))
            speaker_rows[speaker] += 1
            gender_rows[gender] += 1
            dialect_rows[dialect] += 1
            speaker_gender_rows[speaker][gender] += 1

        if not speaker_rows:
            raise ValueError(f"{repo_id} is empty")
        repo_plan = assign_speaker_splits(
            repo_id,
            speaker_rows,
            validation_percent,
            test_percent,
            seed,
        )
        plan[repo_id] = repo_plan

        split_rows: Counter[str] = Counter()
        split_speakers: Counter[str] = Counter()
        split_gender: dict[str, Counter[str]] = defaultdict(Counter)
        for speaker, rows in speaker_rows.items():
            split = repo_plan[speaker]
            split_rows[split] += rows
            split_speakers[split] += 1
            split_gender[split].update(speaker_gender_rows[speaker])

        audio_samples: list[dict[str, Any]] = []
        if audio_samples_per_repo:
            audio_dataset = load_streaming_repository(repo_id, revision, token)
            for index, example in enumerate(audio_dataset):
                if index >= audio_samples_per_repo:
                    break
                waveform, sample_rate, channels = read_streamed_audio(example["audio"])
                audio_samples.append(
                    {
                        "duration_seconds": round(waveform.numel() / sample_rate, 3),
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "text_characters": len(
                            normalize_amharic_text(example.get("text") or "")
                        ),
                    }
                )

        report["repositories"][repo_id] = {
            "rows": sum(speaker_rows.values()),
            "speakers": len(speaker_rows),
            "gender_rows": dict(sorted(gender_rows.items())),
            "dialect_rows": dict(sorted(dialect_rows.items())),
            "license": getattr(dataset.info, "license", None),
            "features": sorted(dataset.features.keys()),
            "split_rows": dict(sorted(split_rows.items())),
            "split_speakers": dict(sorted(split_speakers.items())),
            "split_gender_rows": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(split_gender.items())
            },
            "sample_audio": audio_samples,
        }

    repos = report["repositories"].values()
    report["totals"] = {
        "rows": sum(item["rows"] for item in repos),
        "speakers_namespaced_by_repository": sum(
            item["speakers"] for item in report["repositories"].values()
        ),
        "split_rows": {
            split: sum(
                item["split_rows"].get(split, 0)
                for item in report["repositories"].values()
            )
            for split in ("train", "validation", "test")
        },
        "split_speakers": {
            split: sum(
                item["split_speakers"].get(split, 0)
                for item in report["repositories"].values()
            )
            for split in ("train", "validation", "test")
        },
    }
    return plan, report


@dataclass
class SpeakerSplitFilter:
    speaker_plan: dict[str, str]
    split_name: str

    def __call__(self, example: dict[str, Any]) -> bool:
        speaker = str(example.get("speaker_id"))
        if speaker not in self.speaker_plan:
            raise KeyError(f"Speaker {speaker!r} was absent from metadata scan")
        return self.speaker_plan[speaker] == self.split_name


@dataclass
class StreamingAudioPreprocessor:
    processor: Any
    sample_rate: int
    min_duration_seconds: float
    max_duration_seconds: float
    validate_vocabulary: bool

    def __post_init__(self) -> None:
        self.input_name = self.processor.model_input_names[0]
        self.unknown_token_id = self.processor.tokenizer.unk_token_id

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        text = normalize_amharic_text(example.get("text") or "")
        if not text:
            raise ValueError("Encountered an empty transcript")
        waveform, source_rate, _ = read_streamed_audio(example["audio"])
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_rate, self.sample_rate
            )
        duration = waveform.numel() / self.sample_rate
        if not self.min_duration_seconds <= duration <= self.max_duration_seconds:
            raise ValueError(
                f"Audio duration {duration:.2f}s is outside "
                f"[{self.min_duration_seconds}, {self.max_duration_seconds}]"
            )

        labels = self.processor.tokenizer(text, add_special_tokens=False).input_ids
        if (
            self.validate_vocabulary
            and self.unknown_token_id is not None
            and self.unknown_token_id in labels
        ):
            raise ValueError(f"Tokenizer produced <unk> for: {text[:250]}")
        processed = self.processor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
        )
        return {self.input_name: processed[self.input_name][0], "labels": labels}


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    input_name: str

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = [{self.input_name: item[self.input_name]} for item in features]
        labels = [{"input_ids": item["labels"]} for item in features]
        batch = self.processor.pad(
            inputs,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        label_batch = self.processor.tokenizer.pad(
            labels, padding=True, return_tensors="pt"
        )
        batch["labels"] = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )
        return batch


def make_processed_split(
    repo_id: str,
    split_name: str,
    speaker_plan: dict[str, str],
    preprocessor: StreamingAudioPreprocessor,
    revision: str,
    token: str | None,
) -> Any:
    dataset = load_streaming_repository(repo_id, revision, token)
    source_columns = list(dataset.features.keys())
    dataset = dataset.filter(SpeakerSplitFilter(speaker_plan, split_name))
    return dataset.map(preprocessor, remove_columns=source_columns)


def build_streaming_datasets(
    repo_ids: list[str],
    plan: dict[str, dict[str, str]],
    preprocessor: StreamingAudioPreprocessor,
    revision: str,
    token: str | None,
    shuffle_buffer: int,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    train_parts, validation_parts = [], []
    test_by_dialect: dict[str, Any] = {}
    for repo_id in repo_ids:
        print(f"Building stream: {repo_id}", flush=True)
        train_parts.append(
            make_processed_split(
                repo_id, "train", plan[repo_id], preprocessor, revision, token
            )
        )
        validation_parts.append(
            make_processed_split(
                repo_id, "validation", plan[repo_id], preprocessor, revision, token
            )
        )
        test_by_dialect[repo_slug(repo_id)] = make_processed_split(
            repo_id, "test", plan[repo_id], preprocessor, revision, token
        )
    train = concatenate_datasets(train_parts).shuffle(
        seed=seed, buffer_size=shuffle_buffer
    )
    validation = concatenate_datasets(validation_parts)
    return train, validation, test_by_dialect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-dataset",
        action="append",
        dest="hf_datasets",
        help="Repeat to override the five default Leyu repositories.",
    )
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--hf-token-environment", default="HF_TOKEN")
    parser.add_argument("--output-dir", default="outputs/chaka-asr")
    parser.add_argument("--model-name", default="badrex/Ethio-ASR-amharic")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--min-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-duration-seconds", type=float, default=90.0)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--test-percent", type=int, default=10)
    parser.add_argument("--inspection-audio-samples", type=int, default=3)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=6_200)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--save-steps", type=int, default=1_000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--unfreeze-feature-encoder", action="store_true")
    parser.add_argument("--skip-vocabulary-check", action="store_true")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace, repo_ids: list[str]) -> None:
    if not repo_ids or len(repo_ids) != len(set(repo_ids)):
        raise ValueError("Dataset repository list must be non-empty and unique")
    if args.min_duration_seconds <= 0:
        raise ValueError("Minimum duration must be positive")
    if args.max_duration_seconds <= args.min_duration_seconds:
        raise ValueError("Maximum duration must exceed minimum duration")
    if args.validation_percent <= 0 or args.test_percent <= 0:
        raise ValueError("Validation and test percentages must be positive")
    if args.validation_percent + args.test_percent >= 100:
        raise ValueError("Validation and test percentages must sum below 100")
    if args.max_steps <= 0 and not args.inspect_only:
        raise ValueError("--max-steps must be positive for streaming training")
    if args.save_steps % args.eval_steps:
        raise ValueError("--save-steps must be a multiple of --eval-steps")
    if min(
        args.train_batch_size,
        args.eval_batch_size,
        args.gradient_accumulation_steps,
        args.shuffle_buffer,
        args.eval_steps,
        args.save_steps,
    ) <= 0:
        raise ValueError("Batch, step, and buffer values must be positive")


def decoded_prediction_texts(prediction: Any, processor: Any) -> tuple[list[str], list[str]]:
    prediction_ids = prediction.predictions
    if isinstance(prediction_ids, tuple):
        prediction_ids = prediction_ids[0]
    label_ids = np.array(prediction.label_ids, copy=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    hypotheses = [
        normalize_amharic_text(text)
        for text in processor.batch_decode(prediction_ids)
    ]
    references = [
        normalize_amharic_text(text)
        for text in processor.batch_decode(label_ids, group_tokens=False)
    ]
    return references, hypotheses


def save_predictions(
    prediction: Any, processor: Any, output_path: Path, dialect: str
) -> None:
    references, hypotheses = decoded_prediction_texts(prediction, processor)
    with output_path.open("w", encoding="utf-8") as handle:
        for reference, hypothesis in zip(references, hypotheses):
            handle.write(
                json.dumps(
                    {
                        "dialect": dialect,
                        "reference": reference,
                        "prediction": hypothesis,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    repo_ids = args.hf_datasets or list(DEFAULT_HF_DATASETS)
    validate_arguments(args, repo_ids)
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    token_value = os.environ.get(args.hf_token_environment)
    token = token_value.strip() if token_value and token_value.strip() else None

    plan, inspection = inspect_repositories_and_build_plan(
        repo_ids,
        args.dataset_revision,
        token,
        args.validation_percent,
        args.test_percent,
        args.seed,
        args.inspection_audio_samples,
    )
    (output_dir / "speaker_split_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_inspection.json").write_text(
        json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(inspection, ensure_ascii=False, indent=2), flush=True)
    if args.inspect_only:
        print("Inspection complete; training was not started.", flush=True)
        return

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCTC.from_pretrained(args.model_name)
    expected_rate = getattr(processor.feature_extractor, "sampling_rate", None)
    if expected_rate != args.sample_rate:
        raise ValueError(
            f"Model expects {expected_rate} Hz; configured {args.sample_rate} Hz"
        )
    if model.config.vocab_size != len(processor.tokenizer):
        raise ValueError("Model and tokenizer vocabulary sizes do not match")
    if processor.tokenizer.pad_token_id is None:
        raise ValueError("CTC tokenizer must define a pad token")

    model.config.ctc_loss_reduction = "mean"
    model.config.ctc_zero_infinity = True
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if not args.unfreeze_feature_encoder:
        freeze_method = getattr(model, "freeze_feature_encoder", None)
        if freeze_method is None:
            raise AttributeError("model.freeze_feature_encoder() is unavailable")
        freeze_method()

    preprocessor = StreamingAudioPreprocessor(
        processor,
        args.sample_rate,
        args.min_duration_seconds,
        args.max_duration_seconds,
        not args.skip_vocabulary_check,
    )
    train_dataset, validation_dataset, test_by_dialect = build_streaming_datasets(
        repo_ids,
        plan,
        preprocessor,
        args.dataset_revision,
        token,
        args.shuffle_buffer,
        args.seed,
    )
    collator = DataCollatorCTCWithPadding(
        processor, processor.model_input_names[0]
    )

    def preprocess_logits_for_metrics(logits: Any, labels: Any) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

    def compute_metrics(prediction: Any) -> dict[str, float]:
        references, hypotheses = decoded_prediction_texts(prediction, processor)
        return {
            "wer": float(jiwer.wer(references, hypotheses)),
            "cer": float(jiwer.cer(references, hypotheses)),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
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
        eval_accumulation_steps=4,
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
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience)],
    )

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint or None
    )
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    validation_output = trainer.predict(
        validation_dataset, metric_key_prefix="validation"
    )
    trainer.save_metrics("validation", validation_output.metrics)
    save_predictions(
        validation_output,
        processor,
        output_dir / "validation_predictions.jsonl",
        "all",
    )

    test_metrics: dict[str, dict[str, float]] = {}
    for dialect, test_dataset in test_by_dialect.items():
        print(f"Evaluating held-out {dialect} speakers", flush=True)
        metric_prefix = f"test_{dialect.replace('-', '_')}"
        output = trainer.predict(test_dataset, metric_key_prefix=metric_prefix)
        trainer.save_metrics(metric_prefix, output.metrics)
        save_predictions(
            output,
            processor,
            output_dir / f"test_{dialect}_predictions.jsonl",
            dialect,
        )
        test_metrics[dialect] = {
            key: float(value)
            for key, value in output.metrics.items()
            if isinstance(value, (int, float))
        }

    summary = {
        "base_model": args.model_name,
        "dataset_repositories": repo_ids,
        "streaming": True,
        "max_steps": args.max_steps,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_wer": trainer.state.best_metric,
        "split_totals": inspection["totals"],
        "test_metrics_by_dialect": test_metrics,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

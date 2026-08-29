#!/usr/bin/env python3
"""Download Leyu plus Waxal Amharic replay data for local ASR training.

Example:
    python -u download_leyu_data.py --data-dir data/leyu-amharic

Downloads are resumable. For gated/private repositories, run ``hf auth login``
or export ``HF_TOKEN`` before running this script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


DEFAULT_HF_DATASETS = (
    "leyu-amharic/leyu-amharic-addis-ababa-dialect",
    "leyu-amharic/leyu-amharic-gonder-dialect",
    "leyu-amharic/leyu-amharic-wello-dialect",
    "leyu-amharic/leyu-amharic-gojjam-dialect",
    "leyu-amharic/leyu-amharic-shewa-dialect",
)
WAXAL_REPOSITORY = "google/WaxalNLP"
WAXAL_CONFIG = "amh_asr"
WAXAL_SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Leyu Amharic datasets before training."
    )
    parser.add_argument(
        "--data-dir",
        default="data/leyu-amharic",
        help="Directory that will contain all downloaded repositories.",
    )
    parser.add_argument(
        "--hf-dataset",
        action="append",
        dest="hf_datasets",
        help="Repeat to override the five default Leyu dataset repositories.",
    )
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--hf-token-environment", default="HF_TOKEN")
    parser.add_argument(
        "--skip-waxal-replay",
        action="store_true",
        help="Download only Leyu and omit Waxal Amharic replay data.",
    )
    return parser.parse_args()


def repository_folder(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def download_waxal_amharic(
    data_dir: Path,
    revision: str,
    token: str | None,
) -> dict[str, object]:
    """Download only labeled Amharic ASR files, never the full Waxal repo."""
    print("Downloading WaxalNLP Amharic ASR replay data", flush=True)
    repo_files = list_repo_files(
        WAXAL_REPOSITORY,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    split_files: dict[str, list[str]] = {}
    selected_files: list[str] = []
    for split in WAXAL_SPLITS:
        prefix = f"data/ASR/amh/amh-{split}-"
        files = sorted(
            path
            for path in repo_files
            if path.endswith(".parquet")
            and (
                path.startswith(prefix)
                or (
                    ("/asr/amh/" in path.lower() or "amh_asr" in path.lower())
                    and (
                        f"amh-{split}-" in Path(path).name.lower()
                        or f"/{split}/" in path.lower()
                        or Path(path).name.lower().startswith(f"{split}-")
                    )
                )
            )
        )
        if not files:
            raise FileNotFoundError(
                f"No Waxal Amharic {split} Parquet files matched {prefix}*"
            )
        split_files[split] = files
        selected_files.extend(files)

    local_dir = data_dir / f"{repository_folder(WAXAL_REPOSITORY)}__{WAXAL_CONFIG}"
    snapshot_path = snapshot_download(
        repo_id=WAXAL_REPOSITORY,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=selected_files,
        token=token,
    )
    local_split_files = {
        split: [str(Path(path)) for path in files]
        for split, files in split_files.items()
    }
    size_bytes = sum(
        (local_dir / path).stat().st_size
        for path in selected_files
        if (local_dir / path).is_file()
    )
    print(
        f"Completed Waxal Amharic: {len(selected_files)} Parquet files, "
        f"{size_bytes / 1024**3:.2f} GiB\n",
        flush=True,
    )
    return {
        "repository": WAXAL_REPOSITORY,
        "config": WAXAL_CONFIG,
        "local_directory": str(local_dir.relative_to(data_dir)),
        "snapshot_path": str(Path(snapshot_path).resolve()),
        "split_files": local_split_files,
        "parquet_files": len(selected_files),
        "parquet_size_bytes": size_bytes,
    }


def main() -> None:
    args = parse_args()
    repo_ids = args.hf_datasets or list(DEFAULT_HF_DATASETS)
    if not repo_ids or len(repo_ids) != len(set(repo_ids)):
        raise ValueError("Dataset repository list must be non-empty and unique")

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    token_value = os.environ.get(args.hf_token_environment)
    token = token_value.strip() if token_value and token_value.strip() else None

    free_gib = shutil.disk_usage(data_dir).free / 1024**3
    print(f"Download directory: {data_dir}", flush=True)
    print(f"Free disk space: {free_gib:.1f} GiB", flush=True)
    print("Downloads are resumable if this process is interrupted.\n", flush=True)

    repositories: dict[str, dict[str, object]] = {}
    for position, repo_id in enumerate(repo_ids, start=1):
        local_dir = data_dir / repository_folder(repo_id)
        print(f"[{position}/{len(repo_ids)}] Downloading {repo_id}", flush=True)
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=args.dataset_revision,
            local_dir=local_dir,
            token=token,
        )
        parquet_files = sorted(local_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet data files were downloaded for {repo_id}"
            )
        size_bytes = sum(path.stat().st_size for path in parquet_files)
        repositories[repo_id] = {
            "local_directory": str(local_dir.relative_to(data_dir)),
            "snapshot_path": str(Path(snapshot_path).resolve()),
            "parquet_files": len(parquet_files),
            "parquet_size_bytes": size_bytes,
        }
        print(
            f"Completed {repo_id}: {len(parquet_files)} Parquet files, "
            f"{size_bytes / 1024**3:.2f} GiB\n",
            flush=True,
        )

    replay_datasets: dict[str, dict[str, object]] = {}
    if not args.skip_waxal_replay:
        replay_datasets["waxal_amh_asr"] = download_waxal_amharic(
            data_dir, args.dataset_revision, token
        )

    manifest = {
        "format_version": 2,
        "dataset_revision": args.dataset_revision,
        "data_directory": str(data_dir),
        "repositories": repositories,
        "replay_datasets": replay_datasets,
    }
    manifest_path = data_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"All datasets downloaded successfully: {data_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print("\nUse this directory for training with:", flush=True)
    print(
        f"python -u train_ethio_asr_local.py --data-dir {data_dir} \\",
        flush=True,
    )
    print("  --waxal-replay-ratio 0.30 \\", flush=True)
    print("  --output-dir outputs/chaka-asr", flush=True)


if __name__ == "__main__":
    main()

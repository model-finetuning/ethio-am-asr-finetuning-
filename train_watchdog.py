#!/usr/bin/env python3
"""Keep the full Chaka ASR training run alive and resume complete checkpoints."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"
TRAIN_SCRIPT = PROJECT_DIR / "train_ethio_asr_local.py"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "chaka-asr-waxal30"
LOG_PATH = PROJECT_DIR / "training-waxal30.log"
LOCK_PATH = PROJECT_DIR / ".training_watchdog.lock"
COMPLETE_MARKER = OUTPUT_DIR / "TRAINING_COMPLETE"
RETRY_SECONDS = 60

stop_requested = False
child: subprocess.Popen[str] | None = None


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def announce(message: str, log_handle: object | None = None) -> None:
    line = f"[{timestamp()}] WATCHDOG: {message}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if log_handle is not None:
        log_handle.write(line)  # type: ignore[attr-defined]
        log_handle.flush()  # type: ignore[attr-defined]


def checkpoint_step(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("checkpoint-"):
        return None
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return None


def is_complete_checkpoint(path: Path) -> bool:
    """Reject a checkpoint that may have been interrupted while being written."""
    step = checkpoint_step(path)
    required = (
        path / "trainer_state.json",
        path / "optimizer.pt",
        path / "scheduler.pt",
    )
    if step is None or any(not item.is_file() or item.stat().st_size == 0 for item in required):
        return False
    model_files = list(path.glob("*.safetensors")) + list(
        path.glob("*.safetensors.index.json")
    )
    if not model_files or any(item.stat().st_size == 0 for item in model_files):
        return False
    try:
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        return int(state.get("global_step", -1)) == step
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def latest_complete_checkpoint() -> Path | None:
    if not OUTPUT_DIR.is_dir():
        return None
    candidates = sorted(
        (item for item in OUTPUT_DIR.glob("checkpoint-*") if checkpoint_step(item) is not None),
        key=lambda item: checkpoint_step(item) or -1,
        reverse=True,
    )
    return next((item for item in candidates if is_complete_checkpoint(item)), None)


def existing_manual_trainers() -> list[int]:
    """Find matching trainers so the watchdog never starts a duplicate run."""
    matches: list[int] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (OSError, PermissionError):
            continue
        if TRAIN_SCRIPT.name in command and str(OUTPUT_DIR.relative_to(PROJECT_DIR)) in command:
            matches.append(int(entry.name))
    return matches


def training_command(checkpoint: Path | None) -> list[str]:
    command = [
        str(PYTHON),
        "-u",
        str(TRAIN_SCRIPT),
        "--data-dir",
        "data/leyu-amharic",
        "--output-dir",
        "outputs/chaka-asr-waxal30",
        "--waxal-replay-ratio",
        "0.30",
        "--max-steps",
        "8900",
        "--train-batch-size",
        "4",
        "--eval-batch-size",
        "2",
        "--gradient-accumulation-steps",
        "8",
        "--learning-rate",
        "1e-5",
        "--warmup-ratio",
        "0.10",
        "--eval-steps",
        "1000",
        "--save-steps",
        "1000",
        "--recovery-save-steps",
        "100",
        "--logging-steps",
        "25",
        "--dataloader-workers",
        "0",
    ]
    if checkpoint is not None:
        command.extend(("--resume-from-checkpoint", str(checkpoint)))
    return command


def handle_signal(signum: int, frame: object) -> None:
    del signum, frame
    global stop_requested
    stop_requested = True
    if child is not None and child.poll() is None:
        child.terminate()


def interruptible_wait(seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while not stop_requested and time.monotonic() < deadline:
        time.sleep(min(1.0, deadline - time.monotonic()))


def main() -> int:
    global child
    if not PYTHON.is_file():
        print(f"Missing virtual-environment Python: {PYTHON}", file=sys.stderr)
        return 2
    if not TRAIN_SCRIPT.is_file():
        print(f"Missing training script: {TRAIN_SCRIPT}", file=sys.stderr)
        return 2

    LOCK_PATH.touch(exist_ok=True)
    lock_handle = LOCK_PATH.open("r+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another Chaka training watchdog is already running.", file=sys.stderr)
        return 1

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log_handle:
        announce("started; full 8,900-step offline run is protected", log_handle)
        while not stop_requested:
            if COMPLETE_MARKER.is_file():
                announce("training is already marked complete", log_handle)
                return 0

            manual_pids = existing_manual_trainers()
            if manual_pids:
                announce(
                    f"matching manual trainer still running (PID {manual_pids}); waiting",
                    log_handle,
                )
                interruptible_wait(RETRY_SECONDS)
                continue

            checkpoint = latest_complete_checkpoint()
            if checkpoint is None:
                announce("no complete checkpoint found; starting at step 0", log_handle)
            else:
                announce(f"resuming from {checkpoint.name}", log_handle)

            child = subprocess.Popen(
                training_command(checkpoint),
                cwd=PROJECT_DIR,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert child.stdout is not None
            for line in child.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
            return_code = child.wait()
            child = None

            if stop_requested:
                announce("stopped by system signal", log_handle)
                return 0
            if return_code == 0:
                COMPLETE_MARKER.write_text(
                    f"Completed successfully at {timestamp()}\n", encoding="utf-8"
                )
                announce("training and final evaluation completed successfully", log_handle)
                return 0

            announce(
                f"trainer exited with code {return_code}; retrying in {RETRY_SECONDS}s",
                log_handle,
            )
            interruptible_wait(RETRY_SECONDS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

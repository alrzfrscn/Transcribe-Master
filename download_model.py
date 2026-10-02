"""Download the OpenVINO Whisper model from Hugging Face.

Skips download if the model directory already looks complete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "OpenVINO/whisper-large-v3-turbo-fp16-ov"
REQUIRED_FILES = (
    "openvino_encoder_model.xml",
    "openvino_decoder_model.xml",
    "tokenizer.json",
    "generation_config.json",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download OpenVINO Whisper model.")
    p.add_argument(
        "--model-dir",
        default=str(Path(__file__).resolve().parent / "whisper-large-v3-turbo-fp16-ov"),
        help="Destination directory.",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if present.")
    return p


def looks_complete(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in REQUIRED_FILES)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir).expanduser().resolve()

    if looks_complete(model_dir) and not args.force:
        print(f"[INFO] Model already present at: {model_dir}")
        print("[INFO] Use --force to re-download.")
        return 0

    print("Downloading OpenVINO Whisper model... Please wait.")
    print(f"Repo: {REPO_ID}")
    print(f"Dest: {model_dir}")
    try:
        snapshot_download(repo_id=REPO_ID, local_dir=str(model_dir))
    except Exception as e:  # noqa: BLE001 — user-facing error path
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        return 1

    if not looks_complete(model_dir):
        print("[WARNING] Download finished but some expected files are missing.", file=sys.stderr)
        return 1

    print(f"[SUCCESS] Download completed.\nFiles saved at: {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

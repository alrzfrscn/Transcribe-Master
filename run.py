"""Transcribe Master — Whisper OpenVINO speech-to-text.

Main entry point (unified, fixed version).

Features:
- 16 kHz mono extraction via FFmpeg with full error logging.
- Sliding-window chunking (default 30 s window / 2 s overlap) so words
  on chunk borders are not cut or duplicated.
- Overlap trimming + global merge/dedup of timestamps (time + word-level,
  up to 6 boundary words) and phantom-segment filtering.
- Splitting of over-long subtitles (max ~7 s / ~120 chars) for readable SRT.
- Correct SRT time formatting with millisecond carry handling.
- Device auto-fallback (GPU -> CPU); kernels compile in-memory on the fly.
- Language lock in auto-detect mode: detected once, then pinned.
- Silent-chunk skipping (peak-energy gate) to prevent hallucinations.
- Normalized language codes + interactive menu (fa, en, ar, tr, auto, other).
- Plain argparse CLI with English terminal logging.

Example:
    python run.py                # prompts for file path, then language menu
    python run.py lecture.mp4    # prompts for language (menu)
    python run.py lecture.mp4 --lang fa      # no prompt (automation)
    python run.py lecture.mp4 --lang ar --output-dir ./out
"""

from __future__ import annotations

import argparse
import math
import shutil
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import openvino_genai as ov_genai


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "whisper-large-v3-turbo-fp16-ov"

# Required sentinel file proving a complete OpenVINO model directory.
ENCODER_FILENAME = "openvino_encoder_model.xml"

SAMPLE_RATE = 16000
DEFAULT_CHUNK_SEC = 30.0
DEFAULT_OVERLAP_SEC = 2.0  # optimal for 30 s Whisper windows
DEFAULT_LANG = "en"  # used for non-interactive runs when --lang is omitted

# Silent-chunk gate: chunks quieter than this peak amplitude are pure
# dead air — skipping pipeline.generate() prevents phantom-loop hallucinations.
SILENCE_PEAK_THRESHOLD = 1e-3

# Interactive language menu: (menu key, label, value passed to normalize_language).
# The model itself supports ~100 Whisper language codes; the menu lists the
# most common ones and offers an "other" free-text choice for the rest.
LANGUAGE_MENU: Tuple[Tuple[str, str, str], ...] = (
    ("1", "English (en) [default]", "en"),
    ("2", "Persian (fa)", "fa"),
    ("3", "Arabic (ar)", "ar"),
    ("4", "Turkish (tr)", "tr"),
    ("5", "Auto-detect", "auto"),
    ("6", "Other (enter code, e.g. de, fr, ur, ...)", "other"),
)

# Readable-subtitle limits.
MAX_SUB_DURATION_SEC = 7.0
MAX_SUB_CHARS = 120
MIN_CHUNK_SEC = 0.3  # tails shorter than this are skipped (hallucination-prone)
MERGE_TOLERANCE_SEC = 0.20  # overlap tolerance when merging segments
MAX_BOUNDARY_WORDS = 6  # max repeated words stripped at chunk joints

# Phantom-hallucination filter: Whisper emits sparse confident-sounding
# fragments ("Thank you", single words) over long silent/music stretches.
# Segments this sparse over this long a span are discarded as hallucinations.
PHANTOM_MAX_WORDS = 2
PHANTOM_MIN_DURATION_SEC = 6.0

# Extra punctuation (beyond ASCII) stripped for word-level dedup comparisons.
_EXTRA_PUNCT = "«»‹›‘’“”…–—‐-؟،؛："
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + _EXTRA_PUNCT)

CandidateDevice = str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcribe audio/video to TXT + SRT with Whisper OpenVINO.",
    )
    p.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to video/audio file. If omitted, you will be prompted.",
    )
    p.add_argument(
        "--input",
        dest="input_opt",
        default=None,
        help="Same as positional input (alternative spelling).",
    )
    p.add_argument(
        "--lang",
        default=None,
        help="Speech language: 'fa', 'en', 'ar', 'tr', 'auto', or any Whisper "
             "code (e.g. de, fr, ur). Bare codes and <|fa|> style tokens both "
             "accepted. If omitted, an interactive menu is shown.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Execution device: 'auto', 'GPU', 'CPU', 'NPU'. "
             "'auto' tries GPU then falls back to CPU. Default: auto.",
    )
    p.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Path to OpenVINO model dir. Default: {DEFAULT_MODEL_DIR}",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for .txt/.srt outputs. Default: alongside input file.",
    )
    p.add_argument(
        "--chunk-duration",
        type=float,
        default=DEFAULT_CHUNK_SEC,
        help=f"Sliding-window length in seconds. Default: {DEFAULT_CHUNK_SEC}.",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP_SEC,
        help=f"Overlap between consecutive windows in seconds. Default: {DEFAULT_OVERLAP_SEC}.",
    )
    p.add_argument(
        "--no-srt-split",
        action="store_true",
        help="Disable splitting of long subtitle segments.",
    )
    return p


def normalize_language(raw: Optional[str]) -> Optional[str]:
    """Normalize user language input to a Whisper language token.

    Accepts bare codes ('fa'), token forms ('<|fa|>'), English names
    ('persian', 'arabic', ...), 'auto'/'detect' and None.
    Returns e.g. '<|fa|>' or None for auto-detect.
    """
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ("", "auto", "detect", "auto-detect", "none"):
        return None
    aliases = {
        "fa": "<|fa|>",
        "farsi": "<|fa|>",
        "persian": "<|fa|>",
        "<|fa|>": "<|fa|>",
        "en": "<|en|>",
        "english": "<|en|>",
        "<|en|>": "<|en|>",
        "ar": "<|ar|>",
        "arabic": "<|ar|>",
        "<|ar|>": "<|ar|>",
        "tr": "<|tr|>",
        "turkish": "<|tr|>",
        "turkce": "<|tr|>",
        "<|tr|>": "<|tr|>",
    }
    if s in aliases:
        return aliases[s]
    # Generic fallback: bare code -> <|xx|> token form,
    # which WhisperGenerationConfig documents as valid alongside 'xx'.
    # Whisper large-v3-turbo ships ~100 codes (de, fr, ur, ...).
    cleaned = s.strip("<|> ")
    if len(cleaned) <= 5 and cleaned.isalpha():
        return f"<|{cleaned}|>"
    return f"<|{cleaned}|>"


def prompt_language_menu() -> Optional[str]:
    """Interactive language menu. Returns a Whisper token or None (auto)."""
    print("\n" + "-" * 45)
    print(" Select Audio Language:")
    for _key, label, _value in LANGUAGE_MENU:
        print(f"   [{_key}] {label}")
    print("-" * 45)

    by_key = {key: value for key, _label, value in LANGUAGE_MENU}
    while True:
        try:
            choice = input("Enter choice (1-6) [Default: 1]: ").strip().lower()
        except EOFError:
            print("\n[INFO] Language set to: English (en) [default]")
            return "<|en|>"
        if choice in ("", "1"):
            print("[INFO] Language set to: English (en)")
            return "<|en|>"
        if choice in by_key and by_key[choice] != "other":
            value = by_key[choice]
            token = normalize_language(value)
            print(f"[INFO] Language set to: {value} ({token or 'auto-detect'})")
            return token
        if choice == "6" or choice in ("other", "o"):
            try:
                code = input("Enter language code (e.g. de, fr, ur, zh) or 'auto': ").strip()
            except EOFError:
                code = ""
            if not code:
                print("[INFO] Language set to: English (en) [default]")
                return "<|en|>"
            token = normalize_language(code)
            print(f"[INFO] Language set to: {code} ({token or 'auto-detect'})")
            return token
        # Allow typing a code/name directly at the menu prompt.
        if choice:
            token = normalize_language(choice)
            print(f"[INFO] Language set to: {choice} ({token or 'auto-detect'})")
            return token
        print("[ERROR] Invalid choice. Please enter 1-6 (or a language code).")


def resolve_language(cli_value: Optional[str]) -> Optional[str]:
    """Resolve --lang: explicit value wins, otherwise prompt (TTY) or default."""
    if cli_value is not None:
        return normalize_language(cli_value)
    try:
        interactive = sys.stdin.isatty()
    except Exception:
        interactive = False
    if interactive:
        return prompt_language_menu()
    return normalize_language(DEFAULT_LANG)


def resolve_input_file(positional: Optional[str], opt: Optional[str]) -> Path:
    raw = opt or positional
    if not raw:
        try:
            raw = input("Enter video or audio file path: ").strip()
        except EOFError:
            raw = ""
    if not raw:
        print("[ERROR] Input path cannot be empty.", file=sys.stderr)
        sys.exit(1)
    clean = raw.strip().strip('"').strip("'")
    target = Path(clean).expanduser().resolve()
    if not target.exists():
        print(f"[ERROR] File not found: {target}", file=sys.stderr)
        sys.exit(1)
    if not target.is_file():
        print(f"[ERROR] Path is a directory, not a media file: {target}", file=sys.stderr)
        sys.exit(1)
    return target


def resolve_device_candidates(requested: str) -> List[CandidateDevice]:
    r = (requested or "auto").strip().upper()
    if r == "AUTO":
        # NPU is tried in the middle: some Lunar Lake systems expose it,
        # but Whisper encoder may not support it — failure falls through.
        return ["GPU", "NPU", "CPU"]
    return [r]


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        print("[ERROR] FFmpeg executable not found in system PATH.", file=sys.stderr)
        print("[ERROR] Install it: winget install Gyan.FFmpeg  (https://ffmpeg.org/download.html)",
              file=sys.stderr)
        sys.exit(1)
    return exe


def extract_audio(input_file: Path, output_wav: Path) -> None:
    """Extract 16 kHz mono PCM via FFmpeg, with full stderr on failure."""
    check_ffmpeg()
    command = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    proc = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print(f"[ERROR] FFmpeg failed with exit code {proc.returncode}:", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print((proc.stderr or "").strip(), file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def format_srt_time(seconds: float) -> str:
    """Convert float seconds to 'HH:MM:SS,mmm'.

    Uses total-millisecond rounding so carries propagate correctly
    (e.g. 59.9999 -> 00:01:00,000 instead of the invalid 00:00:60,000).
    """
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _get_attr(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        try:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if val is not None:
                    return val
        except Exception:
            pass
    return None


def find_timing_items(result: Any) -> List[Dict[str, Any]]:
    """Extract {start, end, text} segments from a WhisperDecodedResults object."""
    if hasattr(result, "chunks") and result.chunks is not None:
        parsed = []
        for chunk in result.chunks:
            start = getattr(chunk, "start_ts", None)
            end = getattr(chunk, "end_ts", None)
            text = getattr(chunk, "text", "")
            if start is not None and end is not None:
                parsed.append({
                    "start": float(start),
                    "end": float(end),
                    "text": str(text).strip(),
                })
        if parsed:
            return parsed

    for container_name in ("segments", "chunks", "words", "word_timings", "timings", "results"):
        try:
            if not hasattr(result, container_name):
                continue
            container = getattr(result, container_name)
            if not container:
                continue
            parsed = []
            for item in list(container):
                text = _get_attr(item, ["text", "word", "content"])
                start = _get_attr(item, ["start", "start_time", "start_ts", "begin", "start_sec"])
                end = _get_attr(item, ["end", "end_time", "end_ts", "stop", "end_sec"])
                if text is None or start is None or end is None:
                    continue
                parsed.append({
                    "start": float(start),
                    "end": float(end),
                    "text": str(text).strip(),
                })
            if parsed:
                return parsed
        except Exception:
            pass
    return []


def trim_overlap_region(
    timings: List[Dict[str, Any]],
    chunk_index: int,
    overlap_sec: float,
) -> List[Dict[str, Any]]:
    """Drop/clamp the overlapped head of every chunk after the first.

    Window `i > 0` re-reads `overlap_sec` seconds already covered by window
    `i - 1`. Segments fully inside that head are duplicates -> dropped.
    Segments straddling the boundary are clamped so the word appears once.
    NOTE: timestamp clamping alone cannot catch words the model re-emits
    with shifted timestamps — see word-level dedup in merge_timings().
    """
    if chunk_index == 0 or overlap_sec <= 0 or not timings:
        return timings
    trimmed: List[Dict[str, Any]] = []
    for item in timings:
        s, e = float(item["start"]), float(item["end"])
        if e <= overlap_sec:
            continue
        if s < overlap_sec:
            s = overlap_sec
        if s >= e:
            continue
        trimmed.append({"start": s, "end": e, "text": item["text"]})
    return trimmed


def _normalize_word(word: str) -> str:
    """Lowercase a word and strip punctuation for boundary comparisons."""
    return word.translate(_PUNCT_TABLE).lower().strip()


def _strip_leading_overlap(prev_text: str, cur_text: str, max_words: int = MAX_BOUNDARY_WORDS) -> str:
    """Strip a repeated boundary phrase from the start of `cur_text`.

    If any trailing phrase of `prev_text` (up to `max_words` words) appears
    at the beginning of `cur_text` — compared case- and
    punctuation-insensitively — it is stripped completely. The longest match
    wins, so multi-word clauses (e.g. "be ruled out Why", "cells and thus
    by taking", "taper off down") are removed in one pass, even when the
    previous segment continues past the repeated phrase (e.g. "... down
    slowly"). Returns `cur_text` unchanged when there is no match, or ""
    when fully duplicated.
    """
    prev_words: List[str] = str(prev_text).split()
    cur_words: List[str] = str(cur_text).split()
    if not prev_words or not cur_words:
        return cur_text
    norm_prev: List[str] = [_normalize_word(w) for w in prev_words]
    norm_cur: List[str] = [_normalize_word(w) for w in cur_words]
    # Stage 1: suffix(N) of prev == prefix(N) of cur, longest first.
    limit: int = min(max_words, len(prev_words), len(cur_words))
    for n in range(limit, 0, -1):
        if all(p and p == c for p, c in zip(norm_prev[-n:], norm_cur[:n])):
            return " ".join(cur_words[n:]).strip()
    # Stage 2: the repeated phrase sits inside prev's tail but prev continues
    # past it. Search prev's trailing window for the longest leading run of
    # cur (minimum 2 words — single common words like "the" must not trigger
    # stripping on their own).
    tail: List[str] = norm_prev[-2 * max_words:]
    max_k: int = min(max_words, len(cur_words))
    for k in range(max_k, 1, -1):
        head: List[str] = norm_cur[:k]
        if not all(head):
            continue
        for s in range(len(tail) - k + 1):
            if tail[s:s + k] == head:
                return " ".join(cur_words[k:]).strip()
    return cur_text


def is_phantom_segment(start: float, end: float, text: str) -> bool:
    """Return True for hallucination-like segments.

    Whisper tends to emit sparse fragments ("Thank you", single words) with
    confident timestamps stretched across long silent/music spans. A segment
    with very few words over an abnormally long duration is a phantom.
    """
    words: List[str] = str(text).split()
    return len(words) <= PHANTOM_MAX_WORDS and (float(end) - float(start)) > PHANTOM_MIN_DURATION_SEC


def filter_phantom_segments(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Split out phantom segments. Returns (kept, dropped_count)."""
    kept: List[Dict[str, Any]] = []
    dropped: int = 0
    for item in items:
        if is_phantom_segment(float(item["start"]), float(item["end"]), str(item.get("text", ""))):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def merge_timings(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort by start time and resolve overlaps/duplicates globally.

    Three-stage dedup:
    1. Time-based: contained segments are dropped, partial overlaps clamped.
    2. Word-based: any repeated trailing phrase found in the trailing
       context of up to 3 previously merged segments (up to
       MAX_BOUNDARY_WORDS words, case/punctuation-insensitive) re-emitted
       at the start of the next one is stripped completely. The wider
       context matters because the immediate predecessor may be a 1-2 word
       fragment, pushing the repeated clause's head further back.
    3. Phantom-based: sparse fragments (<=2 words over >6 s) typical of
       Whisper hallucinations on silence/music are discarded.
    """
    cleaned: List[Dict[str, Any]] = []
    for i in items:
        text: str = str(i.get("text", "")).strip()
        if not text:
            continue
        start: float = float(i["start"])
        end: float = float(i["end"])
        if end <= start:
            continue
        if is_phantom_segment(start, end, text):
            continue
        cleaned.append({"start": start, "end": end, "text": text})
    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    merged: List[Dict[str, Any]] = []
    for cur in cleaned:
        if not merged:
            merged.append(cur)
            continue
        prev = merged[-1]
        if cur["start"] < prev["end"] - MERGE_TOLERANCE_SEC:
            # Fully contained duplicate -> skip.
            if cur["end"] <= prev["end"] + MERGE_TOLERANCE_SEC:
                continue
            # Partial overlap -> clamp to avoid double-speaking the same words.
            cur["start"] = prev["end"]
            if cur["start"] >= cur["end"]:
                continue
        # Word-level joint dedup (applies with or without time overlap,
        # since the model may re-emit boundary words with shifted stamps).
        # Context spans up to 3 prior segments: the immediate predecessor may
        # be a 1-2 word fragment (e.g. "Why?"), leaving the repeated clause's
        # head (e.g. "be ruled out.") back in merged[-2] or merged[-3].
        prev_context: str = " ".join(s["text"] for s in merged[-3:])
        deduped: str = _strip_leading_overlap(prev_context, cur["text"])
        if not deduped:
            continue
        cur["text"] = deduped
        merged.append(cur)
    return merged


def split_long_segments(
    items: List[Dict[str, Any]],
    max_dur: float = MAX_SUB_DURATION_SEC,
    max_chars: int = MAX_SUB_CHARS,
) -> List[Dict[str, Any]]:
    """Split segments that are too long (time) or too dense (chars).

    Time is distributed proportionally to character length so each piece
    stays roughly in sync with the audio.
    """
    out: List[Dict[str, Any]] = []
    for item in items:
        start, end, text = float(item["start"]), float(item["end"]), str(item["text"]).strip()
        dur = end - start
        if dur <= 0 or not text:
            continue
        if dur <= max_dur and len(text) <= max_chars:
            out.append({"start": start, "end": end, "text": text})
            continue
        words = text.split()
        if len(words) <= 1:
            # Single token but over-long: just clamp the display duration.
            out.append({"start": start, "end": min(end, start + max_dur), "text": text})
            continue
        n = max(math.ceil(dur / max_dur), math.ceil(len(text) / max_chars))
        n = min(n, len(words))
        chunk_size = math.ceil(len(words) / n)
        groups: List[List[str]] = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        total_chars = sum(len(w) for w in words) or 1
        cursor = start
        for gi, group in enumerate(groups):
            g_text = " ".join(group)
            # +len(group)-1 accounts for spaces.
            g_chars = sum(len(w) for w in group) + max(0, len(group) - 1)
            frac = g_chars / (total_chars + (len(words) - 1))
            if gi == len(groups) - 1:
                g_end = end
            else:
                g_end = cursor + dur * frac
            # Guard against rounding drift producing degenerate pieces.
            if g_end <= cursor:
                g_end = min(end, cursor + 1.0)
            out.append({"start": cursor, "end": g_end, "text": g_text})
            cursor = g_end
    return out


def write_srt(items: List[Dict[str, Any]], output_file: Path) -> int:
    """Write standard SRT with a sequential counter. Returns segment count."""
    counter = 0
    with output_file.open("w", encoding="utf-8") as f:
        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            counter += 1
            f.write(f"{counter}\n")
            f.write(f"{format_srt_time(item['start'])} --> {format_srt_time(item['end'])}\n")
            f.write(f"{text}\n\n")
    return counter


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def validate_model_dir(model_dir: Path) -> Optional[str]:
    """Check the OpenVINO model directory. Returns an error message or None."""
    if not model_dir.exists():
        return f"Model directory not found: {model_dir}"
    if not (model_dir / ENCODER_FILENAME).exists():
        return (
            f"Incomplete model directory: {model_dir} "
            f"(missing {ENCODER_FILENAME})"
        )
    return None


def init_pipeline(model_dir: Path, device_candidates: List[CandidateDevice]) -> Tuple[Any, str]:
    """Initialize WhisperPipeline on the first working device.

    Kernels compile in-memory on the fly (no on-disk cache).
    """
    last_err: Optional[Exception] = None
    for device in device_candidates:
        try:
            pipeline = ov_genai.WhisperPipeline(str(model_dir), device)
            return pipeline, device
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 — must report any backend failure
            print(f"[WARNING] Cannot init model on {device}: {e}")
            last_err = e
    print("[ERROR] Failed to initialize model on any device.", file=sys.stderr)
    if last_err:
        print(f"[ERROR] Last error: {last_err}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

def _coerce_detected_value(value: Any) -> Optional[str]:
    """Coerce a raw detected-language value to a <|xx|> token, or None."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in ("auto", "unknown"):
            return None
        return normalize_language(s)
    return None


def extract_detected_language(result: Any) -> Optional[str]:
    """Best-effort extraction of Whisper's detected language from a result.

    Probes common attribute names on WhisperDecodedResults (and its metadata
    dict, when present). Returns a normalized <|xx|> token or None when the
    backend does not expose the decision.
    """
    for attr in ("language", "detected_language", "lang", "language_token",
                 "language_id", "lang_id", "detected_lang"):
        try:
            val = getattr(result, attr, None)
        except Exception:
            continue
        token = _coerce_detected_value(val)
        if token:
            return token
    try:
        meta = getattr(result, "metadata", None)
    except Exception:
        meta = None
    if isinstance(meta, dict):
        for key in ("language", "detected_language", "lang"):
            token = _coerce_detected_value(meta.get(key))
            if token:
                return token
    return None


def is_silent_chunk(audio: np.ndarray, threshold: float = SILENCE_PEAK_THRESHOLD) -> bool:
    """Return True when a chunk is pure silence/flatline (peak-energy gate)."""
    try:
        peak: float = float(np.max(np.abs(audio)))
    except Exception:
        return False
    return peak < threshold


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def transcribe_wav(
    pipeline: Any,
    wav_file: Path,
    gen_kwargs: Dict[str, Any],
    chunk_sec: float,
    overlap_sec: float,
) -> List[Dict[str, Any]]:
    """Stream the wav through sliding windows and return merged segments."""
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    stride_samples = int((chunk_sec - overlap_sec) * SAMPLE_RATE)
    if chunk_samples <= 0 or stride_samples <= 0:
        print("[ERROR] Invalid chunk/overlap combination.", file=sys.stderr)
        sys.exit(1)

    all_items: List[Dict[str, Any]] = []
    # Language lock: in auto-detect mode Whisper would otherwise re-detect
    # per 30 s chunk and can drift mid-lecture on pauses/English terms.
    active_kwargs: Dict[str, Any] = dict(gen_kwargs)
    auto_detect: bool = "language" not in active_kwargs
    locked_language: Optional[str] = None

    with sf.SoundFile(str(wav_file)) as sf_file:
        if sf_file.samplerate != SAMPLE_RATE:
            print(f"[ERROR] Unexpected sample rate: {sf_file.samplerate} (expected {SAMPLE_RATE}).",
                  file=sys.stderr)
            sys.exit(1)
        total_frames = sf_file.frames
        if total_frames <= 0:
            print("[ERROR] Extracted audio is empty.", file=sys.stderr)
            sys.exit(1)
        total_sec = total_frames / float(sf_file.samplerate)
        total_chunks = max(1, math.ceil(total_frames / float(stride_samples)))

        chunk_index = 0
        start_sample = 0
        while start_sample < total_frames:
            sf_file.seek(start_sample)
            audio = sf_file.read(frames=chunk_samples, dtype="float32")
            if audio is None or len(audio) == 0:
                break
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = np.asarray(audio, dtype=np.float32)

            chunk_dur = len(audio) / float(sf_file.samplerate)
            offset_sec = start_sample / float(sf_file.samplerate)

            if len(audio) < int(MIN_CHUNK_SEC * SAMPLE_RATE):
                # Tiny tail: almost always silence/hallucination.
                break

            if is_silent_chunk(audio):
                peak_val: float = float(np.max(np.abs(audio))) if len(audio) else 0.0
                print(
                    f"[INFO] Chunk {chunk_index + 1}/{total_chunks} "
                    f"| offset {format_srt_time(offset_sec)} — skipped (silence, peak={peak_val:.2e})"
                )
                chunk_index += 1
                start_sample += stride_samples
                continue

            pct = min(100.0, (offset_sec / total_sec) * 100.0) if total_sec > 0 else 100.0
            print(
                f"[INFO] Chunk {chunk_index + 1}/{total_chunks} "
                f"({pct:5.1f}%) | offset {format_srt_time(offset_sec)}",
                end="\r",
                flush=True,
            )

            result = pipeline.generate(audio, **active_kwargs)
            chunk_text = str(result).strip()
            timings = find_timing_items(result)

            # Lock the first auto-detected language so the tokenizer stays
            # stable for the rest of the recording.
            if auto_detect and locked_language is None and (chunk_text or timings):
                detected = extract_detected_language(result)
                if detected:
                    locked_language = detected
                    active_kwargs["language"] = locked_language
                    print(f"\n[INFO] Auto-detected language locked: {locked_language}")

            if timings:
                timings = trim_overlap_region(timings, chunk_index, overlap_sec)
                for item in timings:
                    all_items.append({
                        "start": offset_sec + float(item["start"]),
                        "end": offset_sec + float(item["end"]),
                        "text": str(item["text"]).strip(),
                    })
            elif chunk_text:
                # Fallback: model gave text but no timestamps — cover the
                # window so nothing is lost; it will be split later.
                all_items.append({
                    "start": offset_sec,
                    "end": offset_sec + chunk_dur,
                    "text": chunk_text,
                })

            chunk_index += 1
            start_sample += stride_samples

    print()  # newline after progress line
    return merge_timings(all_items)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print("=" * 60)
    print("    Whisper OpenVINO - Speech to Text & Subtitles")
    print("=" * 60)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.chunk_duration <= 0 or args.overlap < 0 or args.overlap >= args.chunk_duration:
        print("[ERROR] Require: 0 <= --overlap < --chunk-duration.", file=sys.stderr)
        return 1

    input_file = resolve_input_file(args.input, args.input_opt)
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_error = validate_model_dir(model_dir)
    if model_error:
        print(f"[ERROR] {model_error}", file=sys.stderr)
        print("[HINT] Run: python download_model.py", file=sys.stderr)
        return 1

    language = resolve_language(args.lang)
    print(f"[INFO] Language: {'auto-detect' if language is None else language}")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        output_txt = out_dir / (input_file.stem + ".txt")
        output_srt = out_dir / (input_file.stem + ".srt")
    else:
        output_txt = input_file.with_suffix(".txt")
        output_srt = input_file.with_suffix(".srt")
    print(f"[INFO] Output directory: {output_txt.parent}")

    device_candidates = resolve_device_candidates(args.device)

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_file = Path(temp_dir) / "audio_extracted.wav"

        print("\n[1/3] Extracting 16 kHz mono audio via FFmpeg...")
        extract_audio(input_file, wav_file)

        print(f"\n[2/3] Loading model (try: {', '.join(device_candidates)})...")
        pipeline, used_device = init_pipeline(model_dir, device_candidates)
        print(f"[INFO] Model ready on device: {used_device}")

        gen_kwargs: Dict[str, Any] = {"task": "transcribe", "return_timestamps": True}
        if language:
            gen_kwargs["language"] = language

        print(f"\n[3/3] Transcribing (window={args.chunk_duration:g}s, "
              f"overlap={args.overlap:g}s)...")
        try:
            segments = transcribe_wav(
                pipeline, wav_file, gen_kwargs,
                chunk_sec=args.chunk_duration,
                overlap_sec=args.overlap,
            )
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 — user-facing error path
            print(f"\n[ERROR] Transcription failed: {e}", file=sys.stderr)
            return 1

        if not args.no_srt_split:
            segments = split_long_segments(segments)

        # Post-split phantom sweep: splitting a long sparse segment can emit
        # short-text pieces over long spans — drop those hallucinations too.
        segments, phantom_dropped = filter_phantom_segments(segments)
        if phantom_dropped:
            print(f"[INFO] Discarded {phantom_dropped} phantom segment(s) (<=2 words over >6s).")

        print("\nWriting output files...")
        full_text = "\n".join(s["text"] for s in segments)
        output_txt.write_text(full_text, encoding="utf-8")
        print(f"[SUCCESS] TXT saved: {output_txt.name} ({len(segments)} lines)")

        if segments:
            count = write_srt(segments, output_srt)
            print(f"[SUCCESS] SRT saved: {output_srt.name} (segments: {count})")
        else:
            print("[WARNING] No speech segments produced. Only empty TXT was written.")

    print("\n" + "=" * 60)
    print("[DONE] All tasks completed successfully.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

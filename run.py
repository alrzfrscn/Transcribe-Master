import sys
from pathlib import Path
import subprocess
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import openvino_genai as ov_genai


# ============================================================
# تنظیمات پایه (BASE SETTINGS)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# مسیر پوشه مدل
MODEL_DIR = BASE_DIR / "whisper-large-v3-turbo-fp16-ov"

# سخت‌افزار اجرا (GPU برای Intel Arc 140V)
DEVICE = "GPU"


# ============================================================
# دریافت فایل ورودی (GET INPUT FILE)
# ============================================================

def get_input_file() -> Path:
    """دریافت مسیر فایل از آرگومان یا پرامپت خط فرمان"""
    if len(sys.argv) > 1:
        raw_path = sys.argv[1]
    else:
        raw_path = input("Enter video or audio file path: ").strip()

    clean_path = raw_path.strip('"').strip("'")
    return Path(clean_path).resolve()


# ============================================================
# منوی انتخاب زبان (SELECT LANGUAGE)
# ============================================================

def select_language() -> Optional[str]:
    """منوی تعاملی برای انتخاب زبان گفتار"""
    print("\n" + "-" * 40)
    print(" Select Audio Language:")
    print("   [1] English (en)")
    print("   [2] Persian (fa)  <-- Recommended for Persian/Medical")
    print("   [3] Auto-detect")
    print("-" * 40)

    while True:
        choice = input("Enter your choice (1/2/3) [Default: 2]: ").strip()

        if choice == "" or choice == "2":
            print("[INFO] Language set to: Persian (fa)")
            return "<|fa|>"
        elif choice == "1":
            print("[INFO] Language set to: English (en)")
            return "<|en|>"
        elif choice == "3":
            print("[INFO] Language set to: Auto-detect")
            return None
        else:
            print("[ERROR] Invalid input. Please enter 1, 2, or 3.")


# ============================================================
# استخراج صدا با FFmpeg (EXTRACT AUDIO)
# ============================================================

def extract_audio(input_file: Path, output_wav: Path):
    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ============================================================
# بارگذاری صوت (LOAD AUDIO)
# ============================================================

def load_audio(wav_file: Path) -> np.ndarray:
    audio, sample_rate = sf.read(wav_file, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != 16000:
        raise RuntimeError(f"Unexpected sample rate: {sample_rate}")

    return audio.astype(np.float32)


# ============================================================
# فرمت‌دهی زمان زیرنویس (FORMAT SRT TIME)
# ============================================================

def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    seconds -= hours * 3600

    minutes = int(seconds // 60)
    seconds -= minutes * 60

    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))

    if milliseconds >= 1000:
        whole_seconds += 1
        milliseconds -= 1000

    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


# ============================================================
# استخراج بخش‌های زمان‌بندی (EXTRACT TIMINGS)
# ============================================================

def get_attr(obj, names):
    for name in names:
        try:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if val is not None:
                    return val
        except Exception:
            pass
    return None


def find_timing_items(result):
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
                    "text": str(text).strip()
                })
        if parsed:
            return parsed

    containers = ["segments", "chunks", "words", "word_timings", "timings", "results"]
    for container_name in containers:
        try:
            if not hasattr(result, container_name):
                continue
            container = getattr(result, container_name)
            if not container:
                continue

            parsed = []
            for item in list(container):
                text = get_attr(item, ["text", "word", "content"])
                start = get_attr(item, ["start", "start_time", "start_ts", "begin", "start_sec"])
                end = get_attr(item, ["end", "end_time", "end_ts", "stop", "end_sec"])

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


# ============================================================
# نوشتن فایل SRT (WRITE SRT)
# ============================================================

def write_srt(items, output_file: Path):
    with output_file.open("w", encoding="utf-8") as f:
        for index, item in enumerate(items, start=1):
            if not item["text"]:
                continue
            f.write(f"{index}\n")
            f.write(f"{format_srt_time(item['start'])} --> {format_srt_time(item['end'])}\n")
            f.write(f"{item['text']}\n\n")


# ============================================================
# بدنه اصلی (MAIN)
# ============================================================

def main():
    print("=" * 60)
    print("    Whisper OpenVINO - Speech to Text & Subtitles")
    print("=" * 60)

    # ۱. دریافت فایل ورودی
    input_file = get_input_file()

    if not input_file.exists():
        print(f"\n[ERROR] Input file not found:\n{input_file}")
        return

    if not MODEL_DIR.exists():
        print(f"\n[ERROR] Model directory not found:\n{MODEL_DIR}")
        return

    # ۲. انتخاب زبان
    selected_language = select_language()

    # ۳. مشخص کردن مسیر ذخیره فایل‌ها در کنار فایل اصلی
    output_txt = input_file.with_suffix(".txt")
    output_srt = input_file.with_suffix(".srt")

    print(f"\n[INFO] Target directory: {input_file.parent}")

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_file = Path(temp_dir) / "extracted_audio.wav"

        # مرحله اول: استخراج صوت
        print("\n[1/4] Extracting audio stream via FFmpeg...")
        extract_audio(input_file, wav_file)

        # مرحله دوم: لود صوت
        print("[2/4] Loading audio data...")
        audio = load_audio(wav_file)

        # مرحله سوم: لود مدل و پردازش
        print(f"[3/4] Initializing model on device ({DEVICE})...")
        pipeline = ov_genai.WhisperPipeline(str(MODEL_DIR), DEVICE)

        gen_kwargs = {
            "task": "transcribe",
            "return_timestamps": True,
        }
        if selected_language:
            gen_kwargs["language"] = selected_language

        print("[INFO] Transcribing audio with OpenVINO...")
        result = pipeline.generate(audio, **gen_kwargs)

        # مرحله چهارم: ذخیره فایل‌ها دقیقاً در کنار فایل ورودی
        print("\n[4/4] Writing output files alongside source...")

        # ذخیره متن
        full_text = str(result).strip()
        output_txt.write_text(full_text, encoding="utf-8")
        print(f"[SUCCESS] TXT saved: {output_txt.name}")

        # ذخیره زیرنویس
        timing_items = find_timing_items(result)
        if timing_items:
            write_srt(timing_items, output_srt)
            print(f"[SUCCESS] SRT saved: {output_srt.name} (Segments: {len(timing_items)})")
        else:
            print("[WARNING] No subtitle timestamps were detected. Only TXT was generated.")

    print("\n" + "=" * 60)
    print(f"[DONE] Files saved to: {input_file.parent}")
    print("=" * 60)


if __name__ == "__main__":
    main()
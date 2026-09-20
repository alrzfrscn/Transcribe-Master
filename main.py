import os
import subprocess
import tempfile
from pathlib import Path
from openvino_genai import WhisperPipeline

# Path to the exported OpenVINO Whisper model directory
MODEL_DIR = Path(__file__).parent / "whisper-large-v3-turbo-fp16-ov"

# Target device for execution (GPU for Intel Arc graphics)
DEVICE = "GPU"

# Supported video extensions for audio extraction
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm"}


def extract_audio(video_path: str, output_audio_path: str) -> None:
    """Extracts audio track from a video file using FFmpeg (16kHz mono PCM)."""
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_path
    ]
    
    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed or not found in system PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr.decode('utf-8', errors='ignore')}")


def format_timestamp(seconds: float) -> str:
    """Converts seconds to standard SRT timestamp format (HH:MM:SS,mmm)."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    
    if millis >= 1000:
        secs += 1
        millis -= 1000
        if secs >= 60:
            mins += 1
            secs -= 60
            if mins >= 60:
                hrs += 1
                mins -= 60

    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def transcribe(input_file: str, language: str = "fa", output_format: str = "srt") -> None:
    """Transcribes an audio or video file and saves the output as SRT or TXT."""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ Error: Input file '{input_file}' does not exist.")
        return

    if not MODEL_DIR.exists():
        print(f"❌ Error: Model directory '{MODEL_DIR}' not found. Please run the model downloader first.")
        return

    print(f"⚡ Loading Whisper model on {DEVICE}...")
    pipe = WhisperPipeline(str(MODEL_DIR), DEVICE)

    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS
    temp_audio_file = None

    try:
        if is_video:
            print("🎬 Extracting audio from video...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_audio_file = tmp.name
            
            extract_audio(str(input_path), temp_audio_file)
            audio_source = temp_audio_file
        else:
            audio_source = str(input_path)

        print("🎙️ Transcribing audio...")
        config = pipe.get_generation_config()
        config.language = language
        config.task = "transcribe"
        
        if output_format == "srt":
            config.return_timestamps = True

        result = pipe.generate(audio_source, config)

        output_file = input_path.with_suffix(f".{output_format}")

        if output_format == "srt":
            with open(output_file, "w", encoding="utf-8") as f:
                chunks = getattr(result, "chunks", [])
                if not chunks and hasattr(result, "texts"):
                    f.write(f"1\n00:00:00,000 --> 00:00:10,000\n{result.texts[0].strip()}\n\n")
                else:
                    for idx, chunk in enumerate(chunks, 1):
                        start = format_timestamp(chunk.start_ts)
                        end = format_timestamp(chunk.end_ts)
                        text = chunk.text.strip()
                        if text:
                            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                text_content = result.texts[0] if hasattr(result, "texts") else str(result)
                f.write(text_content.strip())

        print(f"🎉 Process completed successfully! Saved to: {output_file}")

    finally:
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)


if __name__ == "__main__":
    # Specify your input video or audio file here
    INPUT_FILE = "test.mp4"
    
    # Run transcription
    transcribe(INPUT_FILE, language="fa", output_format="srt")
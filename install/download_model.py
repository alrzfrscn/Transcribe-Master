from pathlib import Path
from huggingface_hub import snapshot_download

# Define model save location relative to the script directory
project_dir = Path(__file__).resolve().parent
model_dir = project_dir / "whisper-large-v3-turbo-fp16-ov"

print("⏳ Downloading OpenVINO Whisper model... Please wait.")

# Download full model repository from Hugging Face
snapshot_download(
    repo_id="OpenVINO/whisper-large-v3-turbo-fp16-ov",
    local_dir=str(model_dir)
)

print(f"✅ Download completed successfully!\nFiles saved at: {model_dir}")
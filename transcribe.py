import argparse
import os
import sys

def install_instructions():
    print("If you haven't installed Whisper, run the following commands:")
    print("  pip install -U openai-whisper")
    print("You also need ffmpeg installed and available in your PATH.")
    print("  For Windows, download from https://ffmpeg.org/download.html and add the bin folder to your system PATH.")

def transcribe(audio_path: str, model_name: str = "base"):
    try:
        import whisper
    except ImportError:
        print("Whisper library not found in the current environment. Please ensure it is installed in your active virtual environment.")
        sys.exit(1)
    # Load model (will download if not present)
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"Transcribing '{audio_path}'...")
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    output_path = os.path.splitext(audio_path)[0] + ".txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Transcription saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio/video files to text using OpenAI Whisper.")
    parser.add_argument("input", help="Path to the audio or video file (e.g., .mp3, .mp4)")
    parser.add_argument("-m", "--model", default="base", help="Whisper model size (tiny, base, small, medium, large). Default is 'base'.")
    args = parser.parse_args()
    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    # Show install instructions if needed
    install_instructions()
    transcribe(input_path, args.model)

import os
import requests
import logging
import whisper
from utils import reduce_noise, normalize_audio, chunk_audio
from diarize import diarize_audio
from analyze import summarize_performance

# Setup logging
logging.basicConfig(filename="logs/transcription.log", level=logging.INFO)

# Load Whisper model (start with "tiny", upgrade to "medium"/"large" later)
model = whisper.load_model("base")  # Change to "medium" or "large" for better accuracy

# Folder setup
AUDIO_DIR = "audio"
CHUNK_DIR = "chunks"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

def download_audio_from_s3(url, local_path):
    """Download audio file from public S3 URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False

def transcribe_chunks(chunk_paths, translate=False):
    """Transcribe each chunk and combine results"""
    full_text = ""
    for path in chunk_paths:
        result = model.transcribe(path, task="translate" if translate else "transcribe", language="ta")
        full_text += result["text"].strip() + "\n"
    return full_text.strip()

def process_pipeline(s3_urls):
    """Main pipeline for audio processing and analysis"""
    for idx, url in enumerate(s3_urls):
        try:
            print(f"\n🔄 Processing Call {idx}...")
            raw_path = os.path.join(AUDIO_DIR, f"audio_{idx}.wav")
            if not download_audio_from_s3(url, raw_path):
                continue

            # Preprocess audio
            cleaned = raw_path.replace(".wav", "_clean.wav")
            normalized = raw_path.replace(".wav", "_norm.wav")
            reduce_noise(raw_path, cleaned)
            normalize_audio(cleaned, normalized)

            # Chunk audio
            chunk_paths = chunk_audio(normalized, os.path.join(CHUNK_DIR, f"call_{idx}"))

            # Transcribe
            tamil_text = transcribe_chunks(chunk_paths, translate=False)
            english_text = transcribe_chunks(chunk_paths, translate=True)

            print(f"\n📞 Tamil Transcript:\n{tamil_text}")
            print(f"\n🌍 English Translation:\n{english_text}")

            # Diarization + Performance Analysis
            print(f"\n🧑‍🤝‍🧑 Speaker Diarization:")
            diarized = diarize_audio(normalized)
            analysis = summarize_performance(diarized)

            print(f"\n🧑‍💼 Staff Speaker: {analysis['staff_speaker']}")
            print(f"📊 Staff Score: {analysis['staff_score']}")
            print(f"📝 Summary: {analysis['summary']}")
            for segment in diarized:
                print(f"[{segment['speaker']}] {segment['text']}")

            logging.info(f"✅ Call {idx} processed successfully.")

        except Exception as e:
            logging.error(f"❌ Error processing call {idx}: {e}")
            print(f"❌ Error processing call {idx}: {e}")

if __name__ == "__main__":
    # Replace with your actual S3 audio URLs
    s3_audio_urls = [
        "https://ai-elroi-bucket.s3.ap-south-1.amazonaws.com/call_audio/call__audio_bajaj_1.wav",
        "https://ai-elroi-bucket.s3.ap-south-1.amazonaws.com/call_audio/call__audio_bajaj_2.wav"
    ]
    process_pipeline(s3_audio_urls)

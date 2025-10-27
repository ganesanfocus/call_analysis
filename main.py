import os
import requests
import logging
import whisper
from utils import reduce_noise, normalize_audio, chunk_audio
from diarize import diarize_audio
from analyze import summarize_performance

logging.basicConfig(filename="logs/transcription.log", level=logging.INFO)
model = whisper.load_model("medium")  # Change to "large" later
AUDIO_DIR = "audio"
CHUNK_DIR = "chunks"

def download_audio_from_s3(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(response.content)
        return True
    else:
        logging.error(f"Failed to download {url}")
        return False

def transcribe_chunks(chunk_paths, translate=False):
    full_text = ""
    for path in chunk_paths:
        result = model.transcribe(path, task="translate" if translate else "transcribe")
        full_text += result["text"].strip() + "\n"
    return full_text.strip()

def process_pipeline(s3_urls):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)

    for idx, url in enumerate(s3_urls):
        try:
            raw_path = os.path.join(AUDIO_DIR, f"audio_{idx}.wav")
            if not download_audio_from_s3(url, raw_path):
                continue

            cleaned = raw_path.replace(".wav", "_clean.wav")
            normalized = raw_path.replace(".wav", "_norm.wav")

            reduce_noise(raw_path, cleaned)
            normalize_audio(cleaned, normalized)

            chunk_paths = chunk_audio(normalized, os.path.join(CHUNK_DIR, f"call_{idx}"))

            tamil_text = transcribe_chunks(chunk_paths, translate=False)
            english_text = transcribe_chunks(chunk_paths, translate=True)

            print(f"\n📞 Call {idx} — Tamil:\n{tamil_text}")
            print(f"\n🌍 Call {idx} — English Translation:\n{english_text}")

            print(f"\n🧑‍🤝‍🧑 Speaker Diarization for Call {idx}:")
            diarized = diarize_audio(normalized)
            analysis = summarize_performance(diarized)
            
            print(f"\n🧑‍💼 Staff Speaker: {analysis['staff_speaker']}")
            print(f"📊 Staff Score: {analysis['staff_score']}")
            print(f"📝 Summary: {analysis['summary']}")
            for segment in diarized:
                print(f"[{segment['speaker']}] {segment['text']}")

            logging.info(f"Call {idx} processed successfully.")

        except Exception as e:
            logging.error(f"Error processing call {idx}: {e}")

if __name__ == "__main__":
    s3_audio_urls = [
        "https://your-bucket.s3.amazonaws.com/audio1.wav",
        "https://your-bucket.s3.amazonaws.com/audio2.wav"
    ]
    process_pipeline(s3_audio_urls)

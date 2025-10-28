#!/usr/bin/env python3
"""
whisper_fast.py  –  < 1 min end-to-end for 8-min calls
Same OpenAI API, zero disk I/O, parallel requests.
"""
import io
import os
import logging
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from dotenv import load_dotenv
from openai import OpenAI

# ---------- config ----------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(
    filename="logs/transcription_openai.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
# ----------------------------

def fetch_wav_bytes(url: str) -> io.BytesIO:
    """Return 16 kHz mono WAV buffer straight from S3."""
    print(f"[{time.strftime('%H:%M:%S')}] ⬇️  Downloading audio …")
    resp = requests.get(url.strip(), timeout=60)
    resp.raise_for_status()
    raw = io.BytesIO(resp.content)
    print(f"[{time.strftime('%H:%M:%S')}] ⚙️  Converting to 16 kHz mono WAV …")
    audio = AudioSegment.from_file(raw).set_channels(1).set_frame_rate(16000)
    wav = io.BytesIO()
    audio.export(wav, format="wav")
    wav.seek(0)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Conversion done")
    return wav

def _whisper_api(wav_bytes: io.BytesIO, *, translate: bool) -> str:
    """Single API call; reusable for both tasks."""
    wav_bytes.seek(0)
    task = "translation" if translate else "transcription"
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 Starting Whisper {task} …")
    kwargs = dict(
        model="whisper-1",
        file=("audio.wav", wav_bytes.read()),
        response_format="text",
    )
    if translate:  # English translation
        kwargs["prompt"] = "Translate this Tamil audio to English."

    result = client.audio.transcriptions.create(**kwargs)  # same endpoint
    print(f"[{time.strftime('%H:%M:%S')}] ✔️ Whisper {task} finished")
    return result

def process_one_call(idx: int, url: str):
    """Download once, run both API calls in parallel."""
    try:
        print(f"\n===== Call {idx} =====")
        wav = fetch_wav_bytes(url)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_tamil = pool.submit(_whisper_api, wav, translate=False)
            fut_eng   = pool.submit(_whisper_api, wav, translate=True)

            tamil, english = fut_tamil.result(), fut_eng.result()

        print(f"\n📞 Tamil Transcript:\n{tamil}")
        print(f"\n🌍 English Translation:\n{english}")
        logging.info(f"✅ Call {idx} done")
        return tamil, english

    except Exception as e:
        logging.error(f"❌ Call {idx}: {e}")
        print(f"❌ Call {idx}: {e}")
        return None, None

def process_pipeline(s3_urls):
    for idx, url in enumerate(s3_urls):
        process_one_call(idx, url)

if __name__ == "__main__":
    s3_audio_urls = [
       "https://ai-elroi-bucket.s3.ap-south-1.amazonaws.com/call_audio/call__audio_bajaj_2_trimmed.wav"
    ]
    process_pipeline(s3_audio_urls)
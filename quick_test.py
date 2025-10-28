#!/usr/bin/env python3
import io, os, time, requests, logging
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

URL = "https://ai-elroi-bucket.s3.ap-south-1.amazonaws.com/call_audio/call__audio_bajaj_2_trimmed.wav"

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def fetch_convert():
    ts("⬇️  Downloading …")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    raw = io.BytesIO(resp.content)
    ts("⚙️  Converting to 16 kHz mono WAV …")
    audio = AudioSegment.from_file(raw).set_channels(1).set_frame_rate(16000)
    wav = io.BytesIO()
    audio.export(wav, format="wav")
    wav.seek(0)
    ts("✅ Conversion done")
    return wav

def whisper(wav: io.BytesIO, *, translate: bool) -> str:
    task = "translation" if translate else "transcription"
    ts(f"🚀 Whisper {task} start …")
    wav.seek(0)
    kwargs = dict(model="whisper-1", file=("audio.wav", wav.read()), response_format="text")
    if translate:
        kwargs["prompt"] = "Translate this Tamil audio to English."
    text = client.audio.transcriptions.create(**kwargs)
    ts(f"✔️ Whisper {task} finished")
    return text

def main():
    wav = fetch_convert()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_tamil = pool.submit(whisper, wav, translate=False)
        fut_eng   = pool.submit(whisper, wav, translate=True)
    tamil, eng = fut_tamil.result(), fut_eng.result()
    ts("===== RESULTS =====")
    print("\n📞 Tamil:\n", tamil)
    print("\n🌍 English:\n", eng)

if __name__ == "__main__":
    main()
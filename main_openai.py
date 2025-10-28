import os
import requests
import logging
from openai import OpenAI


from analyze import summarize_performance  # Optional: if you want to score staff
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup
logging.basicConfig(filename="logs/transcription_openai.log", level=logging.INFO)
AUDIO_DIR = "audio_openai"
os.makedirs(AUDIO_DIR, exist_ok=True)

def download_audio_from_s3(url, local_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

def transcribe_openai(audio_path, translate=False):
    with open(audio_path, "rb") as f:
        if translate:
            response = client.audio.translations.create(
                model="whisper-1",
                file=f,
                response_format="text",
                prompt="Translate this Tamil audio to English."
            )
        else:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                prompt="Transcribe this Tamil audio."
            )
    return response


def process_pipeline(s3_urls):
    for idx, url in enumerate(s3_urls):
        try:
            print(f"\n🔄 Processing Call {idx}...")
            raw_path = os.path.join(AUDIO_DIR, f"audio_{idx}.mp3")
            wav_path = raw_path.replace(".mp3", ".wav")

            if not download_audio_from_s3(url, raw_path):
                continue

            convert_to_wav(raw_path, wav_path)

            tamil_text = transcribe_openai(wav_path, translate=False)
            english_text = transcribe_openai(wav_path, translate=True)

            print(f"\n📞 Tamil Transcript:\n{tamil_text}")
            print(f"\n🌍 English Translation:\n{english_text}")

            # Optional: Score staff performance (if diarization is done separately)
            # segments = [{"speaker": "SPEAKER_00", "text": tamil_text}]  # Dummy structure
            # analysis = summarize_performance(segments)
            # print(f"\n🧑‍💼 Staff Score: {analysis['staff_score']}")
            # print(f"📝 Summary: {analysis['summary']}")

            logging.info(f"✅ Call {idx} processed successfully.")

        except Exception as e:
            logging.error(f"❌ Error processing call {idx}: {e}")
            print(f"❌ Error processing call {idx}: {e}")

if __name__ == "__main__":
    s3_audio_urls = [
        "https://ai-elroi-bucket.s3.ap-south-1.amazonaws.com/call_audio/call__audio_bajaj_1.wav",
        "https://ai-elroi-bucket.s3.ap-south-1.amazonaws.com/call_audio/call__audio_bajaj_2.wav"
    ]
    process_pipeline(s3_audio_urls)

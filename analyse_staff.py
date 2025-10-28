# file: analyse_staff.py
import os, io, time, json, logging, requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from pydub import AudioSegment
from tqdm import tqdm
from openai import OpenAI
import re

import warnings, torchaudio
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
HF_TOKEN= os.getenv("HF_TOKEN")



AUDIO_SAMPLE_RATE = 16_000
logging.basicConfig(
    filename="logs/staff_score.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Block  Tiny helpers
def ts() -> str:
    return datetime.utcnow().strftime("%H:%M:%S")

def download_to_bytes(url: str) -> io.BytesIO:
    logging.info(f"Downloading {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return io.BytesIO(resp.content)

def convert_to_wav(raw_bytes: io.BytesIO) -> io.BytesIO:
    """Return 16 kHz mono WAV in memory."""
    audio = AudioSegment.from_file(raw_bytes).set_channels(1).set_frame_rate(AUDIO_SAMPLE_RATE)
    out = io.BytesIO()
    audio.export(out, format="wav")
    out.seek(0)
    return out

# Block 3:
def whisper_json(wav_bytes: io.BytesIO, language: str = "ta"):
    wav_bytes.seek(0)
    logging.info("Whisper start")
    result = client.audio.transcriptions.create(
    model="whisper-1",
    file=("audio.wav", wav_bytes.read()),
    response_format="verbose_json",   # gives word-level timestamps
    language=language,
    timestamp_granularities=["word"]
    )
    logging.info("Whisper done")
    return result  # OpenAI object with .words and .text

# BLOCK 4 – speaker diarization (pyannote)
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN  # hugging-face token once
)
# pipeline.to(torch.device("cpu"))  # or "cuda" if you have GPU

def diarize(wav_bytes: io.BytesIO):
    wav_bytes.seek(0)
    logging.info("Diarization start")
    # pyannote needs a file-like object that supports seeking
    diar = pipeline({"uri": "memo", "audio": wav_bytes})
    logging.info("Diarization done")
    return diar  # pyannote.core.Annotation object

# BLOCK 5 – fuse Whisper words + diar labels
def align_words_to_speakers(whisper_result, diar) -> list[dict]:
    """
    Returns list of dicts:
    {"word": "வணக்கம்", "start": 0.34, "end": 0.88, "speaker": "SPEAKER_00"}
    """
    words = whisper_result.words
    aligned = []
    for w in words:
        start, end = w.start, w.end
        # pick speaker that covers the middle of the word
        mid = (start + end) / 2
        # speaker = diar.crop((mid, mid)).argmax()
        segment = diar.crop((mid, mid))
        speaker = segment.argmax() if segment else "UNKNOWN"
        aligned.append({
            "word": w.word.strip(),
            "start": start,
            "end": end,
            "speaker": speaker
        })
    return aligned

# BLOCK 6 – build speaker segments
def build_segments(aligned_words: list[dict], min_sec=1.0) -> list[dict]:
    """
    Group consecutive same-speaker words into segments.
    [{"speaker": "SPEAKER_00", "text": "வணக்கம் மெம்", "start": 0.34, "end": 2.10}, ...]
    """
    segments = []
    current_spk = None
    buf_words = []
    buf_start = None

    for w in aligned_words:
        if w["speaker"] != current_spk:
            # flush previous
            if buf_words and (buf_words[-1]["end"] - buf_start) >= min_sec:
                segments.append({
                    "speaker": current_spk,
                    "text": " ".join([ww["word"] for ww in buf_words]),
                    "start": buf_start,
                    "end": buf_words[-1]["end"]
                })
            # new bucket
            current_spk = w["speaker"]
            buf_words = [w]
            buf_start = w["start"]
        else:
            buf_words.append(w)

    # final flush
    if buf_words and (buf_words[-1]["end"] - buf_start) >= min_sec:
        segments.append({
            "speaker": current_spk,
            "text": " ".join([ww["word"] for ww in buf_words]),
            "start": buf_start,
            "end": buf_words[-1]["end"]
        })
    return segments

# BLOCK 7 – identify staff speaker
def tag_staff_Speaker(segments: list[dict]) -> str:
    """
    Heuristic: speaker who speaks first AND mentions company keyword.
    Returns the speaker label to be scored.
    """
    keywords = {"பஜாஜ்", "bajaj", "பஜார்", "finance", "பைனான்ஸ்"}
    for seg in segments:
        if any(k in seg["text"].lower() for k in keywords):
            return seg["speaker"]
    # fallback: most talkative speaker
    from collections import Counter
    dur = Counter()
    for s in segments:
        dur[s["speaker"]] += s["end"] - s["start"]
    return dur.most_common(1)[0][0]   # longest

# BLOCK 8 – score agent text via GPT-4
import re

def gpt_score(agent_text: str) -> dict:
    prompt = (
        "You are a call-centre quality analyst.\n"
        "Rate this agent transcript 0-100 on:\n"
        "1) Politeness  2) Clarity  3) Product knowledge  4) Compliance\n"
        "Return **only** the following JSON, no other text:\n"
        "{\"politeness\":<int>,\"clarity\":<int>,\"knowledge\":<int>,\"compliance\":<int>}\n\n"
        f"Transcript:\n{agent_text}"
    )
    logging.info("GPT-4 scoring start")
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    txt = resp.choices[0].message.content.strip()
    logging.info("GPT-4 raw reply: %s", txt)

    # --- safe parse ---
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # regex fallback: grab first {...} block
        m = re.search(r'\{.*?\}', txt, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        # ultimate fallback
        logging.warning("GPT returned non-JSON, using 0 scores")
        return {"politeness": 0, "clarity": 0, "knowledge": 0, "compliance": 0}

# BLOCK 9 – coaching summary (optional)
def gpt_summary(agent_text: str) -> str:
    prompt = (
        "Summarise in 3 bullets what the agent did well and 3 bullets for improvement.\n"
        "Keep each bullet under 12 words.\n\n"
        f"Transcript:\n{agent_text}"
    )
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# BLOCK 10 – glue everything together

def print_conversation(segments: list[dict], max_lines=30):
    """Pretty-print the first <max_lines> segments."""
    print("\n========== CONVERSATION ==========")
    for i, s in enumerate(segments[:max_lines], 1):
        start = f"{s['start']:>5.1f}"
        print(f"{start}s  {s['speaker']:<12}  {s['text']}")
    if len(segments) > max_lines:
        print(f"...  {len(segments)-max_lines} more segments")
    print("==================================\n")
    
def analyse_call(s3_url: str) -> dict:
    ts0 = time.time()
    raw = download_to_bytes(s3_url)
    wav = convert_to_wav(raw)

    whisper_result = whisper_json(wav, language="ta")
    diar = diarize(wav)

    aligned = align_words_to_speakers(whisper_result, diar)
    segments = build_segments(aligned, min_sec=1.0)
    print_conversation(segments)          # <-- add this line


    staff_label = tag_staff_Speaker(segments)
    agent_segments = [s for s in segments if s["speaker"] == staff_label]
    agent_text = " ".join([s["text"] for s in agent_segments])

    score_dict = gpt_score(agent_text)
    summary = gpt_summary(agent_text)

    total_sec = segments[-1]["end"] if segments else 0
    staff_score = round(sum(score_dict.values()) / 4, 1)

    out = {
        "call_id": Path(s3_url).stem,
        "staff_label": staff_label,
        "staff_score": staff_score,
        "breakdown": score_dict,
        "summary": summary,
        "duration_sec": round(total_sec, 1),
        "agent_word_count": len(agent_text.split()),
    }
    logging.info(f"Finished call {out['call_id']} in {round(time.time()-ts0,1)} s")
    return out

# BLOCK 11 – batch runner
if __name__ == "__main__":
    urls = [
        "https://ai-elroi-bucket.s3.ap-south-1.amazonaws.com/call_audio/call__audio_bajaj_2_trimmed.wav",
        # add more
    ]
    for url in tqdm(urls, desc="Calls"):
        result = analyse_call(url)
        print(json.dumps(result, indent=2, ensure_ascii=False))
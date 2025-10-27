import os
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize

def reduce_noise(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None)
    reduced = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced, sr)

def normalize_audio(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    normalized = normalize(audio)
    normalized.export(output_path, format="wav")

def chunk_audio(input_path, chunk_dir, chunk_length_ms=60000):
    audio = AudioSegment.from_file(input_path)
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_paths = []
    for idx, chunk in enumerate(chunks):
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    return chunk_paths

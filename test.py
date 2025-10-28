import whisper
model = whisper.load_model("tiny", device="cpu")
result = model.transcribe("call__audio_bajaj_2_trimmed.wav", language="ta", initial_prompt="பஜாஜ் ஃபைனான்ஸ்")
print(result["text"])
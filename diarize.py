import whisperx

def diarize_audio(audio_path, model_size="medium", language="ta", device="cuda"):
    # Load Whisper model
    model = whisperx.load_model(model_size, device=device, language=language)

    # Transcribe with timestamps
    result = model.transcribe(audio_path, batch_size=16, vad_filter=True, vad_parameters={"threshold": 0.5})

    # Load alignment model
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device=device)

    # Speaker diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="your_huggingface_token", device=device)
    diarize_segments = diarize_model(audio_path)

    # Combine speaker labels with aligned segments
    result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result_aligned["word_segments"])
    return result_with_speakers

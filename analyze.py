def map_speakers(segments):
    """
    Automatically map speakers to 'staff' and 'customer' based on loan-related keywords.
    Assumes the speaker who uses more loan-related terms is the staff.
    """
    speaker_counts = {}
    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"].lower()
        if speaker not in speaker_counts:
            speaker_counts[speaker] = {"loan_words": 0, "total": 0}
        if any(word in text for word in ["loan", "interest", "repayment", "approval", "emi", "eligible", "apply"]):
            speaker_counts[speaker]["loan_words"] += 1
        speaker_counts[speaker]["total"] += 1

    staff_speaker = max(speaker_counts.items(), key=lambda x: x[1]["loan_words"])[0]
    customer_speaker = [s for s in speaker_counts if s != staff_speaker][0]

    return {
        "staff": staff_speaker,
        "customer": customer_speaker
    }


def score_staff_segments(segments, staff_speaker):
    """
    Score staff performance based on presence of persuasive phrases, clarity, and confidence.
    """
    score = 0
    total = 0
    for seg in segments:
        if seg["speaker"] != staff_speaker:
            continue
        text = seg["text"].lower()
        if any(word in text for word in ["loan", "interest", "repayment", "approval", "emi"]):
            score += 2
        if any(word in text for word in ["apply", "eligible", "shall i proceed", "can i help"]):
            score += 1
        if any(word in text for word in ["uh", "hmm", "not sure", "don’t know"]):
            score -= 1
        total += 1

    return round(score / max(total, 1), 2)


def summarize_performance(segments):
    """
    Full analysis pipeline: maps speakers, scores staff, returns summary.
    """
    mapping = map_speakers(segments)
    staff_score = score_staff_segments(segments, mapping["staff"])

    summary = ""
    if staff_score >= 6:
        summary = "Staff explained loan terms clearly and used persuasive language. Good performance."
    elif staff_score >= 3:
        summary = "Staff gave basic loan info but missed persuasive cues. Needs improvement."
    else:
        summary = "Staff performance was weak or unclear. Recommend review or coaching."

    return {
        "staff_speaker": mapping["staff"],
        "customer_speaker": mapping["customer"],
        "staff_score": staff_score,
        "summary": summary
    }

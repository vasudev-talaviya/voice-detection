from dataclasses import dataclass
import os
import torch
import librosa
from database.operation.crud import fetch_record_by_id
from helper.voice_embedding_convert import voice_to_embedding
from model.machine_learning.config import PREDICT_THRESHOLD
from model.machine_learning.similarity import best_cosine_match, print_scores, similarity_to_confidence, explain_match

@dataclass

class PredictResult:
    """
    Structured result of a speaker identification query.
    """
    person: dict | None
    best_sim: float
    confidence: float
    above_threshold: bool
    explanation: str
    embedding_tensor: torch.Tensor
    all_scores: list  # list of tuples (name, sim)


def predict(audio_file_path: str, threshold: float = PREDICT_THRESHOLD) -> PredictResult:
    """
    Identify the speaker of an audio file using cosine similarity.
    No deep-learning classifier is involved — matching is done by comparing
    the extracted embedding directly against all stored embeddings.

    Args:
        audio_file_path : Path to the query audio file.
        threshold       : Minimum cosine similarity to accept an identification.
                          Defaults to PREDICT_THRESHOLD from config.py.

    Returns:
        PredictResult   : Dataclass containing identification results, confidence, and explanations.
    """
    # Load audio to check duration for explanation
    try:
        y, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        audio_duration_sec = len(y) / sr
    except Exception:
        audio_duration_sec = None

    embedding_tensor = voice_to_embedding(audio_file_path)
    best_sim, best_id, best_name, all_scores = best_cosine_match(embedding_tensor)

    confidence = similarity_to_confidence(best_sim)
    above_threshold = best_sim >= threshold

    explanation = ""
    if not all_scores:
        print("[INFO] No registered voices in database.")
        person = None
    else:
        print_scores(all_scores, best_name)
        if above_threshold:
            person = fetch_record_by_id(best_id)
            print(f"[RESULT] Identified as: '{best_name}'  (similarity: {best_sim:.4f}, confidence: {confidence:.4f})")
        else:
            person = None
            print(f"[RESULT] Speaker not recognized.  "
                  f"Best similarity: {best_sim:.4f} < threshold {threshold}")
        
        # Populate explanation if match is in the uncertain band or below threshold
        explanation = explain_match(best_sim, best_name, audio_duration_sec)

    return PredictResult(
        person=person,
        best_sim=best_sim,
        confidence=confidence,
        above_threshold=above_threshold,
        explanation=explanation,
        embedding_tensor=embedding_tensor,
        all_scores=all_scores
    )


# ──────────────────────────────────────────────────────────────────────────────
# Interactive CLI entry-points
# ──────────────────────────────────────────────────────────────────────────────

def predict_voice():
    """Interactive CLI for speaker identification using cosine similarity."""
    file_path = input("  Enter audio file path : ").strip()

    if not file_path:
        print("[Error] File path cannot be empty.")
        return

    try:
        res = predict(file_path)
    except Exception as e:
        print(f"\n[Error] Prediction failed: {e}\n")
        return

    if res.above_threshold and res.person:
        print(f"\n Speaker identified → Name : {res.person['name']}  "
              f"(similarity: {res.best_sim:.4f}, confidence: {res.confidence:.4f})\n")
    else:
        print("\n[RESULT] Speaker not recognized.\n")
        if res.explanation:
            print(f"[EXPLANATION] {res.explanation}\n")

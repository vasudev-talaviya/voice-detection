from dataclasses import dataclass, field
import os
import torch
import librosa
from database.operation.crud import fetch_record_by_id
from helper.voice_embedding_convert import voice_to_embedding
from model.machine_learning.config import PREDICT_THRESHOLD
from model.machine_learning.similarity import (
    best_match,
    print_scores,
    similarity_to_confidence,
    explain_match,
)

@dataclass

class PredictResult:
    """
    Structured result of a speaker identification query.
    """
    person: dict | None
    best_fusion: float
    confidence: float
    above_threshold: bool
    explanation: str
    embedding_tensor: torch.Tensor
    all_scores: list       # list of dicts with per-metric details
    best_euclidean: float = 0.0
    best_plda: float = 0.0


def predict(audio_file_path: str, threshold: float = PREDICT_THRESHOLD) -> PredictResult:
    """
    Identify the speaker of an audio file using multi-metric fusion scoring.

    Matching is done by comparing the extracted embedding directly against
    all stored embeddings using Euclidean + PLDA fusion.

    Args:
        audio_file_path : Path to the query audio file.
        threshold       : Minimum fusion score to accept an identification.
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
    best_fusion, best_id, best_name, all_scores = best_match(embedding_tensor)

    confidence = similarity_to_confidence(best_fusion)
    above_threshold = best_fusion >= threshold

    # Extract individual metric scores for the best match
    best_euclidean = 0.0
    best_plda      = 0.0
    if all_scores:
        top = all_scores[0]
        best_euclidean = top.get("euclidean_score", 0.0)
        best_plda      = top.get("plda_score", 0.0)

    explanation = ""
    if not all_scores:
        print("[INFO] No registered voices in database.")
        person = None
    else:
        print_scores(all_scores, best_name)
        if above_threshold:
            person = fetch_record_by_id(best_id)
            print("  +---------------------------------------------------+")
            print("  |              >>  SPEAKER IDENTIFIED                |")
            print("  +------------------+--------------------------------+")
            print(f"  |  Speaker         |  {best_name:<28}  |")
            print(f"  |  Fusion Score    |  {best_fusion:<28.4f}  |")
            print(f"  |  Euclidean       |  {best_euclidean:<28.4f}  |")
            print(f"  |  PLDA            |  {best_plda:<28.4f}  |")
            print(f"  |  Confidence      |  {confidence:<28.4f}  |")
            print(f"  |  Threshold       |  {threshold:<28}  |")
            print("  +------------------+--------------------------------+")
        else:
            person = None
            print("  +---------------------------------------------------+")
            print("  |              X  SPEAKER NOT RECOGNIZED             |")
            print("  +------------------+--------------------------------+")
            print(f"  |  Best Match      |  {best_name or 'N/A':<28}  |")
            print(f"  |  Fusion Score    |  {best_fusion:<28.4f}  |")
            print(f"  |  Threshold       |  {threshold:<28}  |")
            print("  +------------------+--------------------------------+")
        
        # Populate explanation if match is in the uncertain band or below threshold
        explanation = explain_match(best_fusion, best_name, audio_duration_sec)

    return PredictResult(
        person=person,
        best_fusion=best_fusion,
        confidence=confidence,
        above_threshold=above_threshold,
        explanation=explanation,
        embedding_tensor=embedding_tensor,
        all_scores=all_scores,
        best_euclidean=best_euclidean,
        best_plda=best_plda,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Interactive CLI entry-points
# ──────────────────────────────────────────────────────────────────────────────

def predict_voice():
    """Interactive CLI for speaker identification using multi-metric fusion."""
    file_path = input("  Enter audio file path : ").strip()

    if not file_path:
        print("[Error] File path cannot be empty.")
        return

    try:
        res = predict(file_path)
    except Exception as e:
        print(f"\n[Error] Prediction failed: {e}\n")
        return

    if not (res.above_threshold and res.person):
        if res.explanation:
            print(f"\n  [TIP] {res.explanation}\n")

import torch
import math

from database.operation.crud import fetch_all_embedding, fetch_record_by_id
from helper.voice_embedding_convert import voice_to_embedding
from model.machine_learning.config import (
    REGISTER_THRESHOLD,
    PREDICT_THRESHOLD,
    UNCERTAIN_BAND,
    CONFIDENCE_EXPLAIN_RANGE,
    FUSION_WEIGHTS,
    PLDA_SCALE,
    PLDA_SHIFT,
)


# ──────────────────────────────────────────────────────────────────────────────
# Individual similarity metrics
# ──────────────────────────────────────────────────────────────────────────────

def _euclidean_score(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Euclidean-distance-based similarity score.

    For L2-normalised 512-dim x-vectors the Euclidean distance lives in [0, 2]:
      • 0.0  → identical vectors
      • 2.0  → perfectly anti-parallel vectors

    We convert distance → similarity:
        score = 1 - (dist / 2)

    Range: [0, 1].  1 = identical, 0 = perfectly opposite.

    Why Euclidean over cosine?
      For L2-normalised vectors:  dist² = 2 - 2·cos(θ)
      Euclidean captures the same angular information but in a linear [0,1]
      scale that provides better magnitude separation in the high-similarity
      region where speaker verification decisions are made.
    """
    dist = torch.dist(a, b, p=2).item()
    return max(0.0, 1.0 - dist / 2.0)


def _plda_logistic_score(euc_score: float) -> float:
    """
    PLDA-inspired calibrated score using a logistic (sigmoid) transform.

    This maps the Euclidean similarity score through a learned-style sigmoid:
        score = σ(scale * (euc_score - shift))

    Where:
      • `scale` controls the sharpness of the transition (higher = sharper)
      • `shift` is the decision boundary (euc score at which output = 0.5)

    This is modelled after how PLDA log-likelihood-ratio scoring works in
    state-of-the-art speaker verification systems (Kaldi, SpeechBrain).
    The sigmoid naturally produces well-calibrated probabilities, making the
    score more interpretable than raw distance.

    Range: (0, 1).  ~0 = definitely different speaker, ~1 = definitely same.
    """
    return 1.0 / (1.0 + math.exp(-PLDA_SCALE * (euc_score - PLDA_SHIFT)))


# ──────────────────────────────────────────────────────────────────────────────
# Multi-metric fusion — the new core scoring function
# ──────────────────────────────────────────────────────────────────────────────

def compute_fusion_score(
    query_vec: torch.Tensor,
    stored_vec: torch.Tensor,
) -> dict:
    """
    Compute a fused similarity score from complementary metrics.

    Metrics used:
      1. Euclidean distance score — magnitude-based similarity [0, 1]
      2. PLDA logistic score      — calibrated probabilistic output (0, 1)

    The final fusion score is a weighted average:
        fusion = w_euc * euc_score + w_plda * plda_score

    Both components already share the [0, 1] range.

    Args:
        query_vec  : torch.Tensor [512] — query speaker embedding (L2-normalised).
        stored_vec : torch.Tensor [512] — stored speaker embedding (L2-normalised).

    Returns:
        dict with keys:
            euclidean_score (float) — distance-based similarity [0, 1]
            plda_score      (float) — logistic-calibrated score (0, 1)
            fusion_score    (float) — weighted combination [0, 1]
    """
    euc  = _euclidean_score(query_vec, stored_vec)
    plda = _plda_logistic_score(euc)

    w = FUSION_WEIGHTS
    fusion = w["euclidean"] * euc + w["plda"] * plda

    return {
        "euclidean_score":  round(euc,    6),
        "plda_score":       round(plda,   6),
        "fusion_score":     round(fusion, 6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core engine — compares a query embedding against all DB embeddings
# ──────────────────────────────────────────────────────────────────────────────

def best_match(query_tensor: torch.Tensor):
    """
    Scan every stored embedding and find the closest match using the
    multi-metric fusion score.

    Args:
        query_tensor : torch.Tensor of shape [1, 512] (L2-normalised x-vector).

    Returns:
        best_fusion : float               — highest fusion score found (-1.0 if DB empty)
        best_id     : str | None          — MongoDB _id string of the best match
        best_name   : str | None          — registered name of the best match
        all_scores  : list[dict]          — per-speaker score details, sorted desc
    """
    records = fetch_all_embedding()
    if not records:
        return -1.0, None, None, []

    query_vec = query_tensor.squeeze(0).float()   # [512]
    best_fusion = -1.0
    best_id     = None
    best_name   = None
    all_scores: list[dict] = []

    for record in records:
        record_id = str(record["_id"])
        name      = record.get("name", record_id)

        stored_flat = [v for segment in record["embedding"] for v in segment]
        stored_vec  = torch.tensor(stored_flat, dtype=torch.float32)

        metrics = compute_fusion_score(query_vec, stored_vec)
        metrics["name"] = name
        metrics["record_id"] = record_id
        all_scores.append(metrics)

        if metrics["fusion_score"] > best_fusion:
            best_fusion = metrics["fusion_score"]
            best_id     = record_id
            best_name   = name

    all_scores.sort(key=lambda x: x["fusion_score"], reverse=True)
    return best_fusion, best_id, best_name, all_scores


# Legacy wrapper — keeps old call-sites working
def best_cosine_match(query_tensor: torch.Tensor):
    """
    Backward-compatible wrapper that returns the same 4-tuple as the
    original best_cosine_match(), but uses the new fusion engine internally.

    Returns:
        best_sim   : float           — fusion score of best match
        best_id    : str | None
        best_name  : str | None
        all_scores : list[tuple]     — (name, fusion_score) pairs, sorted desc
    """
    best_fusion, best_id, best_name, detailed = best_match(query_tensor)
    legacy_scores = [(s["name"], s["fusion_score"]) for s in detailed]
    return best_fusion, best_id, best_name, legacy_scores


# ──────────────────────────────────────────────────────────────────────────────
# Confidence scoring
# ──────────────────────────────────────────────────────────────────────────────

def similarity_to_confidence(fusion_score: float) -> float:
    """
    Convert a fusion score to a normalised confidence value in [0, 1].

    The fusion score already lives in [0, 1], so this is essentially
    an identity mapping (clamped for safety).

    Args:
        fusion_score : float — fused similarity score in [0.0, 1.0].

    Returns:
        float — confidence in [0.0, 1.0].
    """
    return float(max(0.0, min(1.0, fusion_score)))


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty explanation
# ──────────────────────────────────────────────────────────────────────────────

def explain_match(
    sim: float,
    best_name: str | None,
    audio_duration_sec: float | None = None,
) -> str:
    """
    Produce a human-readable explanation when a match falls in the uncertain band
    [PREDICT_THRESHOLD, PREDICT_THRESHOLD + UNCERTAIN_BAND].

    Args:
        sim                : Fusion score of the best match.
        best_name          : Name of the candidate speaker, or None.
        audio_duration_sec : Duration of the query audio (seconds), used to
                             contextualise borderline results.

    Returns:
        str — explanation text, or empty string if no explanation is needed.
    """
    low, high = CONFIDENCE_EXPLAIN_RANGE

    if sim < low:
        return (
            f"Speaker not recognised — best similarity {sim:.4f} is below the "
            f"identification threshold ({low:.2f}). "
            "Register this voice or provide a longer, clearer sample."
        )

    if low <= sim < high:
        lines = [
            f"Uncertain match to '{best_name}' (similarity {sim:.4f}, "
            f"uncertain band {low:.2f}–{high:.2f}).",
        ]
        if audio_duration_sec is not None and audio_duration_sec < 10.0:
            lines.append(
                f"Audio duration is borderline ({audio_duration_sec:.1f} s). "
                "A longer sample (≥ 10 s) would improve confidence."
            )
        lines.append(
            "Recommendation: register additional samples for this speaker or "
            "lower PREDICT_THRESHOLD if false rejections are too frequent."
        )
        return " ".join(lines)

    return ""   # sim >= high → confident match, no explanation needed


# ──────────────────────────────────────────────────────────────────────────────
# Direct embedding comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare_embeddings(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    threshold: float = PREDICT_THRESHOLD,
) -> dict:
    """
    One-shot comparison of two speaker embeddings using the multi-metric
    fusion scoring system.

    Both embeddings must be L2-normalised [1, 512] x-vectors as returned by
    `voice_to_embedding()`.

    Args:
        emb_a     : torch.Tensor [1, 512] — first speaker embedding.
        emb_b     : torch.Tensor [1, 512] — second speaker embedding.
        threshold : float — minimum fusion score to declare the speakers identical.

    Returns:
        dict with keys:
            euclidean_score (float)  — Euclidean distance-based score [0, 1]
            plda_score      (float)  — PLDA logistic calibrated score (0, 1)
            fusion_score    (float)  — weighted combination [0, 1]
            confidence      (float)  — normalised confidence [0, 1]
            same_speaker    (bool)   — True only if fusion_score >= threshold
            explanation     (str)    — non-empty when match is uncertain
    """
    a = emb_a.squeeze(0).float()
    b = emb_b.squeeze(0).float()

    metrics     = compute_fusion_score(a, b)
    fusion      = metrics["fusion_score"]
    confidence  = similarity_to_confidence(fusion)
    same        = fusion >= threshold
    explanation = explain_match(fusion, best_name="Speaker B")

    return {
        **metrics,
        "confidence":   round(confidence, 6),
        "same_speaker": same,
        "explanation":  explanation,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_scores(all_scores: list, best_name: str):
    """
    Pretty-print per-speaker similarity scores in a structured table format.

    Handles both legacy tuple format and new dict format.
    """
    if not all_scores:
        print("  [INFO] No scores to display.")
        return

    # Check format
    is_dict = isinstance(all_scores[0], dict)

    if is_dict:
        # ── Table header ──
        print()
        print("  +-----+----------------------+------------+------------+------------+--------+")
        print("  |  #  | Speaker              | Euclidean  |   PLDA     |  Fusion    | Status |")
        print("  +-----+----------------------+------------+------------+------------+--------+")

        for i, entry in enumerate(all_scores, 1):
            name   = entry["name"]
            euc    = entry["euclidean_score"]
            plda   = entry["plda_score"]
            fusion = entry["fusion_score"]
            status = "* BEST" if name == best_name else "      "

            # Truncate long names
            display_name = name[:20] if len(name) > 20 else name

            print(
                f"  | {i:<3} | {display_name:<20} |   {euc:.4f}   |   {plda:.4f}   |   {fusion:.4f}   | {status} |"
            )

        print("  +-----+----------------------+------------+------------+------------+--------+")
        print()
    else:
        # ── Legacy tuple format ──
        print()
        print("  +-----+----------------------+------------+------------+--------+")
        print("  |  #  | Speaker              | Similarity | Confidence | Status |")
        print("  +-----+----------------------+------------+------------+--------+")

        for i, (name, score) in enumerate(all_scores, 1):
            conf   = similarity_to_confidence(score)
            status = "* BEST" if name == best_name else "      "
            display_name = name[:20] if len(name) > 20 else name

            print(
                f"  | {i:<3} | {display_name:<20} |   {score:+.4f}  |   {conf:.4f}   | {status} |"
            )

        print("  +-----+----------------------+------------+------------+--------+")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Duplicate check — used by the registration flow
# ──────────────────────────────────────────────────────────────────────────────

def cosine_similarity_check(audio_file_path: str, threshold: float = REGISTER_THRESHOLD):
    """
    Detect whether an audio file matches an already-registered voice.
    Called during registration to prevent duplicate entries.

    Now uses the multi-metric fusion score for more accurate duplicate detection.

    Args:
        audio_file_path : Path to the query audio file.
        threshold       : Fusion score cutoff (0–1).
                          Higher → stricter duplicate detection.
                          Defaults to REGISTER_THRESHOLD from config.py.

    Returns:
        is_duplicate     : bool             — True if a match ≥ threshold exists
        best_similarity  : float            — highest fusion score found
        matched_name     : str | None       — name of the closest existing voice
        embedding_tensor : torch.Tensor     — reused by register.py to avoid re-extraction
    """
    embedding_tensor = voice_to_embedding(audio_file_path)
    best_fusion, best_id, best_name, detailed = best_match(embedding_tensor)

    if not detailed:
        print("[INFO] No stored embeddings found — treating as new voice.")
        return False, 0.0, None, embedding_tensor

    print_scores(detailed, best_name)

    is_duplicate = best_fusion >= threshold

    if is_duplicate:
        print("  +---------------------------------------------------+")
        print("  |            !  DUPLICATE VOICE DETECTED             |")
        print("  +------------------+--------------------------------+")
        print(f"  |  Matched Speaker |  {best_name:<28}  |")
        print(f"  |  Fusion Score    |  {best_fusion:<28.4f}  |")
        print(f"  |  Threshold       |  {threshold:<28}  |")
        print("  +------------------+--------------------------------+")
    else:
        print("  +---------------------------------------------------+")
        print("  |            >>  NO DUPLICATE FOUND                  |")
        print("  +------------------+--------------------------------+")
        print(f"  |  Closest Match   |  {best_name or 'N/A':<28}  |")
        print(f"  |  Fusion Score    |  {best_fusion:<28.4f}  |")
        print(f"  |  Threshold       |  {threshold:<28}  |")
        print("  +------------------+--------------------------------+")

    return is_duplicate, best_fusion, best_name, embedding_tensor

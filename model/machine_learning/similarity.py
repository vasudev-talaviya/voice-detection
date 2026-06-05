import torch
import torch.nn.functional as F

from database.operation.crud import fetch_all_embedding, fetch_record_by_id
from helper.voice_embedding_convert import voice_to_embedding
from model.machine_learning.config import (
    REGISTER_THRESHOLD,
    PREDICT_THRESHOLD,
    UNCERTAIN_BAND,
    CONFIDENCE_EXPLAIN_RANGE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Core engine — compares a query embedding against all DB embeddings
# ──────────────────────────────────────────────────────────────────────────────

def best_cosine_match(query_tensor: torch.Tensor):
    """
    Scan every stored embedding in the database and find the closest match
    to the query using cosine similarity (Rule 4).

    Args:
        query_tensor : torch.Tensor of shape [1, 512]  (L2-normalised x-vector).

    Returns:
        best_sim   : float               — highest similarity found (-1.0 if DB empty)
        best_id    : str | None          — MongoDB _id string of the best match
        best_name  : str | None          — registered name of the best match
        all_scores : list[tuple]         — (name, similarity) pairs, sorted desc
    """
    records = fetch_all_embedding()   # list[dict]; each has _id, name, embedding
    if not records:
        return -1.0, None, None, []

    query_vec = query_tensor.squeeze(0).float()   # [512]
    best_sim  = -1.0
    best_id   = None
    best_name = None
    all_scores: list[tuple] = []

    for record in records:
        record_id = str(record["_id"])
        name      = record.get("name", record_id)

        stored_flat = [v for segment in record["embedding"] for v in segment]
        stored_vec  = torch.tensor(stored_flat, dtype=torch.float32)

        sim = F.cosine_similarity(
            query_vec.unsqueeze(0),
            stored_vec.unsqueeze(0)
        ).item()

        all_scores.append((name, sim))

        if sim > best_sim:
            best_sim  = sim
            best_id   = record_id
            best_name = name

    all_scores.sort(key=lambda x: x[1], reverse=True)
    return best_sim, best_id, best_name, all_scores


# ──────────────────────────────────────────────────────────────────────────────
# Confidence scoring (Rule 7)
# ──────────────────────────────────────────────────────────────────────────────

def similarity_to_confidence(sim: float) -> float:
    """
    Convert a cosine similarity score to a normalised confidence value in [0, 1].

    Cosine similarity from L2-normalised x-vectors lives in [-1, 1].
    We remap [-1, 1] → [0, 1] linearly so that:
      •  sim = -1  →  confidence = 0.0
      •  sim =  0  →  confidence = 0.5
      •  sim = +1  →  confidence = 1.0

    Args:
        sim : float — raw cosine similarity in [-1.0, 1.0].

    Returns:
        float — confidence in [0.0, 1.0].
    """
    return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty explanation (Rule 8)
# ──────────────────────────────────────────────────────────────────────────────

def explain_match(
    sim: float,
    best_name: str | None,
    audio_duration_sec: float | None = None,
) -> str:
    """
    Produce a human-readable explanation when a match falls in the uncertain band
    [PREDICT_THRESHOLD, PREDICT_THRESHOLD + UNCERTAIN_BAND]  (Rule 8).

    Args:
        sim                : Cosine similarity score of the best match.
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
# Direct embedding comparison (Rules 3, 4, 7)
# ──────────────────────────────────────────────────────────────────────────────

def compare_embeddings(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    threshold: float = PREDICT_THRESHOLD,
) -> dict:
    """
    One-shot comparison of two speaker embeddings using cosine similarity (Rules 3–7).

    Both embeddings must be L2-normalised [1, 512] x-vectors as returned by
    `voice_to_embedding()`.

    Args:
        emb_a     : torch.Tensor [1, 512] — first speaker embedding.
        emb_b     : torch.Tensor [1, 512] — second speaker embedding.
        threshold : float — minimum similarity to declare the speakers identical.

    Returns:
        dict with keys:
            similarity   (float)  — cosine similarity in [-1, 1]
            confidence   (float)  — normalised confidence in [0, 1]
            same_speaker (bool)   — True only if similarity >= threshold (Rule 5)
            explanation  (str)    — non-empty when match is uncertain (Rule 8)
    """
    a = emb_a.squeeze(0).float()
    b = emb_b.squeeze(0).float()

    sim        = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    confidence = similarity_to_confidence(sim)
    same       = sim >= threshold
    explanation = explain_match(sim, best_name="Speaker B")

    return {
        "similarity":   round(sim, 6),
        "confidence":   round(confidence, 6),
        "same_speaker": same,
        "explanation":  explanation,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_scores(all_scores: list, best_name: str):
    """Pretty-print per-speaker similarity scores (Rule 7)."""
    print("[INFO] Cosine similarity scores vs stored voices:")
    for name, sim in all_scores:
        conf   = similarity_to_confidence(sim)
        marker = " ← best" if name == best_name else ""
        print(f"       {name:<20}  sim={sim:+.4f}  conf={conf:.4f}{marker}")


# ──────────────────────────────────────────────────────────────────────────────
# Duplicate check — used by the registration flow
# ──────────────────────────────────────────────────────────────────────────────

def cosine_similarity_check(audio_file_path: str, threshold: float = REGISTER_THRESHOLD):
    """
    Detect whether an audio file matches an already-registered voice.
    Called during registration to prevent duplicate entries (Rule 6).

    Args:
        audio_file_path : Path to the query audio file.
        threshold       : Cosine-similarity cutoff (0–1).
                          Higher → stricter duplicate detection.
                          Defaults to REGISTER_THRESHOLD from config.py.

    Returns:
        is_duplicate     : bool             — True if a match ≥ threshold exists
        best_similarity  : float            — highest similarity score found
        matched_name     : str | None       — name of the closest existing voice
        embedding_tensor : torch.Tensor     — reused by register.py to avoid re-extraction
    """
    embedding_tensor = voice_to_embedding(audio_file_path)
    best_sim, _, best_name, all_scores = best_cosine_match(embedding_tensor)

    if not all_scores:
        print("[INFO] No stored embeddings found — treating as new voice.")
        return False, 0.0, None, embedding_tensor

    print_scores(all_scores, best_name)

    is_duplicate = best_sim >= threshold

    if is_duplicate:
        print(
            f"[RESULT] Duplicate detected → '{best_name}'  "
            f"(similarity {best_sim:.4f} ≥ threshold {threshold})"
        )
    else:
        print(
            f"[RESULT] No duplicate found.  "
            f"Best similarity: {best_sim:.4f} < threshold {threshold}"
        )

    return is_duplicate, best_sim, best_name, embedding_tensor

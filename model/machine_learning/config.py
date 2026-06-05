
# ── Audio Input Requirements ───────────────────────────────────────────────────

# Minimum duration (seconds) a voice file must have after silence trimming.
# Files shorter than this are rejected before embedding extraction.
# Too-short clips produce unreliable speaker embeddings.
MIN_AUDIO_DURATION_SEC: float = 3.0

# ── Similarity Thresholds ─────────────────────────────────────────────────────
# These apply to the FUSION SCORE (weighted combo of Euclidean + PLDA).
# All comparisons use L2-normalised embeddings.
# Fusion scores live in [0, 1] (higher = more similar).

# REGISTRATION: how similar a new voice must be to an existing one to block it.
# Higher → stricter (0.90 → almost identical required to flag as duplicate).
# Recommended range: 0.80 – 0.95
REGISTER_THRESHOLD: float = 0.90

# IDENTIFICATION: minimum fusion score to confidently name a speaker.
# Lower → more permissive (accepts weaker matches as a known speaker).
# Recommended range: 0.70 – 0.85
PREDICT_THRESHOLD: float = 0.80

# ── Uncertainty Band ───────────────────────────────────────────────────────────
# Matches whose fusion score falls in [PREDICT_THRESHOLD, PREDICT_THRESHOLD + UNCERTAIN_BAND]
# are reported as "uncertain" and accompanied by an explanation.
UNCERTAIN_BAND: float = 0.10

# Derived convenience tuple — used by explain_match()
CONFIDENCE_EXPLAIN_RANGE: tuple = (PREDICT_THRESHOLD, PREDICT_THRESHOLD + UNCERTAIN_BAND)

# ── Multi-Metric Fusion Weights ───────────────────────────────────────────────
# Controls how much each metric contributes to the final fusion score.
# Must sum to 1.0.
#
#   euclidean — L2-distance-based score [0,1]; provides strong magnitude
#               separation in the high-similarity region.
#   plda      — PLDA-inspired logistic calibration; maps Euclidean score to a
#               calibrated probabilistic score with a sharp decision boundary.
#               Dramatically improves discrimination near the threshold.
#
# 50/50 weighting: Euclidean's raw accuracy + PLDA's calibrated sharpness.
FUSION_WEIGHTS: dict = {
    "euclidean": 0.50,
    "plda":      0.50,
}

# ── PLDA Logistic Calibration Parameters ──────────────────────────────────────
# Controls the sigmoid transform:  score = σ(PLDA_SCALE * (euc_score - PLDA_SHIFT))
#
# PLDA_SCALE : steepness of the sigmoid.  Higher = sharper transition.
#              15–25 works well for WavLM x-vectors where same-speaker
#              Euclidean scores typically fall in [0.85, 0.98].
#
# PLDA_SHIFT : the Euclidean score at which the sigmoid output = 0.5.
#              This acts as the "natural" decision boundary of the calibration.
#              Set it near your expected same/different speaker crossover point.
#              For WavLM L2-normalised embeddings, ~0.85 is a good starting point.
#              (Corresponds to cosine ~ 0.70 via the relationship euc = (1+cos)/2)
PLDA_SCALE: float = 20.0
PLDA_SHIFT: float = 0.85

# ── Reproducibility ────────────────────────────────────────────────────────────
# Fixed random seed for numpy / sklearn operations.  Change only deliberately.
RANDOM_SEED: int = 42

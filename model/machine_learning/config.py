
# ── Audio Input Requirements ───────────────────────────────────────────────────

# Minimum duration (seconds) a voice file must have after silence trimming.
# Files shorter than this are rejected before embedding extraction.
# Too-short clips produce unreliable speaker embeddings.
MIN_AUDIO_DURATION_SEC: float = 3.0

# ── Cosine Similarity Thresholds ──────────────────────────────────────────────
# Tune these values to control strictness of duplicate detection and identification.
# All comparisons use L2-normalised embeddings so values live in [-1, 1].

# REGISTRATION: how similar a new voice must be to an existing one to block it.
# Higher → stricter (0.95 → almost identical required to flag as duplicate).
# Recommended range: 0.85 – 0.95
REGISTER_THRESHOLD: float = 0.90

# IDENTIFICATION: minimum similarity to confidently name a speaker.
# Lower → more permissive (accepts weaker matches as a known speaker).
# Recommended range: 0.70 – 0.85  (looser than REGISTER to handle real-world variation)
PREDICT_THRESHOLD: float = 0.90

# ── Uncertainty Band ───────────────────────────────────────────────────────────
# Matches whose cosine similarity falls in [PREDICT_THRESHOLD, PREDICT_THRESHOLD + UNCERTAIN_BAND]
# are reported as "uncertain" and accompanied by an explanation (Rule 8).
UNCERTAIN_BAND: float = 0.10

# Derived convenience tuple — used by explain_match()
CONFIDENCE_EXPLAIN_RANGE: tuple = (PREDICT_THRESHOLD, PREDICT_THRESHOLD + UNCERTAIN_BAND)

# ── Reproducibility ────────────────────────────────────────────────────────────
# Fixed random seed for numpy / sklearn operations.  Change only deliberately.
RANDOM_SEED: int = 42


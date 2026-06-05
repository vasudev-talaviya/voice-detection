import librosa
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic thresholds for the human-speech presence guard (Rules 1 & 2).
# These are intentionally conservative — the goal is to reject pure silence,
# music, and broadband noise, NOT to do full VAD.
# ──────────────────────────────────────────────────────────────────────────────
_SPEECH_ENERGY_FLOOR   = 1e-4   # RMS energy below this → treat as silence
_ZCR_MAX               = 0.35   # zero-crossing rate above this → likely noise/music
_ZCR_MIN               = 0.01   # zero-crossing rate below this → likely silence
_SPEECH_RATIO_MIN      = 0.10   # fraction of frames that must pass the energy test


def is_speech_present(audio: np.ndarray, sr: int = 16000) -> bool:
    """
    Lightweight heuristic to confirm human speech is present in an audio array.

    Rejects:
      • Pure silence      (RMS energy too low throughout)
      • Broadband noise   (uniformly high zero-crossing rate without energy structure)
      • Tonal music       (narrow energy bands with no voiced-speech ZCR pattern)

    This is NOT a full VAD — it is a fast gating test run before expensive
    embedding extraction (Rules 1 and 2).

    Args:
        audio : np.ndarray — 1-D float32 audio waveform at `sr`.
        sr    : int        — sample rate (default 16 000 Hz).

    Returns:
        bool — True if speech is likely present, False otherwise.
    """
    if len(audio) == 0:
        return False

    frame_length = int(0.025 * sr)   # 25 ms frames
    hop_length   = int(0.010 * sr)   # 10 ms hop

    # ── RMS energy per frame ──────────────────────────────────────────────────
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    voiced_frames = rms > _SPEECH_ENERGY_FLOOR
    speech_ratio  = voiced_frames.mean()

    if speech_ratio < _SPEECH_RATIO_MIN:
        return False   # nearly all frames are silent

    # ── Zero-Crossing Rate (ZCR) on voiced frames only ───────────────────────
    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    voiced_zcr = zcr[voiced_frames]
    mean_zcr   = float(voiced_zcr.mean()) if len(voiced_zcr) else 0.0

    # Human speech ZCR sits between silence and broadband noise
    if mean_zcr < _ZCR_MIN or mean_zcr > _ZCR_MAX:
        return False

    return True


def audio_file(filelocation: str, pre_emphasis: float = 0.97, top_db: int = 30) -> np.ndarray:
    """
    Reads, normalizes, and filters audio to prepare it for high-accuracy speaker verification.

    Pipeline:
      1. Load as 16 kHz mono via librosa.
      2. Peak normalisation — scale to unit dynamic range.
      3. Pre-emphasis high-pass filter — boosts high-frequency speaker formants.
      4. Robust silence removal via librosa.effects.trim.
      5. Human-speech presence guard (Rules 1 & 2) — raises ValueError for
         silence, music, or broadband noise.

    Args:
        filelocation  (str)   : Path to the audio file.
        pre_emphasis  (float) : Coefficient for pre-emphasis high-pass filter (0.97 standard).
        top_db        (int)   : Silence-trim threshold in dB below peak.

    Returns:
        np.ndarray: Clean, normalised, pre-emphasised audio array at 16 kHz mono.

    Raises:
        ValueError: If the file contains no usable human speech.
    """
    # 1. Load audio as 16 kHz mono
    audio, sr = librosa.load(filelocation, sr=16000, mono=True)

    if len(audio) == 0:
        return audio

    # 2. Peak Normalisation: scale audio to standard dynamic range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # 3. Pre-Emphasis Filtering: enhances high-frequency speaker-specific formants
    #    Formula: y[t] = x[t] - alpha * x[t-1]
    if pre_emphasis > 0:
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # 4. Robust Silence Removal: strip quiet parts while keeping speech intact
    audio, _ = librosa.effects.trim(audio, top_db=top_db)

    # 5. Human-speech presence guard (Rules 1 & 2)
    if not is_speech_present(audio, sr=sr):
        raise ValueError(
            f"No human speech detected in '{filelocation}'. "
            "The file appears to contain silence, music, or environmental noise only. "
            "Please provide an audio file with clear human speech."
        )

    return audio

import warnings
# Suppress PyTorch's mismatched key_padding_mask and attn_mask deprecation warnings in WavLM
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask is deprecated")

import torch
import torch.nn.functional as F
import numpy as np
from model.pre_train_voice_embedding_model import feature_extractor, model
from model.machine_learning.config import MIN_AUDIO_DURATION_SEC
from helper.audio_read import audio_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# Why WavLMForXVector + outputs.embeddings?
#
# WavLMForXVector (microsoft/wavlm-base-plus-sv) is fine-tuned on VoxCeleb1+2
# with an AM-Softmax speaker verification objective. Its forward pass:
#
#   Audio waveform
#       ↓  WavLM transformer (12 layers, 768-dim)
#       ↓  Adaptive average temporal pooling
#       ↓  Linear projection
#   outputs.embeddings  →  512-dim x-vector
#       (L2-normalised, trained for cosine-similarity speaker verification)
#
# This is the purpose-built speaker embedding. Using raw hidden states (as we
# did before) gives generic audio similarity — NOT speaker identity.
# ──────────────────────────────────────────────────────────────────────────────


def _extract_chunk_embedding(inputs: dict) -> torch.Tensor:
    """
    Run WavLMForXVector and return the 512-dim x-vector speaker embedding.

    The model internally handles temporal pooling — no manual mean/attention
    pooling is needed here. The x-vector is already L2-normalised by the model.

    Args:
        inputs : dict — feature-extractor output tensors on the target device.

    Returns:
        torch.Tensor  [1, 512]  L2-normalised x-vector.
    """
    outputs = model(**inputs)
    # outputs.embeddings: [1, 512] — speaker x-vector from the SV-trained head
    return F.normalize(outputs.embeddings, p=2, dim=-1)


def voice_to_embedding(voicefile, chunk_size_sec=5.0, hop_size_sec=2.5):
    """
    Extract a 512-dim speaker x-vector embedding from an audio file.

    Uses the fine-tuned x-vector head of WavLMForXVector, which was trained
    with AM-Softmax loss on VoxCeleb1+2 specifically to produce embeddings
    that have:
      • HIGH cosine similarity for the same speaker.
      • LOW cosine similarity for different speakers.

    For recordings longer than one chunk, overlapping chunks are extracted
    and their x-vectors are averaged for a stable global representation.

    Args:
        voicefile      (str)   : Path to the audio file.
        chunk_size_sec (float) : Duration of each sliding-window chunk (seconds).
        hop_size_sec   (float) : Hop between consecutive chunks (seconds).

    Returns:
        torch.Tensor: L2-normalised 512-dim speaker x-vector, shape [1, 512].
    """
    audio = audio_file(voicefile)

    if len(audio) == 0:
        raise ValueError(f"Audio file '{voicefile}' has no usable speech after silence trimming.")

    sample_rate     = 16000
    duration_sec    = len(audio) / sample_rate
    if duration_sec < MIN_AUDIO_DURATION_SEC:
        raise ValueError(
            f"Audio file '{voicefile}' is too short: "
            f"{duration_sec:.2f}s (minimum required: {MIN_AUDIO_DURATION_SEC}s). "
            "Please provide a longer voice sample."
        )

    chunk_len   = int(chunk_size_sec * sample_rate)
    hop_len     = int(hop_size_sec   * sample_rate)

    # Slice audio into overlapping chunks
    chunks = []
    if len(audio) <= chunk_len:
        # Pad short clips to exactly one chunk length
        chunks.append(np.pad(audio, (0, chunk_len - len(audio)), mode='constant'))
    else:
        for start in range(0, len(audio) - chunk_len + 1, hop_len):
            chunks.append(audio[start : start + chunk_len])
        # Capture any trailing remainder as a final overlapping chunk
        if (len(audio) - chunk_len) % hop_len != 0:
            chunks.append(audio[-chunk_len:])

    # Extract 512-dim x-vector per chunk
    chunk_embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            inputs = feature_extractor(chunk, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb    = _extract_chunk_embedding(inputs)
            chunk_embeddings.append(emb.cpu())

    # Average chunk x-vectors → stable global speaker representation
    chunk_tensors = torch.stack(chunk_embeddings)    # [num_chunks, 1, 512]
    global_emb    = chunk_tensors.mean(dim=0)         # [1, 512]

    return F.normalize(global_emb, p=2, dim=-1)

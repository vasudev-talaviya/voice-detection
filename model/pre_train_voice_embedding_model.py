import os
import shutil
from transformers import AutoFeatureExtractor, WavLMForXVector

# ──────────────────────────────────────────────────────────────────────────────
# Model selection
#
# WavLMForXVector (microsoft/wavlm-base-plus-sv) is the purpose-built speaker
# verification variant of WavLM.  It exposes:
#
#   outputs.embeddings  →  512-dim L2-normalised x-vector
#
# This is required by voice_embedding_convert.py.  The base WavLMModel does NOT
# have this attribute and will AttributeError at runtime.
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME       = "microsoft/wavlm-base-plus-sv"
local_model_path = os.path.join(os.path.dirname(__file__), "wavlm-base-plus-sv-local")


def _local_cache_is_valid(path: str) -> bool:
    """
    Returns True only if the local cache contains a WavLMForXVector config
    (i.e. was not accidentally saved from WavLMModel).
    """
    config_file = os.path.join(path, "config.json")
    if not os.path.exists(config_file):
        return False
    import json
    with open(config_file, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    # WavLMForXVector stores "WavLMForXVector" in architectures list
    architectures = cfg.get("architectures", [])
    return any("XVector" in a for a in architectures)


if os.path.exists(local_model_path):
    if _local_cache_is_valid(local_model_path):
        print(f"[Model] Loading {MODEL_NAME} from local cache …")
        feature_extractor = AutoFeatureExtractor.from_pretrained(local_model_path)
        model             = WavLMForXVector.from_pretrained(local_model_path)
    else:
        print(
            f"[Model] Local cache at '{local_model_path}' is from a different model class "
            "(WavLMModel, not WavLMForXVector). Removing stale cache and re-downloading …"
        )
        shutil.rmtree(local_model_path)
        print(f"[Model] Downloading {MODEL_NAME} from HuggingFace …")
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        model             = WavLMForXVector.from_pretrained(MODEL_NAME)
        print(f"[Model] Saving to '{local_model_path}' for future offline use …")
        feature_extractor.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)
        print("[Model] Model saved locally successfully.")
else:
    print(f"[Model] Downloading {MODEL_NAME} from HuggingFace …")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model             = WavLMForXVector.from_pretrained(MODEL_NAME)
    print(f"[Model] Saving to '{local_model_path}' for future offline use …")
    feature_extractor.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print("[Model] Model saved locally successfully.")
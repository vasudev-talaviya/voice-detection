import os
import sys
import unittest
import numpy as np
import torch

# Add workspace directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.machine_learning.similarity import compare_embeddings, similarity_to_confidence, explain_match
from helper.audio_read import is_speech_present


class TestAntigravityAudioIntel(unittest.TestCase):
    def test_speech_present_on_noise(self):
        # Generate pure white noise (should fail speech guard)
        sr = 16000
        duration = 5.0
        noise = np.random.normal(0, 0.1, int(sr * duration))
        self.assertFalse(is_speech_present(noise, sr))

    def test_similarity_confidence_mapping(self):
        # Test extreme and normal bounds of similarity to confidence mapping
        self.assertEqual(similarity_to_confidence(1.0), 1.0)
        self.assertEqual(similarity_to_confidence(-1.0), 0.0)
        self.assertEqual(similarity_to_confidence(0.0), 0.5)
        self.assertTrue(0.0 <= similarity_to_confidence(0.75) <= 1.0)

    def test_explain_match(self):
        # Match outside uncertain band (sim >= 0.85) -> no explanation
        self.assertEqual(explain_match(0.92, "Alice"), "")
        # Match below prediction threshold -> should suggest registering or longer sample
        explanation_fail = explain_match(0.50, "Alice")
        self.assertIn("not recognised", explanation_fail)
        # Match in uncertain band (0.75 <= sim < 0.85) -> should return explanation
        explanation_uncertain = explain_match(0.78, "Alice")
        self.assertIn("Uncertain match to 'Alice'", explanation_uncertain)

    def test_compare_embeddings(self):
        # Generate dummy 512-dim normalized vectors
        emb_a = torch.randn(1, 512)
        emb_a = torch.nn.functional.normalize(emb_a, p=2, dim=-1)
        
        # Test identical embedding
        result_same = compare_embeddings(emb_a, emb_a)
        self.assertTrue(result_same["same_speaker"])
        self.assertAlmostEqual(result_same["similarity"], 1.0, places=4)
        self.assertEqual(result_same["explanation"], "")


if __name__ == "__main__":
    unittest.main()

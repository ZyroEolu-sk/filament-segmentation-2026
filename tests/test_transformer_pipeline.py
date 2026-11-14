"""Check native-resolution tiling and opt-in augmentation without using a GPU."""
import unittest
import numpy as np
import torch
from filaments.infer import predict, predict_tiled
from filaments.data import CropGenerator


class Pointwise(torch.nn.Module):
    def forward(self, x):
        return torch.cat([x * 4 - 2, -x * 3 + 1], dim=1)


class TransformerPipelineTests(unittest.TestCase):
    def test_tiling_preserves_pixels_and_inverts_tta(self):
        # Asymmetric, non-divisible size exercises edges and TTA alignment.
        im = np.random.default_rng(7).integers(0, 256, (93, 117), dtype=np.uint8)
        model = Pointwise()
        expected = predict(model, im, tta=False, device="cpu")
        for tta in (True, False):
            actual = predict_tiled(model, im, tile=64, overlap=23, tta=tta)
            np.testing.assert_allclose(actual, expected, atol=5e-4)
        model.inference_tile, model.inference_overlap = 64, 23
        np.testing.assert_allclose(predict(model, im, device="cpu"), expected, atol=5e-4)

    def test_small_images_and_invalid_tiles(self):
        im = np.zeros((19, 35), np.uint8)
        out = predict_tiled(Pointwise(), im, tile=64, overlap=16)
        self.assertEqual(out.shape, (2, 19, 35))
        self.assertTrue(np.isfinite(out).all())
        for tile, overlap in ((0, 0), (65, 16), (64, 64), (64, -1)):
            with self.assertRaises(ValueError):
                predict_tiled(Pointwise(), im, tile, overlap)

    def test_illumination_is_bounded_smooth_and_deterministic(self):
        im = np.full((64, 64), .6, dtype=np.float32)
        for seed in range(20):
            result = CropGenerator._solar_illumination(im, np.random.default_rng(seed))
            repeat = CropGenerator._solar_illumination(im, np.random.default_rng(seed))
            np.testing.assert_array_equal(result, repeat)
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(0 <= result.min() <= result.max() <= 1)
            self.assertLess(np.abs(np.diff(result, axis=0)).max(), .01)


if __name__ == "__main__":
    unittest.main()

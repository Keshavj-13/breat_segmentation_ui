"""
3D Patch-Based Occlusion Sensitivity for volumetric segmentation models.

Slides a 3D patch (cube) across the input volume, masks each patch,
re-runs inference, and measures the change in tumour prediction to build
a volumetric importance map.
"""

import torch
import torch.nn.functional as F
import numpy as np
from itertools import product


class PatchOcclusion3D:
    """
    Compute a 3D occlusion-sensitivity / importance volume.

    Usage:
        occ = PatchOcclusion3D(model, patch_size=16, stride=8)
        importance = occ.generate(input_volume, baseline_pred)
        # importance: np.ndarray [D,H,W] in [0,1]
    """

    def __init__(self, model, patch_size=16, stride=8, mask_value="mean",
                 fast_mode=False):
        """
        Args:
            model: 3D segmentation model in eval mode.
            patch_size: int or tuple(d,h,w). Size of the occluding cube.
            stride: int or tuple(d,h,w). Step between patches.
            mask_value: 'zero' | 'mean' | float. Value to fill occluded region.
            fast_mode: If True, uses larger stride (2×) for faster evaluation.
        """
        self.model = model
        self.patch_size = self._to_tuple(patch_size)
        self.stride = self._to_tuple(stride)
        self.mask_value = mask_value
        if fast_mode:
            self.stride = tuple(s * 2 for s in self.stride)

    @staticmethod
    def _to_tuple(v):
        return (v, v, v) if isinstance(v, int) else tuple(v)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_tensor, baseline_pred=None, batch_size=4,
                 metric="mean_prob"):
        """
        Build a 3D importance volume.

        Args:
            input_tensor: [1, C, D, H, W] tensor on model device.
            baseline_pred: Optional pre-computed baseline prediction tensor.
                           If None, a forward pass is run first.
            batch_size: Number of occluded volumes evaluated per GPU batch.
            metric: 'mean_prob' – mean sigmoid prob inside predicted tumour.
                    'total_prob' – sum of sigmoid probabilities.

        Returns:
            importance_np: np.ndarray [D, H, W] normalised [0, 1].
        """
        device = input_tensor.device
        D, H, W = input_tensor.shape[2:]

        # Baseline prediction (on padded tensor to avoid skip-connection mismatch)
        if baseline_pred is None:
            baseline_out = self.model(input_tensor)
            baseline_pred = torch.sigmoid(baseline_out)
        baseline_score = self._score(baseline_pred, metric)

        # Resolve mask fill value
        if self.mask_value == "zero":
            fill = 0.0
        elif self.mask_value == "mean":
            fill = float(input_tensor.mean())
        else:
            fill = float(self.mask_value)

        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        # Compute patch centre positions
        positions = list(product(
            range(0, max(D - pd + 1, 1), sd),
            range(0, max(H - ph + 1, 1), sh),
            range(0, max(W - pw + 1, 1), sw),
        ))

        # Accumulator arrays (for overlapping patches)
        importance_sum = np.zeros((D, H, W), dtype=np.float64)
        importance_cnt = np.zeros((D, H, W), dtype=np.float64)

        # Batch evaluation
        for batch_start in range(0, len(positions), batch_size):
            batch_positions = positions[batch_start:batch_start + batch_size]
            occluded_batch = []

            for (d0, h0, w0) in batch_positions:
                masked = input_tensor.clone()
                masked[:, :, d0:d0+pd, h0:h0+ph, w0:w0+pw] = fill
                occluded_batch.append(masked)

            # Stack into a single batch [B, C, D, H, W]
            occluded_tensor = torch.cat(occluded_batch, dim=0)
            occluded_out = self.model(occluded_tensor)
            occluded_probs = torch.sigmoid(occluded_out)

            for i, (d0, h0, w0) in enumerate(batch_positions):
                occ_score = self._score(occluded_probs[i:i+1], metric)
                delta = max(baseline_score - occ_score, 0.0)  # drop = importance

                importance_sum[d0:d0+pd, h0:h0+ph, w0:w0+pw] += delta
                importance_cnt[d0:d0+pd, h0:h0+ph, w0:w0+pw] += 1.0

            # Free batch tensors to keep VRAM bounded
            del occluded_tensor, occluded_out, occluded_probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average overlapping contributions
        importance_cnt[importance_cnt == 0] = 1.0
        importance = importance_sum / importance_cnt

        # Normalise to [0, 1]
        vmin, vmax = importance.min(), importance.max()
        if vmax - vmin > 1e-8:
            importance = (importance - vmin) / (vmax - vmin)
        else:
            importance = np.zeros_like(importance)

        return importance.astype(np.float32)

    # ------------------------------------------------------------------
    # Patch-level detail (for interactive explorer)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def probe_patch(self, input_tensor, d0, h0, w0):
        """
        Evaluate a single patch and return detailed delta information.

        Returns:
            dict with original_prob, masked_prob, delta
        """
        pd, ph, pw = self.patch_size

        # Pad to divisible-by-32 so ResNet50 skip connections match
        D, H, W = input_tensor.shape[2:]
        div = 32
        pad_D = (div - D % div) % div
        pad_H = (div - H % div) % div
        pad_W = (div - W % div) % div
        if pad_D + pad_H + pad_W > 0:
            input_tensor = F.pad(input_tensor, (0, pad_W, 0, pad_H, 0, pad_D), mode="replicate")

        # Baseline
        baseline_out = self.model(input_tensor)
        baseline_prob = float(torch.sigmoid(baseline_out).mean())

        # Masked
        fill = float(input_tensor.mean()) if self.mask_value == "mean" else 0.0
        masked = input_tensor.clone()
        masked[:, :, d0:d0+pd, h0:h0+ph, w0:w0+pw] = fill
        masked_out = self.model(masked)
        masked_prob = float(torch.sigmoid(masked_out).mean())

        return {
            "original_prob": round(baseline_prob, 6),
            "masked_prob": round(masked_prob, 6),
            "delta": round(baseline_prob - masked_prob, 6),
            "patch_origin": [int(d0), int(h0), int(w0)],
            "patch_size": list(self.patch_size),
        }

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _score(pred_probs, metric):
        """Scalar score from a prediction tensor."""
        if metric == "mean_prob":
            return float(pred_probs.mean())
        elif metric == "total_prob":
            return float(pred_probs.sum())
        else:
            return float(pred_probs.mean())

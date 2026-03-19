"""
XAI Integration Runner — orchestrates Grad-CAM and Occlusion analysis
on completed segmentation jobs **without modifying** the existing inference
pipeline or worker.py.

This module:
    - Loads the cached volume + prediction from disk (saved by worker.py)
    - Re-uses the global model in eval mode
    - Attaches hooks dynamically for Grad-CAM
    - Runs occlusion sensitivity
    - Caches all XAI artefacts to disk
    - Computes quantitative metrics
"""

import gc
import os
import numpy as np
import torch
import torch.nn.functional as F

from .gradcam3d import GradCAM3D
from .occlusion3d import PatchOcclusion3D
from .utils import compute_xai_metrics, save_heatmap_nifti


# XAI result cache — keyed by job_id
XAI_CACHE = {}

# Maximum spatial size for XAI forward passes.
# Must be divisible by 32 (ResNet50 stride).  Matches the inference ROI.
_XAI_MAX_DHW = (128, 128, 96)


def _pad_to_divisible(tensor, divisor=32):
    """
    Pad spatial dims [D, H, W] so each is divisible by `divisor`.
    ResNet50 has total stride 32, so the model requires this.
    Returns (padded_tensor, original_shape_DHW).
    """
    D, H, W = tensor.shape[2:]
    pad_D = (divisor - D % divisor) % divisor
    pad_H = (divisor - H % divisor) % divisor
    pad_W = (divisor - W % divisor) % divisor
    if pad_D == pad_H == pad_W == 0:
        return tensor, (D, H, W)
    # F.pad pads from last dim inward: (W₀, W₁, H₀, H₁, D₀, D₁)
    padded = F.pad(tensor, (0, pad_W, 0, pad_H, 0, pad_D), mode="replicate")
    return padded, (D, H, W)


def _prepare_xai_input(input_tensor):
    """
    Downsample the volume to _XAI_MAX_DHW if it exceeds that size, then pad
    to a multiple of 32.  Returns (xai_tensor, orig_dhw, was_downsampled).

    Downsampling prevents CUDA OOM when running full-volume forward passes for
    Grad-CAM / Occlusion (unlike normal inference which uses sliding windows).
    The heatmaps are upsampled back to original size after generation.
    """
    orig_D, orig_H, orig_W = input_tensor.shape[2:]
    max_D, max_H, max_W = _XAI_MAX_DHW

    needs_ds = orig_D > max_D or orig_H > max_H or orig_W > max_W
    if needs_ds:
        ds = F.interpolate(
            input_tensor,
            size=(max_D, max_H, max_W),
            mode="trilinear",
            align_corners=False,
        )
    else:
        ds = input_tensor

    padded, pre_pad_dhw = _pad_to_divisible(ds, divisor=32)
    return padded, pre_pad_dhw, (orig_D, orig_H, orig_W), needs_ds


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_xai_analysis(job_id, model, device, temp_dir,
                     # Grad-CAM options
                     run_gradcam=True,
                     gradcam_target_layer=None,
                     # Occlusion options
                     run_occlusion=True,
                     occlusion_patch_size=16,
                     occlusion_stride=8,
                     occlusion_mask_value="mean",
                     occlusion_batch_size=1,
                     occlusion_fast_mode=False):
    """
    Run XAI analysis on a completed segmentation job.

    Args:
        job_id: UUID of the finished job.
        model: The loaded 3D segmentation model (already on device).
        device: torch.device.
        temp_dir: Path to worker_temp/ where vol.npy / pred.npy live.
        run_gradcam: Whether to compute Grad-CAM.
        run_occlusion: Whether to compute occlusion maps.
        ... other options forwarded to the respective modules.

    Returns:
        dict with xai_status, metrics, and paths to cached artefacts.
    """
    vol_path = os.path.join(temp_dir, f"{job_id}_vol.npy")
    pred_path = os.path.join(temp_dir, f"{job_id}_pred.npy")

    if not os.path.exists(vol_path) or not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Volume/prediction arrays missing for job {job_id}")

    # Load cached arrays
    vol_np = np.load(vol_path)      # [D, H, W]
    pred_np = np.load(pred_path)    # [D, H, W]

    # Prepare input tensor  [1, 1, D, H, W]
    input_tensor = (
        torch.from_numpy(vol_np)
        .unsqueeze(0).unsqueeze(0)
        .float()
        .to(device)
    )
    orig_vol_D, orig_vol_H, orig_vol_W = input_tensor.shape[2:]

    result = {
        "job_id": job_id,
        "xai_status": "PROCESSING",
        "gradcam_available": False,
        "occlusion_available": False,
        "metrics": {},
    }

    heatmap_3d = None
    occlusion_3d = None

    # Downsample to XAI ROI cap and pad to multiple of 32.
    # Full-volume forward passes OOM on large MRI — this caps VRAM usage.
    # Heatmaps are upsampled back to original spatial size after generation.
    input_xai, pre_pad_dhw, orig_full_dhw, was_downsampled = _prepare_xai_input(input_tensor)
    pre_D, pre_H, pre_W = pre_pad_dhw      # size after downsample, before pad

    # ---- Grad-CAM ----
    if run_gradcam:
        _free_cuda()
        was_training = model.training
        model.eval()

        # Temporarily enable grad
        for p in model.parameters():
            p.requires_grad_(True)

        gcam = GradCAM3D(model, target_layer=gradcam_target_layer)
        try:
            # generate() returns array at input_xai spatial size
            heatmap_xai = gcam.generate(input_xai)          # [D_xai, H_xai, W_xai]
            heatmap_cropped = heatmap_xai[:pre_D, :pre_H, :pre_W]  # strip padding

            # Upsample back to original volume size if we downsampled
            if was_downsampled:
                t = torch.from_numpy(heatmap_cropped).unsqueeze(0).unsqueeze(0).float()
                t = F.interpolate(
                    t,
                    size=(orig_vol_D, orig_vol_H, orig_vol_W),
                    mode="trilinear",
                    align_corners=False,
                )
                heatmap_3d = t.squeeze().cpu().numpy()
            else:
                heatmap_3d = heatmap_cropped

            cam_path = os.path.join(temp_dir, f"{job_id}_gradcam.npy")
            np.save(cam_path, heatmap_3d)
            result["gradcam_available"] = True
        finally:
            gcam.remove_hooks()
            for p in model.parameters():
                p.requires_grad_(False)
            if was_training:
                model.train()
            _free_cuda()

    # ---- Occlusion ----
    if run_occlusion:
        _free_cuda()
        model.eval()
        occ = PatchOcclusion3D(
            model,
            patch_size=occlusion_patch_size,
            stride=occlusion_stride,
            mask_value=occlusion_mask_value,
            fast_mode=occlusion_fast_mode,
        )
        # Run on downsampled+padded input to keep VRAM bounded
        occ_xai = occ.generate(input_xai, batch_size=occlusion_batch_size)
        occ_cropped = occ_xai[:pre_D, :pre_H, :pre_W]  # strip padding

        # Upsample back to original volume size if we downsampled
        if was_downsampled:
            t = torch.from_numpy(occ_cropped).unsqueeze(0).unsqueeze(0).float()
            t = F.interpolate(
                t,
                size=(orig_vol_D, orig_vol_H, orig_vol_W),
                mode="trilinear",
                align_corners=False,
            )
            occlusion_3d = t.squeeze().cpu().numpy()
        else:
            occlusion_3d = occ_cropped

        occ_path = os.path.join(temp_dir, f"{job_id}_occlusion.npy")
        np.save(occ_path, occlusion_3d)
        result["occlusion_available"] = True
        _free_cuda()

    # ---- Metrics ----
    metrics = compute_xai_metrics(heatmap_3d, pred_np, occlusion_3d)
    result["metrics"] = metrics

    # ---- NIfTI export (if original nii exists) ----
    nii_candidates = [
        f for f in os.listdir(temp_dir)
        if f.startswith(job_id) and f.endswith((".nii", ".nii.gz"))
           and "_mask" not in f and "_gradcam" not in f and "_occlusion" not in f
    ]
    if nii_candidates:
        ref_path = os.path.join(temp_dir, nii_candidates[0])
        if heatmap_3d is not None:
            save_heatmap_nifti(
                heatmap_3d, ref_path,
                os.path.join(temp_dir, f"{job_id}_gradcam.nii.gz"),
            )
        if occlusion_3d is not None:
            save_heatmap_nifti(
                occlusion_3d, ref_path,
                os.path.join(temp_dir, f"{job_id}_occlusion.nii.gz"),
            )

    result["xai_status"] = "DONE"

    # Cache result
    XAI_CACHE[job_id] = result
    return result


def get_cached_xai(job_id):
    """Return cached XAI result or None."""
    return XAI_CACHE.get(job_id)

"""
XAI API Router — mounts onto the existing FastAPI worker app.

Provides endpoints for:
    - Triggering XAI analysis (Grad-CAM + Occlusion)
    - Querying XAI status / results / metrics
    - Rendering XAI-overlaid slices
    - Downloading XAI NIfTI / heatmap volumes
    - Probing individual occlusion patches
"""

import os
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response, FileResponse, JSONResponse
import io

from .xai_runner import run_xai_analysis, get_cached_xai, XAI_CACHE
from .utils import render_overlay_slice, get_cmap
from .occlusion3d import PatchOcclusion3D

import torch

router = APIRouter(prefix="/xai", tags=["XAI"])

# These will be injected by the mount script
_model = None
_device = None
_temp_dir = None
_jobs_ref = None  # reference to worker's JOBS dict


def init_xai_router(model, device, temp_dir, jobs_dict):
    """Called once at startup to inject references from worker.py globals."""
    global _model, _device, _temp_dir, _jobs_ref
    _model = model
    _device = device
    _temp_dir = temp_dir
    _jobs_ref = jobs_dict


# ---------------------------------------------------------------
# Trigger XAI analysis
# ---------------------------------------------------------------
@router.post("/analyze/{job_id}")
async def trigger_xai(
    job_id: str,
    background_tasks: BackgroundTasks,
    run_gradcam: bool = Query(True),
    run_occlusion: bool = Query(True),
    occlusion_patch_size: int = Query(16),
    occlusion_stride: int = Query(8),
    occlusion_fast_mode: bool = Query(False),
    occlusion_batch_size: int = Query(4),
):
    if _model is None:
        raise HTTPException(500, "Model not loaded")
    if _jobs_ref is None or job_id not in _jobs_ref:
        raise HTTPException(404, "Job not found")
    if _jobs_ref[job_id].get("status") != "DONE":
        raise HTTPException(400, "Segmentation not complete yet")

    # Check if already running / done
    cached = get_cached_xai(job_id)
    if cached and cached.get("xai_status") == "PROCESSING":
        return {"job_id": job_id, "xai_status": "PROCESSING"}

    # Mark as processing
    XAI_CACHE[job_id] = {"job_id": job_id, "xai_status": "PROCESSING"}

    def _run():
        try:
            run_xai_analysis(
                job_id=job_id,
                model=_model,
                device=_device,
                temp_dir=_temp_dir,
                run_gradcam=run_gradcam,
                run_occlusion=run_occlusion,
                occlusion_patch_size=occlusion_patch_size,
                occlusion_stride=occlusion_stride,
                occlusion_batch_size=occlusion_batch_size,
                occlusion_fast_mode=occlusion_fast_mode,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            XAI_CACHE[job_id] = {
                "job_id": job_id,
                "xai_status": "FAILED",
                "error": str(e),
            }

    background_tasks.add_task(_run)
    return {"job_id": job_id, "xai_status": "QUEUED"}


# ---------------------------------------------------------------
# Status / Metrics
# ---------------------------------------------------------------
@router.get("/status/{job_id}")
def xai_status(job_id: str):
    cached = get_cached_xai(job_id)
    if not cached:
        return {"job_id": job_id, "xai_status": "NOT_STARTED"}
    return cached


@router.get("/metrics/{job_id}")
def xai_metrics(job_id: str):
    cached = get_cached_xai(job_id)
    if not cached or cached.get("xai_status") != "DONE":
        raise HTTPException(400, "XAI analysis not complete")
    return cached.get("metrics", {})


# ---------------------------------------------------------------
# Slice rendering with XAI overlays
# ---------------------------------------------------------------
@router.get("/slice/{job_id}")
def xai_slice(
    job_id: str,
    plane: str = Query("axial"),
    index: int = Query(0),
    show_mask: bool = Query(True),
    show_gradcam: bool = Query(True),
    show_occlusion: bool = Query(False),
    mask_alpha: float = Query(0.35),
    gradcam_alpha: float = Query(0.5),
    occlusion_alpha: float = Query(0.5),
    gradcam_cmap: str = Query("jet"),
    occlusion_cmap: str = Query("hot"),
):
    """Render a slice with multi-layer XAI overlays."""
    vol_path = os.path.join(_temp_dir, f"{job_id}_vol.npy")
    pred_path = os.path.join(_temp_dir, f"{job_id}_pred.npy")
    cam_path = os.path.join(_temp_dir, f"{job_id}_gradcam.npy")
    occ_path = os.path.join(_temp_dir, f"{job_id}_occlusion.npy")

    if not os.path.exists(vol_path):
        raise HTTPException(404, "Volume not found")

    vol = np.load(vol_path, mmap_mode="r")
    pred = np.load(pred_path, mmap_mode="r") if os.path.exists(pred_path) else None
    cam = np.load(cam_path, mmap_mode="r") if os.path.exists(cam_path) else None
    occ = np.load(occ_path, mmap_mode="r") if os.path.exists(occ_path) else None

    # Extract slice
    def _slice(arr, plane, idx):
        if arr is None:
            return None
        if plane == "axial":
            idx = max(0, min(idx, arr.shape[0] - 1))
            return arr[idx, :, :]
        elif plane == "coronal":
            idx = max(0, min(idx, arr.shape[1] - 1))
            return arr[:, idx, :]
        elif plane == "sagittal":
            idx = max(0, min(idx, arr.shape[2] - 1))
            return arr[:, :, idx]
        raise HTTPException(400, "Invalid plane")

    vol_s = _slice(vol, plane, index)
    pred_s = _slice(pred, plane, index)
    cam_s = _slice(cam, plane, index)
    occ_s = _slice(occ, plane, index)

    png_bytes = render_overlay_slice(
        volume_slice=vol_s,
        mask_slice=pred_s,
        heatmap_slice=cam_s,
        occlusion_slice=occ_s,
        mask_alpha=mask_alpha,
        heatmap_alpha=gradcam_alpha,
        occlusion_alpha=occlusion_alpha,
        heatmap_cmap=gradcam_cmap,
        occlusion_cmap=occlusion_cmap,
        show_mask=show_mask,
        show_heatmap=show_gradcam,
        show_occlusion=show_occlusion,
    )

    return Response(content=png_bytes, media_type="image/png")


# ---------------------------------------------------------------
# 3D volume data (for frontend 3D rendering)
# ---------------------------------------------------------------
@router.get("/volume/{job_id}/{kind}")
def xai_volume_data(job_id: str, kind: str):
    """
    Return raw float32 volume data as binary for frontend 3D rendering.
    kind: 'gradcam' | 'occlusion'
    """
    if kind not in ("gradcam", "occlusion"):
        raise HTTPException(400, "kind must be 'gradcam' or 'occlusion'")

    path = os.path.join(_temp_dir, f"{job_id}_{kind}.npy")
    if not os.path.exists(path):
        raise HTTPException(404, f"{kind} volume not available")

    arr = np.load(path).astype(np.float32)
    # Return shape metadata in headers, binary body
    buf = io.BytesIO()
    buf.write(arr.tobytes())
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Shape": ",".join(str(s) for s in arr.shape),
            "X-Dtype": "float32",
        },
    )


# ---------------------------------------------------------------
# NIfTI download
# ---------------------------------------------------------------
@router.get("/download/{job_id}/{kind}")
def xai_download_nifti(job_id: str, kind: str):
    """Download Grad-CAM or occlusion heatmap as NIfTI."""
    if kind not in ("gradcam", "occlusion"):
        raise HTTPException(400, "kind must be 'gradcam' or 'occlusion'")
    path = os.path.join(_temp_dir, f"{job_id}_{kind}.nii.gz")
    if not os.path.exists(path):
        raise HTTPException(404, f"{kind} NIfTI not available")
    return FileResponse(path, media_type="application/gzip",
                        filename=f"{job_id}_{kind}.nii.gz")


# ---------------------------------------------------------------
# Patch probe (interactive occlusion explorer)
# ---------------------------------------------------------------
@router.get("/probe/{job_id}")
def probe_patch(
    job_id: str,
    d: int = Query(0),
    h: int = Query(0),
    w: int = Query(0),
    patch_size: int = Query(16),
):
    """Probe a single occlusion patch at given coordinates."""
    if _model is None:
        raise HTTPException(500, "Model not loaded")
    vol_path = os.path.join(_temp_dir, f"{job_id}_vol.npy")
    if not os.path.exists(vol_path):
        raise HTTPException(404, "Volume not found")

    vol_np = np.load(vol_path)
    input_tensor = (
        torch.from_numpy(vol_np)
        .unsqueeze(0).unsqueeze(0)
        .float()
        .to(_device)
    )

    occ = PatchOcclusion3D(_model, patch_size=patch_size, stride=patch_size)
    info = occ.probe_patch(input_tensor, d, h, w)
    return info


# ---------------------------------------------------------------
# Available colormaps
# ---------------------------------------------------------------
@router.get("/colormaps")
def list_colormaps():
    return {"colormaps": ["jet", "hot", "plasma", "inferno", "viridis", "medical"]}

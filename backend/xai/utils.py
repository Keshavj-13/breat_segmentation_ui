"""
XAI utility functions for 3D medical image analysis.

Provides:
    - Heatmap-to-slice rendering (with colormaps)
    - Overlay compositing
    - Metric computation (Grad-CAM inside/outside tumour)
    - NIfTI / PNG export helpers
    - Colormap management
"""

import io
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------
# Supported colormaps
# ---------------------------------------------------------------
COLORMAPS = {
    "jet": "jet",
    "hot": "hot",
    "plasma": "plasma",
    "inferno": "inferno",
    "viridis": "viridis",
    "medical": None,  # custom – built below
}


def _build_medical_cmap():
    """Dark-blue → cyan → yellow clinical colormap."""
    colors = [
        (0.0, "#000033"),
        (0.25, "#003366"),
        (0.5, "#06b6d4"),
        (0.75, "#fbbf24"),
        (1.0, "#ef4444"),
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        "medical", [(v, c) for v, c in colors], N=256
    )


COLORMAPS["medical"] = _build_medical_cmap()


def get_cmap(name="jet"):
    cm = COLORMAPS.get(name, "jet")
    if isinstance(cm, str):
        return plt.get_cmap(cm)
    return cm


# ---------------------------------------------------------------
# Slice rendering
# ---------------------------------------------------------------
def render_heatmap_slice(volume_slice, heatmap_slice, cmap_name="jet",
                         alpha=0.5, dpi=100):
    """
    Render a 2D image slice with heatmap overlay → PNG bytes.

    Args:
        volume_slice: 2D np.ndarray (grayscale MRI slice).
        heatmap_slice: 2D np.ndarray in [0,1] (Grad-CAM or occlusion).
        cmap_name: Colormap name.
        alpha: Overlay opacity.

    Returns:
        bytes (PNG image).
    """
    cmap = get_cmap(cmap_name)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
    ax.imshow(np.rot90(volume_slice), cmap="gray", aspect="equal")
    ax.imshow(np.rot90(heatmap_slice), cmap=cmap, alpha=alpha,
              vmin=0, vmax=1, aspect="equal")
    ax.axis("off")
    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_overlay_slice(volume_slice, mask_slice, heatmap_slice=None,
                          occlusion_slice=None, mask_alpha=0.35,
                          heatmap_alpha=0.5, occlusion_alpha=0.5,
                          heatmap_cmap="jet", occlusion_cmap="hot",
                          show_mask=True, show_heatmap=True,
                          show_occlusion=True, dpi=100):
    """
    Composite overlay with optional mask + Grad-CAM + occlusion layers.

    Returns:
        bytes (PNG image).
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
    ax.imshow(np.rot90(volume_slice), cmap="gray", aspect="equal")

    if show_mask and mask_slice is not None:
        rgba = np.zeros((*np.rot90(mask_slice).shape, 4))
        rotated_mask = np.rot90(mask_slice)
        rgba[rotated_mask > 0.5] = [1, 0, 0, mask_alpha]
        ax.imshow(rgba, aspect="equal")

    if show_heatmap and heatmap_slice is not None:
        cmap = get_cmap(heatmap_cmap)
        ax.imshow(np.rot90(heatmap_slice), cmap=cmap, alpha=heatmap_alpha,
                  vmin=0, vmax=1, aspect="equal")

    if show_occlusion and occlusion_slice is not None:
        cmap = get_cmap(occlusion_cmap)
        ax.imshow(np.rot90(occlusion_slice), cmap=cmap,
                  alpha=occlusion_alpha, vmin=0, vmax=1, aspect="equal")

    ax.axis("off")
    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def compute_xai_metrics(heatmap_3d, pred_mask_3d, occlusion_3d=None):
    """
    Compute quantitative XAI metrics.

    Args:
        heatmap_3d: Grad-CAM volume [D,H,W] in [0,1].
        pred_mask_3d: Binary segmentation mask [D,H,W].
        occlusion_3d: Optional occlusion importance volume [D,H,W] in [0,1].

    Returns:
        dict with numeric metrics.
    """
    tumor_mask = pred_mask_3d > 0.5
    non_tumor_mask = ~tumor_mask
    tumor_voxels = int(tumor_mask.sum())
    total_voxels = int(pred_mask_3d.size)

    metrics = {
        "tumor_volume_voxels": tumor_voxels,
        "total_voxels": total_voxels,
    }

    # Grad-CAM metrics
    if heatmap_3d is not None:
        if tumor_voxels > 0:
            cam_inside = float(heatmap_3d[tumor_mask].mean())
            cam_inside_pct = float(
                heatmap_3d[tumor_mask].sum() / (heatmap_3d.sum() + 1e-8) * 100
            )
        else:
            cam_inside = 0.0
            cam_inside_pct = 0.0

        non_tumor_count = int(non_tumor_mask.sum())
        cam_outside = float(heatmap_3d[non_tumor_mask].mean()) if non_tumor_count > 0 else 0.0
        cam_outside_pct = 100.0 - cam_inside_pct

        metrics.update({
            "gradcam_mean_inside_tumor": round(cam_inside, 4),
            "gradcam_mean_outside_tumor": round(cam_outside, 4),
            "gradcam_pct_inside_tumor": round(cam_inside_pct, 2),
            "gradcam_pct_outside_tumor": round(cam_outside_pct, 2),
        })

    # Occlusion metrics
    if occlusion_3d is not None:
        if tumor_voxels > 0:
            occ_inside = float(occlusion_3d[tumor_mask].mean())
        else:
            occ_inside = 0.0

        non_tumor_count = int(non_tumor_mask.sum())
        occ_outside = float(occlusion_3d[non_tumor_mask].mean()) if non_tumor_count > 0 else 0.0

        metrics.update({
            "occlusion_mean_inside_tumor": round(occ_inside, 4),
            "occlusion_mean_outside_tumor": round(occ_outside, 4),
            "occlusion_sensitivity_score": round(occ_inside - occ_outside, 4),
        })

    return metrics


# ---------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------
def save_heatmap_nifti(heatmap_3d, reference_nifti_path, output_path):
    """Save a [0,1] heatmap as a NIfTI file using the affine of a reference."""
    ref = nib.load(reference_nifti_path)
    img = nib.Nifti1Image(heatmap_3d.astype(np.float32), affine=ref.affine)
    nib.save(img, output_path)
    return output_path


def save_slices_png(volume_3d, heatmap_3d, output_dir, prefix="slice",
                    axis=0, cmap_name="jet", alpha=0.5):
    """Save every slice along an axis as a PNG with overlay."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    n_slices = volume_3d.shape[axis]
    paths = []
    for i in range(n_slices):
        if axis == 0:
            vs, hs = volume_3d[i], heatmap_3d[i]
        elif axis == 1:
            vs, hs = volume_3d[:, i, :], heatmap_3d[:, i, :]
        else:
            vs, hs = volume_3d[:, :, i], heatmap_3d[:, :, i]
        png_bytes = render_heatmap_slice(vs, hs, cmap_name=cmap_name, alpha=alpha)
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        with open(path, "wb") as f:
            f.write(png_bytes)
        paths.append(path)
    return paths

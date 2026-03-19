import os
import uuid
import asyncio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel
import shutil

# DL Imports
import torch
import torch.nn as nn
from monai.networks.nets import FlexibleUNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader
import nibabel as nib
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = FastAPI(title="Medical Segmentation GPU Worker")

# --- Configuration & Paths ---
MODEL_FILE = "../best_model2020.pth"
TEMP_DIR = "./worker_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Application State
JOBS = {}  # Store job status, meta, and results

# --- Model Definitions (Match Jupyter Notebook Exactly) ---
class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_excitation = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        chn_se = self.channel_excitation(x) * x
        spa_se = self.spatial_excitation(x) * x
        return chn_se + spa_se

class AttentionAllDecoderUNet(FlexibleUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        decoder_channels = kwargs.get("decoder_channels", (512, 256, 128, 64, 32))
        self.attention_blocks = nn.ModuleList([
            SCSEBlock(ch) for ch in decoder_channels
        ])

    def forward(self, x):
        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        x = features[-1]

        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            x = self.attention_blocks[i](x)

        x = self.segmentation_head(x)
        return x

def get_model(device):
    model = AttentionAllDecoderUNet(
        in_channels=1,
        out_channels=1,
        backbone="resnet50",
        pretrained=False,
        decoder_channels=(512, 256, 128, 64, 32),
        spatial_dims=3,
        norm=('instance', {'affine': True}),
        act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
        dropout=0.2,
        decoder_bias=False,
        upsample='deconv',
        interp_mode='trilinear',
        is_pad=False
    ).to(device)
    return model

def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=0.0, a_max=1555.0, 
            b_min=0.0, b_max=1.0, 
            clip=True
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"])
    ])

# --- Global Model Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on: {device}")
try:
    global_model = get_model(device)
    state = torch.load(MODEL_FILE, map_location=device)
    if "state_dict" in state:
        global_model.load_state_dict(state["state_dict"])
    else:
        global_model.load_state_dict(state)
    global_model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load model weights on startup: {e}")
    global_model = None

# --- Background Processing Logic ---
def run_inference(job_id: str, file_path: str, threshold: float, overlap: float, sw_batch_size: int):
    global global_model, device
    try:
        JOBS[job_id]["status"] = "PROCESSING"
        
        if global_model is None:
            raise RuntimeError("Model weight loading failed. Cannot infer.")

        ds = Dataset(data=[{"image": file_path}], transform=get_transforms())
        loader = DataLoader(ds, batch_size=1, num_workers=0)

        print(f"[{job_id}] Starting inference...")
        with torch.no_grad():
            for batch in loader:
                inputs = batch["image"].to(device)
                
                outputs = sliding_window_inference(
                    inputs, 
                    roi_size=(128, 128, 96), 
                    sw_batch_size=sw_batch_size, 
                    predictor=global_model,
                    overlap=overlap,
                    mode="gaussian"
                )
                
                preds = (torch.sigmoid(outputs) > threshold).float()
                
                img_np = inputs[0, 0].cpu().numpy()
                pred_np = preds[0, 0].cpu().numpy()

                # Save Mask NIfTI
                mask_nifti_path = os.path.join(TEMP_DIR, f"{job_id}_mask.nii.gz")
                original_nifti = nib.load(file_path)
                
                # Convert prediction back to original affine (simplified here)
                mask_img = nib.Nifti1Image(pred_np, affine=original_nifti.affine)
                nib.save(mask_img, mask_nifti_path)
                
                # Compute Meta (Volume bounds, best slices)
                tumor_counts = np.sum(pred_np, axis=(1, 2))
                if np.sum(tumor_counts) > 0:
                    best_z = int(np.argmax(tumor_counts))
                else:
                    best_z = img_np.shape[0] // 2
                    
                meta = {
                    "shape_xyz": [img_np.shape[2], img_np.shape[1], img_np.shape[0]], # typical X, Y, Z
                    "best_slices": { "z": best_z, "y": img_np.shape[1]//2, "x": img_np.shape[2]//2 }
                }

                # Save standard overlay preview (PNG)
                img_slice = np.rot90(img_np[best_z, :, :])
                mask_slice = np.rot90(pred_np[best_z, :, :])
                
                plt.figure(figsize=(6, 6))
                plt.imshow(img_slice, cmap="gray")
                overlay = np.zeros((*mask_slice.shape, 4))
                overlay[mask_slice == 1] = [1, 0, 0, 0.4] 
                plt.imshow(overlay)
                plt.axis("off")
                overlay_path = os.path.join(TEMP_DIR, f"{job_id}_overlay.png")
                plt.tight_layout()
                plt.savefig(overlay_path, transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close()

                # Also save volume arrays for interactive slicing (heavy so using numpy arrays)
                np.save(os.path.join(TEMP_DIR, f"{job_id}_vol.npy"), img_np)
                np.save(os.path.join(TEMP_DIR, f"{job_id}_pred.npy"), pred_np)

                JOBS[job_id]["status"] = "DONE"
                JOBS[job_id]["result"] = {
                    "tumor_volume_voxels": int(np.sum(pred_np)),
                    "mask_path": mask_nifti_path,
                }
                JOBS[job_id]["meta"] = meta
                print(f"[{job_id}] Finished successfully.")
                break # only 1 item

    except Exception as e:
        import traceback
        traceback.print_exc()
        JOBS[job_id]["status"] = "FAILED"
        JOBS[job_id]["error"] = str(e)


# --- API Routes ---

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/gpu")
def gpu_status():
    if torch.cuda.is_available():
        memory_alloc = torch.cuda.memory_allocated() / (1024**3)
        return {
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(memory_alloc, 2),
            "worker": "FastAPI"
        }
    return {"device_name": "CPU", "memory_allocated_gb": 0, "worker": "FastAPI"}

@app.post("/submit")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    overlap: float = Form(0.5),
    sw_batch_size: int = Form(4)
):
    job_id = str(uuid.uuid4())
    temp_file_path = os.path.join(TEMP_DIR, f"{job_id}_{file.filename}")
    
    # Save the huge .nii file directly to local disk in chunks to avoid OOM
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    JOBS[job_id] = {
        "status": "QUEUED",
        "job_id": job_id,
        "filename": file.filename
    }

    # Start inference async
    background_tasks.add_task(run_inference, job_id, temp_file_path, threshold, overlap, sw_batch_size)
    return {"job_id": job_id, "status": "QUEUED"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]

@app.get("/result/{job_id}")
def get_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    if JOBS[job_id]["status"] != "DONE":
         raise HTTPException(status_code=400, detail="Job not yet complete")
    return JOBS[job_id]["result"]

@app.get("/volume/{job_id}/meta")
def get_meta(job_id: str):
    if job_id not in JOBS or JOBS[job_id]["status"] != "DONE":
        raise HTTPException(status_code=400, detail="Not available")
    return JOBS[job_id]["meta"]

@app.get("/mask/{job_id}")
def get_mask(job_id: str):
    if job_id not in JOBS or JOBS[job_id]["status"] != "DONE":
        raise HTTPException(status_code=400, detail="Not available")
    path = os.path.join(TEMP_DIR, f"{job_id}_mask.nii.gz")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Mask file missing")
    return FileResponse(path, media_type="application/gzip", filename=f"{job_id}_mask.nii.gz")

@app.get("/image/{job_id}")
def get_image(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    filename = JOBS[job_id]["filename"]
    path = os.path.join(TEMP_DIR, f"{job_id}_{filename}")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Original image file missing")
    return FileResponse(path, filename=filename)

@app.get("/overlay/{job_id}")
def get_overlay(job_id: str):
    path = os.path.join(TEMP_DIR, f"{job_id}_overlay.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Overlay not available")
    return FileResponse(path, media_type="image/png")

@app.get("/volume/{job_id}/slice")
def get_slice(job_id: str, plane: str = "axial", index: int = 0, overlay: int = 1, alpha: float = 0.4):
    """Dynamically slices the cached 3D Volume"""
    if job_id not in JOBS or JOBS[job_id]["status"] != "DONE":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    vol_path = os.path.join(TEMP_DIR, f"{job_id}_vol.npy")
    pred_path = os.path.join(TEMP_DIR, f"{job_id}_pred.npy")

    if not os.path.exists(vol_path) or not os.path.exists(pred_path):
        raise HTTPException(status_code=404, detail="Volume arrays missing")

    vol = np.load(vol_path, mmap_mode="r") # Memory-map for speed without huge RAM
    pred = np.load(pred_path, mmap_mode="r")
    
    # vol shape is [Z, Y, X]
    try:
        if plane == "axial":
            if index < 0 or index >= vol.shape[0]:
                index = vol.shape[0] // 2
            img_slice = np.rot90(vol[index, :, :])
            mask_slice = np.rot90(pred[index, :, :])
        elif plane == "coronal":
            if index < 0 or index >= vol.shape[1]:
                index = vol.shape[1] // 2
            img_slice = np.rot90(vol[:, index, :])
            mask_slice = np.rot90(pred[:, index, :])
        elif plane == "sagittal":
            if index < 0 or index >= vol.shape[2]:
                index = vol.shape[2] // 2
            img_slice = np.rot90(vol[:, :, index])
            mask_slice = np.rot90(pred[:, :, index])
        else:
            raise HTTPException(status_code=400, detail="Invalid plane")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Slicing failed: {e}")
    
    # Fast rendering to bytes
    plt.figure(figsize=(5, 5), dpi=100)
    plt.imshow(img_slice, cmap="gray")
    if int(overlay) == 1:
        rgba = np.zeros((*mask_slice.shape, 4))
        rgba[mask_slice == 1] = [1, 0, 0, float(alpha)] 
        plt.imshow(rgba)
    plt.axis("off")
    
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/png")

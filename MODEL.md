# MODEL Reference

This document is the technical reference for the segmentation and explainability pipeline in this repository.

The root [README.md](README.md) is user-facing. This file is model-focused. XAI module-specific maintainer notes live in [backend/xai/README.md](backend/xai/README.md).

## Project intent

The system performs 3D tumor segmentation on NIfTI MRI volumes and exposes outputs through a web workflow. It is split into a web UI, an API gateway, and a GPU inference worker.

## Inference model (high level)

The segmentation worker uses a 3D U-Net style architecture with attention blocks in the decoder path. The worker:
- loads the trained model weights,
- applies medical image preprocessing,
- runs sliding-window inference,
- thresholds logits into a binary tumor mask,
- stores outputs for downstream visualization and download.

## Preprocessing and output flow

Typical processing includes:
- loading NIfTI input,
- channel formatting and orientation normalization,
- intensity scaling and normalization,
- inference on GPU when available,
- postprocessing into mask volume and preview overlays.

Saved artifacts include:
- segmentation mask volume,
- overlay preview image,
- cached volume arrays for interactive slicing,
- metadata for shape and suggested slice indices.

## XAI flow (high level)

After segmentation is complete, an explainability pass can be requested per job.

Implemented methods:
- 3D Grad-CAM for activation-based saliency
- 3D patch occlusion for perturbation-based importance

XAI outputs are exposed as:
- rendered overlay slices,
- downloadable NIfTI heatmaps,
- volume data for interactive visualization,
- summary metrics.

## Tooling overview

- Frontend: React, Vite, Axios, Niivue, Framer Motion
- API gateway: Node.js, Express, Multer, Axios
- Worker service: FastAPI, PyTorch, MONAI, NumPy, NiBabel, Matplotlib
- Explainability: repository-local 3D Grad-CAM and occlusion modules

## Scope note

This file is the canonical technical reference for model behavior and system-level inference/XAI flow in this repository.

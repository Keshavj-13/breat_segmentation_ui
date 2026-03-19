# Medical Segmentation Platform (with XAI)

This project is a full-stack medical imaging application for 3D breast tumor MRI segmentation, with integrated explainability workflows for clinical-style review.

It supports a complete user flow:
- upload NIfTI MRI volumes,
- run segmentation inference on the backend worker,
- inspect slice overlays and generated masks,
- run XAI analysis (Grad-CAM and patch occlusion),
- export segmentation and explanation artifacts.

## XAI capabilities

The XAI layer extends normal segmentation review with:
- 3D Grad-CAM saliency volumes,
- 3D patch-based occlusion importance maps,
- overlay rendering for axial/coronal/sagittal exploration,
- downloadable heatmaps for downstream analysis,
- job-level status and metrics endpoints for explainability runs.

## What this is for

This repository is intended for:
- rapid experimentation with MRI segmentation + explanation,
- UI-first review of model outputs,
- backend orchestration of large-volume inference jobs,
- structured export of masks and XAI outputs.

## Tools used (rough)

- Frontend: React, Vite, Axios, Niivue, Framer Motion
- API gateway: Node.js, Express, Multer, Axios
- Inference worker: FastAPI, PyTorch, MONAI, NumPy, NiBabel, Matplotlib
- Explainability: custom 3D Grad-CAM and patch occlusion modules

## Model and implementation notes

To keep this README user-facing, model-centric and implementation-centric notes are documented in [MODEL.md](MODEL.md).

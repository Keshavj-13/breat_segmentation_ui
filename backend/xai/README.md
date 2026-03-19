# XAI Notes (Model/Maintainer Focus)

This file is intentionally minimal and only records details that are easy to forget or not obvious from quick code browsing.

## Integration contract

- XAI is mounted onto the worker app via `init_xai_router(model, device, temp_dir, jobs_dict)`.
- XAI routes assume segmentation job state is available through the shared `JOBS` dictionary and require job status `DONE` before analysis.
- Cached segmentation arrays (`{job_id}_vol.npy`, `{job_id}_pred.npy`) are the canonical inputs for post-hoc explainability.

## Runtime behavior

- Triggering `/xai/analyze/{job_id}` sets in-memory cache status to `PROCESSING` immediately.
- Analysis runs in background tasks and writes intermediate and final artifacts to the worker temp directory.
- Failures are surfaced by writing `xai_status=FAILED` plus `error` in cache.

## Artifact conventions

- Grad-CAM volume: `{job_id}_gradcam.npy` and `{job_id}_gradcam.nii.gz`
- Occlusion volume: `{job_id}_occlusion.npy` and `{job_id}_occlusion.nii.gz`
- Binary volume endpoint responds as float32 bytes with `X-Shape` and `X-Dtype` headers.

## Endpoint semantics that matter

- `/xai/slice/{job_id}` overlays segmentation, Grad-CAM, and occlusion on a single rendered PNG.
- `/xai/volume/{job_id}/{kind}` serves raw binary arrays for frontend-side 3D rendering.
- `/xai/probe/{job_id}` computes a localized occlusion delta at a requested voxel coordinate.

## Performance assumptions

- NumPy memory mapping is used to avoid loading full arrays into RAM for every slice request.
- Occlusion cost scales with patch grid cardinality; `fast_mode` effectively coarsens the grid by increasing stride.
- GPU availability changes latency substantially for both segmentation and XAI passes.

## Practical guardrails

- Keep artifact naming stable; frontend and proxy routes rely on current filename patterns.
- If worker temp paths change, update XAI file discovery logic and download endpoints together.
- Preserve route parameter meanings (`patch_size`, `stride`, alpha/cmap controls) to avoid UI/API drift.

# Backend

This backend powers a medical image segmentation workflow with two layers:

- an HTTP gateway that accepts uploads, routes jobs, and exposes unified API endpoints,
- a GPU worker service that runs 3D segmentation inference and serves generated artifacts.

The backend accepts NIfTI scans, schedules inference jobs, tracks job state, and returns outputs such as segmentation masks, overlays, and volume metadata for slice viewing.

It also includes an explainability extension that can run Grad-CAM and occlusion analysis on completed jobs and expose those results through dedicated endpoints.

## Tools used (rough)

- Express for the gateway API
- CORS and Multer for upload handling and cross-origin support
- Axios and FormData for forwarding requests between services
- FastAPI for the Python inference service
- PyTorch and MONAI for 3D model inference
- NumPy for tensor and volume processing
- NiBabel for NIfTI I/O
- Matplotlib for generated overlay images
- Uvicorn for serving the Python API

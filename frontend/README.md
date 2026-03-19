# Frontend

This frontend is a medical imaging interface for running and reviewing 3D segmentation jobs on NIfTI scans.

It supports:
- uploading one or more MRI volumes,
- submitting jobs to the backend inference service,
- live job polling and status tracking,
- downloading generated masks and overlays,
- interactive slice exploration (axial/coronal/sagittal),
- an XAI mode that adds Grad-CAM and occlusion visual analysis.

The UI is built for fast feedback during model inference, with a dashboard-style layout, progress reporting for large uploads, and per-job result panels.

## Tools used (rough)

- React for the component-based UI
- Vite for local development and build tooling
- Axios for API communication
- Niivue for medical volume visualization support
- Framer Motion for interface animation
- Lucide React for iconography
- React Hot Toast for notifications
- ESLint for code quality checks

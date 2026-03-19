"""
XAI Mount Script — attaches XAI routes to the existing worker FastAPI app.

Run using:  uvicorn xai_mount:app --host 0.0.0.0 --port 8000
This replaces the normal  uvicorn worker:app  command.

It imports everything from worker.py unchanged, then layers the XAI
router on top — zero modifications to the original worker module.
"""

# Import the original app with all its routes, model, and state
from worker import app, global_model, device, TEMP_DIR, JOBS

# Import the XAI router and its initializer
from xai.xai_routes import router as xai_router, init_xai_router

# Initialize the XAI router with references to the worker's globals
init_xai_router(
    model=global_model,
    device=device,
    temp_dir=TEMP_DIR,
    jobs_dict=JOBS,
)

# Mount the XAI router onto the existing app
app.include_router(xai_router)

print("✓ XAI routes mounted successfully")

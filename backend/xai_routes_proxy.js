/**
 * XAI Proxy Routes for Express Gateway
 * 
 * Proxies all /api/jobs/:id/xai/* requests to the Python FastAPI worker's
 * /xai/* endpoints. Keeps server.js completely unmodified.
 * 
 * Usage: import and call attachXaiRoutes(app) from server entry point,
 * or mount via xai_server.js wrapper.
 */

import axios from "axios";
import { getWorkers } from "./runtime_config.js";

/**
 * Attach XAI proxy routes to an Express app.
 * @param {import('express').Express} app - Express application instance
 * @param {Function} workerForFn - Function that resolves job_id → worker URL
 * @param {Function|Array<string>} workersOrGetter - Workers list or getter for dynamic updates
 */
export function attachXaiRoutes(app, workerForFn, workersOrGetter = getWorkers) {
  const resolveWorkers = () => {
    if (typeof workersOrGetter === "function") return workersOrGetter();
    if (Array.isArray(workersOrGetter)) return workersOrGetter;
    return getWorkers();
  };

  // Trigger XAI analysis
  app.post("/api/jobs/:id/xai/analyze", async (req, res) => {
    try {
      const w = workerForFn(req.params.id);
      const qs = new URLSearchParams(req.query).toString();
      const url = `${w}/xai/analyze/${req.params.id}${qs ? "?" + qs : ""}`;
      const r = await axios.post(url, {}, { timeout: 30000 });
      res.json(r.data);
    } catch (e) {
      const status = e?.response?.status || 500;
      const detail = e?.response?.data?.detail || e?.message || "XAI analysis failed";
      res.status(status).json({ error: detail });
    }
  });

  // XAI Status
  app.get("/api/jobs/:id/xai/status", async (req, res) => {
    try {
      const w = workerForFn(req.params.id);
      const r = await axios.get(`${w}/xai/status/${req.params.id}`, { timeout: 10000 });
      res.json(r.data);
    } catch (e) {
      const status = e?.response?.status || 500;
      res.status(status).json({ error: e?.response?.data?.detail || e?.message });
    }
  });

  // XAI Metrics
  app.get("/api/jobs/:id/xai/metrics", async (req, res) => {
    try {
      const w = workerForFn(req.params.id);
      const r = await axios.get(`${w}/xai/metrics/${req.params.id}`, { timeout: 10000 });
      res.json(r.data);
    } catch (e) {
      const status = e?.response?.status || 500;
      res.status(status).json({ error: e?.response?.data?.detail || e?.message });
    }
  });

  // XAI Slice (image proxy)
  app.get("/api/jobs/:id/xai/slice", async (req, res) => {
    try {
      const w = workerForFn(req.params.id);
      const qs = new URLSearchParams(req.query).toString();
      const url = `${w}/xai/slice/${req.params.id}${qs ? "?" + qs : ""}`;
      const r = await axios.get(url, { responseType: "arraybuffer", timeout: 20000 });
      res.setHeader("Content-Type", r.headers["content-type"] || "image/png");
      res.setHeader("Cache-Control", "no-store");
      res.send(Buffer.from(r.data));
    } catch (e) {
      const status = e?.response?.status || 500;
      res.status(status).json({ error: e?.response?.data?.detail || e?.message });
    }
  });

  // XAI Volume data (binary download for 3D rendering)
  app.get("/api/jobs/:id/xai/volume/:kind", async (req, res) => {
    try {
      const w = workerForFn(req.params.id);
      const r = await axios.get(`${w}/xai/volume/${req.params.id}/${req.params.kind}`, {
        responseType: "arraybuffer", timeout: 60000,
      });
      res.setHeader("Content-Type", "application/octet-stream");
      res.setHeader("X-Shape", r.headers["x-shape"] || "");
      res.setHeader("X-Dtype", r.headers["x-dtype"] || "float32");
      res.setHeader("Cache-Control", "no-store");
      res.send(Buffer.from(r.data));
    } catch (e) {
      const status = e?.response?.status || 500;
      res.status(status).json({ error: e?.response?.data?.detail || e?.message });
    }
  });

  // XAI NIfTI download
  app.get("/api/jobs/:id/xai/download/:kind", async (req, res) => {
    try {
      const w = workerForFn(req.params.id);
      const r = await axios.get(`${w}/xai/download/${req.params.id}/${req.params.kind}`, {
        responseType: "arraybuffer", timeout: 30000,
      });
      res.setHeader("Content-Type", r.headers["content-type"] || "application/gzip");
      res.setHeader("Content-Disposition",
        `attachment; filename="${req.params.id}_${req.params.kind}.nii.gz"`);
      res.setHeader("Cache-Control", "no-store");
      res.send(Buffer.from(r.data));
    } catch (e) {
      const status = e?.response?.status || 500;
      res.status(status).json({ error: e?.response?.data?.detail || e?.message });
    }
  });

  // Patch probe (interactive occlusion)
  app.get("/api/jobs/:id/xai/probe", async (req, res) => {
    try {
      const w = workerForFn(req.params.id);
      const qs = new URLSearchParams(req.query).toString();
      const url = `${w}/xai/probe/${req.params.id}${qs ? "?" + qs : ""}`;
      const r = await axios.get(url, { timeout: 30000 });
      res.json(r.data);
    } catch (e) {
      const status = e?.response?.status || 500;
      res.status(status).json({ error: e?.response?.data?.detail || e?.message });
    }
  });

  // Available colormaps
  app.get("/api/xai/colormaps", async (req, res) => {
    try {
      const workers = resolveWorkers();
      if (!workers.length) throw new Error("No workers configured");
      const w = workers[0];
      const r = await axios.get(`${w}/xai/colormaps`, { timeout: 5000 });
      res.json(r.data);
    } catch (e) {
      res.json({ colormaps: ["jet", "hot", "plasma", "inferno", "viridis", "medical"] });
    }
  });

  console.log("✓ XAI proxy routes attached");
}

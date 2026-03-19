/**
 * XAI-Enhanced Server Entry Point
 * 
 * Wraps the original Express server with XAI proxy routes.
 * Run this instead of server.js to get XAI support:
 *   node xai_server.js
 * 
 * The original server.js remains completely untouched.
 */

import express from "express";
import cors from "cors";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { attachXaiRoutes } from "./xai_routes_proxy.js";
import { getWorkers, getGatewayPort, getRuntimeSummary, parseWorkers } from "./runtime_config.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const UPLOAD_DIR = path.join(__dirname, "uploads");
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const app = express();
app.use(cors());
app.use(express.json());

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + "-" + file.originalname);
  },
});
const upload = multer({ storage });

let WORKERS = getWorkers();

let rr = 0;
const JOB_ROUTE = new Map();

function pickWorkers(k) {
  const n = WORKERS.length;
  const chosen = [];
  for (let i = 0; i < Math.min(k, n); i++) chosen.push(WORKERS[(rr + i) % n]);
  rr = (rr + 1) % n;
  return chosen;
}

function workerFor(job_id) {
  const w = JOB_ROUTE.get(job_id);
  if (!w) throw new Error("Unknown job_id (not routed via backend)");
  return w;
}

// ─── Original routes (exact copy from server.js) ───

app.get("/api/health", (req, res) => res.json({ ok: true }));

app.get("/api/compute/services", (req, res) => {
  res.json({
    workers: WORKERS,
    count: WORKERS.length,
  });
});

app.put("/api/compute/services", (req, res) => {
  const workers = parseWorkers(req.body?.workers);
  if (!workers.length) {
    return res.status(400).json({
      error: "No valid compute service endpoints provided.",
      hint: "Use URLs like http://127.0.0.1:8001 or shorthand like 8001",
    });
  }

  WORKERS = workers;
  rr = 0;
  return res.json({
    ok: true,
    workers: WORKERS,
    count: WORKERS.length,
  });
});

app.get("/api/gpu", async (req, res) => {
  for (const w of WORKERS) {
    try {
      let r;
      try { r = await axios.get(`${w}/gpu`, { timeout: 2500 }); }
      catch { r = await axios.get(`${w}/health`, { timeout: 2500 }); }
      return res.json({ ...(r.data || {}), count: WORKERS.length });
    } catch {}
  }
  return res.status(503).json({ error: "No GPU workers reachable", count: WORKERS.length });
});

app.post("/api/jobs/submit", upload.array("files"), async (req, res) => {
  const files = req.files || [];
  if (!files.length) return res.status(400).json({ error: "No files uploaded" });

  const threshold = req.body.threshold ?? "0.5";
  const overlap = req.body.overlap ?? "0.5";
  const sw_batch_size = req.body.sw_batch_size ?? "4";
  const gpus = Math.max(1, Math.min(Number(req.body.gpus || 1), WORKERS.length));
  const chosen = pickWorkers(gpus);

  const jobs = [];
  for (let i = 0; i < files.length; i++) {
    const worker = chosen[i % chosen.length];
    const f = files[i];

    const form = new FormData();
    form.append("file", fs.createReadStream(f.path), f.originalname);
    form.append("threshold", String(threshold));
    form.append("overlap", String(overlap));
    form.append("sw_batch_size", String(sw_batch_size));

    try {
      const r = await axios.post(`${worker}/submit`, form, {
        headers: form.getHeaders(),
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        timeout: 60 * 60 * 1000,
      });
      fs.unlink(f.path, (err) => { if (err) console.error("Failed to delete temp file:", err); });
      const job_id = r.data.job_id;
      JOB_ROUTE.set(job_id, worker);
      jobs.push({ job_id, filename: f.originalname, routed_to: worker, status: "QUEUED" });
    } catch (err) {
      console.error(`Failed to dispatch job to worker ${worker}:`, err.message);
      fs.unlink(f.path, (e) => { if (e) console.error("Failed to delete temp file:", e); });
      return res.status(503).json({ error: `Worker ${worker} refused connection. Ensure python backend is running.` });
    }
  }
  res.json({ jobs });
});

app.get("/api/jobs/:id/status", async (req, res) => {
  try {
    const w = workerFor(req.params.id);
    const r = await axios.get(`${w}/status/${req.params.id}`, { timeout: 5000 });
    res.json({ ...(r.data || {}), routed_to: w });
  } catch (e) { res.status(500).json({ error: String(e?.message || e) }); }
});

app.get("/api/jobs/:id/result", async (req, res) => {
  try {
    const w = workerFor(req.params.id);
    const r = await axios.get(`${w}/result/${req.params.id}`, { timeout: 15000 });
    const data = r.data || {};
    res.json({
      ...data, routed_to: w,
      image_url: `/api/jobs/${req.params.id}/image`,
      overlay_url: `/api/jobs/${req.params.id}/overlay`,
      mask_url: `/api/jobs/${req.params.id}/mask`,
      meta_url: `/api/jobs/${req.params.id}/meta`,
    });
  } catch (e) { res.status(500).json({ error: String(e?.message || e) }); }
});

app.get("/api/jobs/:id/overlay", async (req, res) => {
  try {
    const w = workerFor(req.params.id);
    const r = await axios.get(`${w}/overlay/${req.params.id}`, { responseType: "arraybuffer", timeout: 20000 });
    res.setHeader("Content-Type", r.headers["content-type"] || "image/png");
    res.setHeader("Cache-Control", "no-store");
    res.send(Buffer.from(r.data));
  } catch { res.status(404).json({ error: "overlay not available yet" }); }
});

app.get("/api/jobs/:id/image", async (req, res) => {
  try {
    const w = workerFor(req.params.id);
    const r = await axios.get(`${w}/image/${req.params.id}`, { responseType: "stream", timeout: 20000 });
    res.setHeader("Content-Type", r.headers["content-type"] || "application/gzip");
    if (r.headers["content-disposition"]) {
      res.setHeader("Content-Disposition", r.headers["content-disposition"]);
    } else {
      res.setHeader("Content-Disposition", `attachment; filename="${req.params.id}_image.nii.gz"`);
    }
    res.setHeader("Cache-Control", "no-store");
    r.data.pipe(res);
  } catch { res.status(404).json({ error: "image not available" }); }
});

app.get("/api/jobs/:id/mask", async (req, res) => {
  try {
    const w = workerFor(req.params.id);
    const r = await axios.get(`${w}/mask/${req.params.id}`, { responseType: "arraybuffer", timeout: 20000 });
    res.setHeader("Content-Type", r.headers["content-type"] || "application/gzip");
    res.setHeader("Content-Disposition", `attachment; filename="${req.params.id}_mask.nii.gz"`);
    res.setHeader("Cache-Control", "no-store");
    res.send(Buffer.from(r.data));
  } catch { res.status(404).json({ error: "mask not available yet" }); }
});

app.get("/api/jobs/:id/meta", async (req, res) => {
  try {
    const w = workerFor(req.params.id);
    const r = await axios.get(`${w}/volume/${req.params.id}/meta`, { timeout: 15000 });
    res.setHeader("Cache-Control", "no-store");
    res.json(r.data);
  } catch (e) { res.status(500).json({ error: String(e?.message || e) }); }
});

app.get("/api/jobs/:id/slice", async (req, res) => {
  try {
    const w = workerFor(req.params.id);
    const qs = new URLSearchParams(req.query).toString();
    const url = `${w}/volume/${req.params.id}/slice${qs ? "?" + qs : ""}`;
    const r = await axios.get(url, { responseType: "arraybuffer", timeout: 20000 });
    res.setHeader("Content-Type", r.headers["content-type"] || "image/png");
    res.setHeader("Cache-Control", "no-store");
    res.send(Buffer.from(r.data));
  } catch (e) { res.status(500).json({ error: String(e?.message || e) }); }
});

// ─── XAI Routes ───
attachXaiRoutes(app, workerFor, () => WORKERS);

const PORT = getGatewayPort("5000");
app.listen(PORT, () => {
  const summary = getRuntimeSummary();
  console.log(`XAI-Enhanced Backend running on http://127.0.0.1:${PORT}`);
  console.log(`Workers (${summary.workerCount}): ${summary.workers.join(", ")}`);
});

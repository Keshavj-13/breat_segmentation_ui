import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import "./xai.css";
import SliceViewer from "./components/SliceViewer";
import XaiDashboard from "./components/XaiDashboard";

const benchmarkMetrics = [
  { label: "Dice", value: 85.43 },
  { label: "IoU", value: 75.72 },
  { label: "Precision", value: 82.94 },
  { label: "Recall", value: 89.39 },
];

function formatStatus(status) {
  if (status === "DONE") return "Completed";
  if (status === "FAILED") return "Needs Review";
  if (status === "PROCESSING") return "Processing";
  if (status === "QUEUED") return "Queued";
  return status || "Unknown";
}

function formatDuration(startMs, endMs) {
  if (!startMs) return "-";
  const seconds = Math.max(0, Math.round(((endMs || Date.now()) - startMs) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs}s`;
}

function readPath(source, path) {
  return path.split(".").reduce((acc, key) => (acc ? acc[key] : undefined), source);
}

function resolveMetricNumber(result, candidatePaths) {
  for (const path of candidatePaths) {
    const value = readPath(result, path);
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function resolveMetricPercent(result, candidatePaths) {
  const value = resolveMetricNumber(result, candidatePaths);
  if (value === null) return "-";
  if (value <= 1) return `${(value * 100).toFixed(2)}%`;
  return `${value.toFixed(2)}%`;
}

export default function XaiApp() {
  const [gpu, setGpu] = useState(null);
  const [gpuErr, setGpuErr] = useState("");
  const [live, setLive] = useState(true);

  const [files, setFiles] = useState([]);
  const fileRef = useRef(null);

  const [threshold, setThreshold] = useState(0.5);
  const [overlap, setOverlap] = useState(0.5);
  const [swBatch, setSwBatch] = useState(4);

  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadEta, setUploadEta] = useState(null);
  const [computeServicesInput, setComputeServicesInput] = useState("");
  const [computeServicesMsg, setComputeServicesMsg] = useState("");
  const [updatingComputeServices, setUpdatingComputeServices] = useState(false);

  async function refreshGpu() {
    try {
      setGpuErr("");
      setGpu(await api.gpu());
    } catch (e) {
      setGpu(null);
      setGpuErr(String(e?.message || e));
    }
  }

  async function refreshComputeServices(showError = false) {
    try {
      const r = await api.computeServices();
      const workers = Array.isArray(r?.workers) ? r.workers : [];
      setComputeServicesInput(workers.join(", "));
      if (!showError) setComputeServicesMsg("");
    } catch (e) {
      if (showError) setComputeServicesMsg(String(e?.message || e));
    }
  }

  async function applyComputeServices() {
    const workers = computeServicesInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);

    if (!workers.length) {
      setComputeServicesMsg("Enter at least one compute endpoint (URL or port).");
      return;
    }

    setUpdatingComputeServices(true);
    try {
      const r = await api.setComputeServices(workers);
      const activeWorkers = Array.isArray(r?.workers) ? r.workers : workers;
      setComputeServicesInput(activeWorkers.join(", "));
      setComputeServicesMsg(`Applied ${activeWorkers.length} compute service(s).`);
      await refreshGpu();
    } catch (e) {
      setComputeServicesMsg(String(e?.message || e));
    } finally {
      setUpdatingComputeServices(false);
    }
  }

  useEffect(() => {
    refreshGpu();
    refreshComputeServices();
  }, []);

  useEffect(() => {
    if (!live) return;
    const t = setInterval(refreshGpu, 5000);
    return () => clearInterval(t);
  }, [live]);

  useEffect(() => {
    const t = setInterval(async () => {
      const active = jobs.filter((j) => j.status !== "DONE" && j.status !== "FAILED");
      if (!active.length) return;

      const updates = await Promise.allSettled(
        active.map(async (j) => {
          const s = await api.status(j.job_id);
          let result = j.result;
          let meta = j.meta;

          if (s.status === "DONE") {
            if (!result) result = await api.result(j.job_id);
            if (!meta) meta = await api.meta(j.job_id);
          }

          return {
            job_id: j.job_id,
            status: s.status,
            error: s.error || null,
            routed_to: s.routed_to,
            result,
            meta,
          };
        }),
      );

      setJobs((prev) =>
        prev.map((j) => {
          const u = updates
            .filter((x) => x.status === "fulfilled")
            .map((x) => x.value)
            .find((x) => x.job_id === j.job_id);

          if (!u) return j;

          const finished = u.status === "DONE" || u.status === "FAILED";
          return {
            ...j,
            ...u,
            completedAt: finished ? j.completedAt || Date.now() : j.completedAt,
          };
        }),
      );
    }, 1500);

    return () => clearInterval(t);
  }, [jobs]);

  useEffect(() => {
    if (!jobs.length) {
      setSelectedJobId(null);
      return;
    }

    if (!selectedJobId || !jobs.some((j) => j.job_id === selectedJobId)) {
      setSelectedJobId(jobs[0].job_id);
    }
  }, [jobs, selectedJobId]);

  async function submit() {
    if (!files.length) return;

    setSubmitting(true);
    setUploadProgress(0);
    setUploadEta(null);

    try {
      const fd = new FormData();
      for (const f of files) fd.append("files", f, f.name);
      fd.append("gpus", "1");
      fd.append("threshold", String(threshold));
      fd.append("overlap", String(overlap));
      fd.append("sw_batch_size", String(swBatch));

      const startTime = Date.now();
      const r = await api.submit(fd, (progressEvent) => {
        if (!progressEvent.lengthComputable) return;

        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);

        const elapsed = (Date.now() - startTime) / 1000;
        if (elapsed > 1 && percentCompleted > 0 && percentCompleted < 100) {
          const speed = progressEvent.loaded / elapsed;
          const remaining = progressEvent.total - progressEvent.loaded;
          const etaSecs = remaining / speed;
          if (etaSecs && Number.isFinite(etaSecs)) {
            setUploadEta(Math.round(etaSecs));
          }
        }
      });

      const now = Date.now();
      const newJobs = (r.jobs || []).map((j) => ({
        ...j,
        error: null,
        result: null,
        meta: null,
        submittedAt: now,
        completedAt: null,
      }));

      setJobs((prev) => [...newJobs, ...prev]);
      setFiles([]);
      if (fileRef.current) fileRef.current.value = "";
    } finally {
      setSubmitting(false);
      setUploadProgress(0);
      setUploadEta(null);
    }
  }

  const activeJobs = jobs.filter((j) => j.status !== "DONE" && j.status !== "FAILED").length;
  const completedJobs = jobs.filter((j) => j.status === "DONE").length;
  const systemStatus = gpuErr ? "Limited" : "Operational";
  const selectedJob = jobs.find((j) => j.job_id === selectedJobId) || null;

  return (
    <div className="researchPage">
      <header className="systemHeader">
        <div className="systemIdentity">
          <h1>Clinical Segmentation System</h1>
          <p>Enhanced Attention U-Net for Breast MRI</p>
        </div>
        <div className="systemReadouts">
          <div className="readoutRow"><span>Compute Status</span><strong>{gpuErr ? "Unavailable" : "Available"}</strong></div>
          <div className="readoutRow"><span>Model Version</span><strong>Enhanced Attention U-Net v1</strong></div>
          <div className="readoutRow"><span>Dataset Name</span><strong>MAMA-MIA</strong></div>
        </div>
      </header>

      <section className="chapter chapterConsole auraField auraConsole">
        <div className="chapterHead">
          <h2>Live Segmentation Console</h2>
          <p>Operational controls for volumetric scan ingestion and inference execution.</p>
        </div>

        <div className="consoleGrid">
          <div className="consolePane">
            <label className="uploadField" onClick={() => fileRef.current?.click()}>
              <span className="uploadLabel">Scan Input</span>
              <span className="uploadHint">NIfTI volumes: .nii, .nii.gz</span>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".nii,.nii.gz"
                onChange={(e) => setFiles(Array.from(e.target.files || []))}
              />
            </label>

            {files.length > 0 ? (
              <div className="fileLine">
                <strong>Selected:</strong> {files.map((f) => f.name).join(", ")}
              </div>
            ) : null}

            <div className="paramGrid">
              <label>
                Probability Threshold
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
              </label>
              <label>
                Sliding Window Overlap
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="0.95"
                  value={overlap}
                  onChange={(e) => setOverlap(Number(e.target.value))}
                />
              </label>
              <label>
                Batch Size
                <input
                  type="number"
                  min="1"
                  max="32"
                  value={swBatch}
                  onChange={(e) => setSwBatch(Number(e.target.value))}
                />
              </label>
            </div>

            {submitting && uploadProgress > 0 && uploadProgress < 100 ? (
              <div className="uploadProgress">
                <div className="uploadProgressHead">
                  <span>Upload in progress</span>
                  <span>{uploadProgress}% {uploadEta !== null ? `(~${uploadEta}s)` : ""}</span>
                </div>
                <div className="progressTrack">
                  <div className="progressFill" style={{ width: `${uploadProgress}%` }} />
                </div>
              </div>
            ) : null}

            <button className="btn btnPrimary" disabled={!files.length || submitting} onClick={submit}>
              {submitting ? "Processing Upload..." : "Run Segmentation"}
            </button>
          </div>

          <aside className="consolePane statusPane">
            <div className="statusGrid">
              <div className="statusLine"><span>System</span><strong>{systemStatus}</strong></div>
              <div className="statusLine"><span>Active Jobs</span><strong>{activeJobs}</strong></div>
              <div className="statusLine"><span>Completed Jobs</span><strong>{completedJobs}</strong></div>
            </div>

            <div className={`statusFlag ${gpuErr ? "critical" : "normal"}`}>
              <strong>{gpuErr ? "Status: Limited Availability" : "Status: Operational"}</strong>
              <p>
                {gpuErr
                  ? "Some compute services are temporarily unavailable. You can continue and retry jobs as needed."
                  : "Compute and API services are reachable."}
              </p>
            </div>

            <div className="statusControls">
              <button className="btn btnGhost" onClick={refreshGpu}>Refresh Compute Status</button>
              <label className="toggleLine">
                <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
                Auto-refresh {live ? "Enabled" : "Paused"}
              </label>
            </div>

            <div className="computeControl">
              <span className="computeLabel">Compute Services</span>
              <div className="computeEditor">
                <input
                  className="computeInput"
                  type="text"
                  value={computeServicesInput}
                  onChange={(e) => setComputeServicesInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") applyComputeServices();
                  }}
                  placeholder="8001, 8002 or http://host:8001"
                />
                <button
                  className="btn btnGhost btnSmall"
                  type="button"
                  onClick={applyComputeServices}
                  disabled={updatingComputeServices}
                >
                  {updatingComputeServices ? "Applying..." : "Apply"}
                </button>
                <button
                  className="btn btnGhost btnSmall"
                  type="button"
                  onClick={() => refreshComputeServices(true)}
                  disabled={updatingComputeServices}
                >
                  Reload
                </button>
              </div>
              <p className="computeHint">Comma-separated services. Port-only values are allowed.</p>
              {computeServicesMsg ? <p className="computeMsg">{computeServicesMsg}</p> : null}
            </div>

            <pre className="systemDump">{JSON.stringify(gpu, null, 2)}</pre>
          </aside>
        </div>
      </section>

      <section className="chapter chapterResults auraField auraResults">
        <div className="chapterHead">
          <h2>Results and Metrics</h2>
          <p>Case-level segmentation outputs and quantitative quality indicators.</p>
        </div>

        {!jobs.length ? (
          <div className="emptyState">
            <p>No active segmentation jobs.</p>
            <p>Upload a case and run segmentation to populate performance records.</p>
          </div>
        ) : (
          <>
            <div className="resultsTableWrap">
              <table className="resultsTable">
                <thead>
                  <tr>
                    <th>Case ID</th>
                    <th>Status</th>
                    <th>Dice</th>
                    <th>IoU</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Processing Time</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j) => (
                    <tr key={j.job_id} className={selectedJobId === j.job_id ? "active" : ""}>
                      <td>{j.job_id.slice(0, 8)}</td>
                      <td>{formatStatus(j.status)}</td>
                      <td>{resolveMetricPercent(j.result, ["dice", "dice_score", "metrics.dice", "metrics.dice_score", "dsc"])}</td>
                      <td>{resolveMetricPercent(j.result, ["iou", "metrics.iou", "metrics.jaccard", "jaccard", "iou_score"])}</td>
                      <td>{resolveMetricPercent(j.result, ["precision", "metrics.precision", "metrics.ppv", "ppv"])}</td>
                      <td>{resolveMetricPercent(j.result, ["recall", "metrics.recall", "metrics.sensitivity", "sensitivity"])} </td>
                      <td>{formatDuration(j.submittedAt, j.completedAt)}</td>
                      <td>
                        <button className="btn btnGhost btnSmall" onClick={() => setSelectedJobId(j.job_id)}>
                          Open
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {selectedJob ? (
              <div className="activeCaseBlock auraField auraInteraction">
                <div className="activeCaseHead">
                  <h3>Active Case: {selectedJob.filename || selectedJob.job_id}</h3>
                  <span className={`statusPill ${selectedJob.status}`}>{formatStatus(selectedJob.status)}</span>
                </div>

                {selectedJob.status === "FAILED" ? (
                  <div className="statusFlag critical compact">
                    <strong>Run Incomplete</strong>
                    <p>{selectedJob.error || "This case could not be completed in the current run."}</p>
                  </div>
                ) : null}

                {selectedJob.status === "DONE" && selectedJob.result ? (
                  <>
                    <div className="linkRow">
                      <a className="btn btnGhost btnSmall" href={selectedJob.result.mask_url} download>
                        Download Mask (.nii.gz)
                      </a>
                      <a className="btn btnGhost btnSmall" href={selectedJob.result.overlay_url} target="_blank" rel="noreferrer">
                        View Overlay
                      </a>
                    </div>

                    {selectedJob.meta ? (
                      <SliceViewer jobId={selectedJob.job_id} shapeXYZ={selectedJob.meta.shape_xyz} bestSlices={selectedJob.meta.best_slices} />
                    ) : (
                      <p className="loadingLine">Loading volume metadata...</p>
                    )}

                    {selectedJob.meta ? (
                      <XaiDashboard jobId={selectedJob.job_id} shapeXYZ={selectedJob.meta.shape_xyz} bestSlices={selectedJob.meta.best_slices} />
                    ) : null}

                    <details className="rawDetails">
                      <summary>Raw Output Metrics</summary>
                      <pre className="systemDump">
                        {JSON.stringify(
                          {
                            tumor_volume_voxels: selectedJob.result.tumor_volume_voxels,
                            resolution: selectedJob.meta?.shape_xyz,
                            metrics: selectedJob.result.metrics || null,
                          },
                          null,
                          2,
                        )}
                      </pre>
                    </details>
                  </>
                ) : null}

                {selectedJob.status !== "DONE" && selectedJob.status !== "FAILED" ? (
                  <p className="loadingLine">Segmentation pipeline is processing this case...</p>
                ) : null}
              </div>
            ) : null}
          </>
        )}
      </section>

      <section className="chapter chapterNarrative chapterOverview flowSection flowLeft">
        <div className="narrativeFlow">
          <div className="flowPane">
            <div className="chapterHead">
              <h2>Research Overview</h2>
            </div>
            <p className="overviewLead">
              Breast cancer remains one of the most common and lethal malignancies among women, where early detection directly impacts survival.
              The proposed Enhanced Attention U-Net improves automatic tumor delineation by integrating spatial and channel attention,
              suppressing irrelevant background responses, and preserving clinically relevant boundaries.
              The model combines attention-guided representation learning with multi-scale feature fusion to improve sensitivity to small,
              irregular lesions in challenging low-contrast breast imaging.
            </p>
          </div>
          <div className="flowSide" aria-hidden="true" />
        </div>
      </section>

      <section className="chapter chapterNarrative chapterArchitecture flowSection flowRight">
        <div className="narrativeFlow">
          <div className="flowPane">
            <div className="chapterHead">
              <h2>Architecture Explanation</h2>
            </div>
            <div className="architectureGrid">
              <div className="architectureText">
                <p><strong>Encoder:</strong> hierarchical Conv3D downsampling captures local texture and global context.</p>
                <p><strong>SCSE Attention:</strong> spatial and channel recalibration improves focus on tumor-relevant structure.</p>
                <p><strong>Decoder + Skip Links:</strong> feature fusion restores fine boundaries while preserving anatomical continuity.</p>
                <p><strong>Segmentation Head:</strong> 1x1x1 projection with sigmoid yields voxel-wise tumor probability maps.</p>
              </div>
              <div className="diagramPlaceholder" role="img" aria-label="Architecture diagram placeholder">
                <div className="diagramStage">Input Volume</div>
                <div className="diagramArrow" />
                <div className="diagramStage">Encoder + SCSE</div>
                <div className="diagramArrow" />
                <div className="diagramStage">Bottleneck Fusion</div>
                <div className="diagramArrow" />
                <div className="diagramStage">Decoder + Skip Links</div>
                <div className="diagramArrow" />
                <div className="diagramStage">Segmentation Head</div>
              </div>
            </div>
          </div>
          <div className="flowSide" aria-hidden="true" />
        </div>
      </section>

      <section className="chapter chapterNarrative chapterDataset flowSection flowLeft">
        <div className="narrativeFlow">
          <div className="flowPane">
            <div className="chapterHead">
              <h2>Dataset Description</h2>
            </div>
            <div className="datasetGrid">
              <div className="datasetText">
                <p>
                  MAMA-MIA is assembled from multi-institutional DCE-MRI cohorts, harmonized to support reproducible breast tumor segmentation research.
                  Cases were curated for pre-treatment clinical relevance and structured metadata consistency.
                </p>
                <p>
                  The dataset unifies orientation and metadata conventions while preserving native image characteristics,
                  enabling downstream teams to choose preprocessing strategies matched to their own modeling objectives.
                </p>
              </div>
              <dl className="datasetStats">
                <div><dt>Total Cases</dt><dd>1,506 DCE-MRI volumes</dd></div>
                <div><dt>Sources</dt><dd>I-SPY1, I-SPY2, NACT-Pilot, DUKE</dd></div>
                <div><dt>Scope</dt><dd>Pre-treatment breast cancer MRI</dd></div>
                <div><dt>Structure</dt><dd>Harmonized metadata + standardized orientation</dd></div>
              </dl>
            </div>
          </div>
          <div className="flowSide" aria-hidden="true" />
        </div>
      </section>

      <section className="chapter chapterNarrative chapterPerformance flowSection flowRight">
        <div className="narrativeFlow">
          <div className="flowPane">
            <div className="chapterHead">
              <h2>Performance and Results</h2>
              <p>Representative benchmark values from the proposed model on the reported study setting.</p>
            </div>

            <div className="metricBars">
              {benchmarkMetrics.map((m) => (
                <div key={m.label} className="metricRow">
                  <div className="metricLabel">{m.label}</div>
                  <div className="metricTrack">
                    <div className="metricFill" style={{ width: `${m.value}%` }} />
                  </div>
                  <div className="metricValue">{m.value.toFixed(2)}%</div>
                </div>
              ))}
            </div>
          </div>
          <div className="flowSide" aria-hidden="true" />
        </div>
      </section>

      <section className="chapter chapterNarrative chapterDirection flowSection flowLeft">
        <div className="narrativeFlow">
          <div className="flowPane">
            <div className="chapterHead">
              <h2>Team and Direction</h2>
            </div>
            <div className="directionGrid">
              <article className="directionBlock directionWho">
                <h3>Who We Are</h3>
                <p>
                  A clinical AI research effort focused on dependable tumor segmentation workflows for breast imaging.
                </p>
              </article>
              <article className="directionBlock directionBuilt">
                <h3>What We Built</h3>
                <p>
                  A modular, attention-enhanced 3D segmentation pipeline unifying interpretability, quantitative evaluation,
                  and operational execution in a single interface.
                </p>
              </article>
              <article className="directionBlock directionNext">
                <h3>Where We Are Going</h3>
                <p>
                  Cross-dataset validation, deployment simplification, and tighter radiology workflow integration with auditable model behavior.
                </p>
              </article>
            </div>
          </div>
          <div className="flowSide" aria-hidden="true" />
        </div>
      </section>
    </div>
  );
}

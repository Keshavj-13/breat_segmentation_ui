function normalizeWorkerUrl(value, defaultHost = "127.0.0.1") {
  if (value === null || value === undefined) return "";
  let worker = String(value).trim();
  if (!worker) return "";

  // Allow shorthand:
  // - "8001" -> http://127.0.0.1:8001
  // - "127.0.0.1:8001" -> http://127.0.0.1:8001
  if (/^\d+$/.test(worker)) {
    worker = `http://${defaultHost}:${worker}`;
  } else if (!/^https?:\/\//i.test(worker) && /^[^/\s]+:\d+$/.test(worker)) {
    worker = `http://${worker}`;
  }

  try {
    const parsed = new URL(worker);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return "";
    return `${parsed.protocol}//${parsed.host}`;
  } catch {
    return "";
  }
}

export function parseWorkers(rawValue, defaultHost = "127.0.0.1") {
  const inputList = Array.isArray(rawValue) ? rawValue : String(rawValue || "").split(",");
  const normalized = inputList
    .map((value) => normalizeWorkerUrl(value, defaultHost))
    .filter(Boolean);

  return Array.from(new Set(normalized));
}

export function getWorkers() {
  const defaultHost = process.env.WORKER_HOST || "127.0.0.1";
  const defaultPort = process.env.WORKER_PORT || "8001";
  const defaultWorker = `http://${defaultHost}:${defaultPort}`;

  const raw = process.env.WORKERS || defaultWorker;
  const workers = parseWorkers(raw, defaultHost);
  return workers.length ? workers : [defaultWorker];
}

export function getGatewayPort(defaultPort = "5000") {
  return process.env.PORT || defaultPort;
}

export function getRuntimeSummary() {
  const workers = getWorkers();
  return {
    workers,
    workerCount: workers.length,
  };
}

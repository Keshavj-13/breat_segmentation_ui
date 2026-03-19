async function req(path, options = {}) {
  const res = await fetch(path, { ...options });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const msg = typeof data === "string" ? data : (data?.error || data?.detail || res.statusText);
    throw new Error(`${options.method || "GET"} ${path} failed: ${res.status} ${msg}`);
  }
  return data;
}

export const api = {
  gpu: () => req("/api/gpu"),
  health: () => req("/api/health"),
  submit: (fd, onProgress) => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/jobs/submit", true);
      if (onProgress && xhr.upload) xhr.upload.onprogress = onProgress;
      xhr.onload = () => {
        let data;
        try { data = JSON.parse(xhr.responseText); } catch { data = xhr.responseText; }
        if (xhr.status >= 400) reject(new Error(data.error || xhr.statusText));
        else resolve(data);
      };
      xhr.onerror = () => reject(new Error("Network Error"));
      xhr.send(fd);
    });
  },
  status: (id) => req(`/api/jobs/${id}/status`),
  result: (id) => req(`/api/jobs/${id}/result`),
  meta: (id) => req(`/api/jobs/${id}/meta`),
  imageUrl: (id) => `/api/jobs/${id}/image`,
  maskUrl: (id) => `/api/jobs/${id}/mask`,
  sliceUrl: (id, params) => `/api/jobs/${id}/slice?${new URLSearchParams(params).toString()}`,
};

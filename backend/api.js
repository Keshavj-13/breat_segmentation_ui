import axios from "axios";

const backendHost = process.env.BACKEND_HOST || "127.0.0.1";
const backendPort = process.env.BACKEND_PORT || "5000";
export const API_BASE = process.env.API_BASE || `http://${backendHost}:${backendPort}`;

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 0,
  maxBodyLength: Infinity,
  maxContentLength: Infinity,
});

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function safeArray(x) {
  return Array.isArray(x) ? x : [];
}

// Normalize possible shapes:
// - { submitted: [...] }
// - [...] directly
export function normalizeSubmitted(respData) {
  if (Array.isArray(respData)) return respData;
  if (respData && Array.isArray(respData.submitted)) return respData.submitted;
  return [];
}

"use strict";
// Config + analysis shell logic. Talks only to the local backend (/api/*).

const $ = (id) => document.getElementById(id);

let activeTab = "folder";
let pollTimer = null;
let redirected = false;

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || res.statusText);
  return data;
}

function configPayload() {
  const payload = {
    api_base: $("apiBase").value.trim(),
    model: $("model").value.trim(),
    language: $("language").value,
    supports_system_message: $("supportsSystem").checked,
    ca_cert_path: $("caCertPath").value.trim(),
  };
  const key = $("apiKey").value;
  if (key) payload.api_key = key; // empty = keep the saved key
  return payload;
}

function setMsg(el, text, kind) {
  el.textContent = text || "";
  el.className = "msg" + (kind ? " " + kind : "");
}

async function loadConfig() {
  try {
    const c = await getJSON("/api/config");
    $("apiBase").value = c.api_base || "";
    $("model").value = c.model || "";
    $("language").value = c.language || "en";
    $("supportsSystem").checked = c.supports_system_message !== false;
    $("caCertPath").value = c.ca_cert_path || "";
    if (c.api_key_set) $("apiKey").placeholder = "•••••••••• (saved — leave blank to keep)";
  } catch (e) {
    /* ignore */
  }
}

async function refreshStatus() {
  try {
    const s = await getJSON("/api/status");
    const badge = $("config-badge");
    if (s.configured) {
      badge.textContent = "configured";
      badge.className = "badge ok";
    } else {
      badge.textContent = "not configured";
      badge.className = "badge warn";
    }
    $("btn-analyze").disabled = !s.configured || s.running;
    if (s.has_graph) $("open-dashboard").classList.remove("hidden");
    if (s.running && !pollTimer) startPolling();
  } catch (e) {
    /* ignore */
  }
}

async function onSave() {
  setMsg($("config-msg"), "Saving…");
  try {
    await postJSON("/api/config", configPayload());
    setMsg($("config-msg"), "Saved.", "ok");
    $("apiKey").value = "";
    await loadConfig();
    await refreshStatus();
  } catch (e) {
    setMsg($("config-msg"), "Save failed: " + e.message, "err");
  }
}

async function onTest() {
  setMsg($("config-msg"), "Testing connection (single egress to your apiBase)…");
  $("btn-test").disabled = true;
  try {
    const r = await postJSON("/api/test-connection", configPayload());
    setMsg($("config-msg"), r.message, r.ok ? "ok" : "err");
  } catch (e) {
    setMsg($("config-msg"), "Test failed: " + e.message, "err");
  } finally {
    $("btn-test").disabled = false;
  }
}

function switchTab(tab) {
  activeTab = tab;
  $("tab-folder").classList.toggle("tab-active", tab === "folder");
  $("tab-zip").classList.toggle("tab-active", tab === "zip");
  $("panel-folder").classList.toggle("hidden", tab !== "folder");
  $("panel-zip").classList.toggle("hidden", tab !== "zip");
}

async function onAnalyze() {
  setMsg($("analyze-msg"), "");
  redirected = false;
  $("open-dashboard").classList.add("hidden");
  const name = $("projectName").value.trim();
  try {
    if (activeTab === "folder") {
      const path = $("folderPath").value.trim();
      if (!path) return setMsg($("analyze-msg"), "Please enter a folder path.", "err");
      await postJSON("/api/analyze/folder", { path, name });
    } else {
      const file = $("zipFile").files[0];
      if (!file) return setMsg($("analyze-msg"), "Please choose a .zip file.", "err");
      const fd = new FormData();
      fd.append("file", file);
      const url = name ? "/api/analyze/zip?name=" + encodeURIComponent(name) : "/api/analyze/zip";
      const res = await fetch(url, { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || res.statusText);
    }
    $("btn-analyze").disabled = true;
    $("progress").classList.remove("hidden");
    startPolling();
  } catch (e) {
    setMsg($("analyze-msg"), "Could not start: " + e.message, "err");
  }
}

function renderProgress(p) {
  $("bar-fill").style.width = (p.percent || 0).toFixed(0) + "%";
  $("phase").textContent = p.phase || "";
  $("percent").textContent = (p.percent || 0).toFixed(0) + "%";
  $("current-file").textContent = p.current_file
    ? p.current_file + (p.total ? `   (${p.processed}/${p.total})` : "")
    : "";
  if (p.stats) {
    const s = p.stats;
    $("stats").textContent = `${s.nodes} nodes · ${s.edges} edges · ${s.files} files · ${s.functions} functions · ${s.classes} classes · ${s.layers} layers`;
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  $("progress").classList.remove("hidden");
  pollTimer = setInterval(async () => {
    let p;
    try {
      p = await getJSON("/api/progress");
    } catch (e) {
      return;
    }
    renderProgress(p);
    if (p.status === "done") {
      clearInterval(pollTimer);
      pollTimer = null;
      setMsg($("analyze-msg"), "Analysis complete — opening the dashboard…", "ok");
      $("open-dashboard").classList.remove("hidden");
      $("btn-analyze").disabled = false;
      if (!redirected) {
        redirected = true;
        setTimeout(() => (window.location.href = "/dashboard/"), 1000);
      }
    } else if (p.status === "error") {
      clearInterval(pollTimer);
      pollTimer = null;
      setMsg($("analyze-msg"), "Analysis failed: " + (p.error || "unknown error"), "err");
      $("btn-analyze").disabled = false;
    }
  }, 800);
}

function init() {
  $("btn-save").addEventListener("click", onSave);
  $("btn-test").addEventListener("click", onTest);
  $("btn-analyze").addEventListener("click", onAnalyze);
  $("tab-folder").addEventListener("click", () => switchTab("folder"));
  $("tab-zip").addEventListener("click", () => switchTab("zip"));
  loadConfig().then(refreshStatus);
}

document.addEventListener("DOMContentLoaded", init);

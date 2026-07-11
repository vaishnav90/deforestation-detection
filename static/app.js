const $ = (sel) => document.querySelector(sel);

const state = {
  map: null,
  overlayLayer: null,
  patches: [],
  models: [],
};

function setStatus(message, type = "info") {
  const el = $("#status");
  el.hidden = !message;
  el.className = `status ${type}`;
  el.textContent = message || "";
}

function pct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  let data = {};
  try {
    data = await res.json();
  } catch (_) {

  }
  if (!res.ok) {
    const detail = data.detail;
    const msg = Array.isArray(detail)
      ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
      : detail || data.message || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

function fillModelSelects(models) {
  state.models = models;
  const options = models
    .filter((m) => m.available)
    .map((m) => {
      const tag = m.recommended ? " ★ best" : "";
      const corr =
        m.corr_with_rate != null ? ` · corr ${m.corr_with_rate.toFixed(2)}` : "";
      const family = m.family === "classical_ml" ? " [ML]" : " [DL]";
      return `<option value="${m.id}">${m.name}${family}${tag}${corr}</option>`;
    })
    .join("");
  $("#demo-model").innerHTML = options;
  $("#upload-model").innerHTML = options;
  updateModelBlurb();
  $("#demo-model").addEventListener("change", updateModelBlurb);
}

function updateModelBlurb() {
  const id = $("#demo-model").value;
  const model = state.models.find((m) => m.id === id);
  $("#model-blurb").textContent = model?.blurb || "";
}

function fillPatchSelects(patches) {
  state.patches = patches;

  const ranked = [...patches].sort(
    (a, b) =>
      (b.ground_truth_deforestation_rate || 0) -
      (a.ground_truth_deforestation_rate || 0)
  );
  const defaultId = ranked[0]?.id ?? 0;

  const options = patches
    .map((p) => {
      const gt =
        p.ground_truth_deforestation_rate != null
          ? ` · GT ${(p.ground_truth_deforestation_rate * 100).toFixed(1)}%`
          : "";
      return `<option value="${p.id}">${p.label}${gt}</option>`;
    })
    .join("");

  $("#patch-select").innerHTML = options;
  $("#compare-patch").innerHTML = options;
  $("#patch-select").value = String(defaultId);
  $("#compare-patch").value = String(defaultId);

  updatePatchMeta("#patch-select", "#patch-meta");
  updatePatchMeta("#compare-patch", "#compare-meta");
  $("#patch-select").addEventListener("change", () =>
    updatePatchMeta("#patch-select", "#patch-meta")
  );
  $("#compare-patch").addEventListener("change", () =>
    updatePatchMeta("#compare-patch", "#compare-meta")
  );
}

function updatePatchMeta(selectId, metaId) {
  const id = Number($(selectId).value);
  const patch = state.patches.find((p) => p.id === id);
  if (!patch) {
    $(metaId).textContent = "";
    return;
  }
  const gt =
    patch.ground_truth_deforestation_rate != null
      ? `Ground-truth deforestation: ${(patch.ground_truth_deforestation_rate * 100).toFixed(1)}%`
      : "No ground-truth mask";
  $(metaId).textContent = gt;
}

function renderLeaderboard(data) {
  const baseline = data.baseline;
  const rows = data.models
    .map((m, i) => {
      const fam =
        m.family === "classical_ml"
          ? '<span class="family-pill ml">Classical ML</span>'
          : '<span class="family-pill">Deep Learning</span>';
      const star = m.recommended ? " ★" : "";
      const f1 = m.val_f1 ?? m.f1;
      const acc = m.val_accuracy ?? m.accuracy;
      const beat =
        m.beats_majority === true
          ? '<span class="badge ok">beats baseline</span>'
          : m.beats_majority === false
            ? '<span class="badge idle">≈ majority</span>'
            : "";
      return `<tr class="${m.recommended ? "recommended" : ""}">
        <td>${i + 1}. ${m.name}${star}</td>
        <td>${fam}</td>
        <td><strong>${m.corr_with_rate != null ? m.corr_with_rate.toFixed(2) : "—"}</strong></td>
        <td>${f1 != null ? (f1 * 100).toFixed(1) + "%" : "—"}</td>
        <td>${acc != null ? (acc * 100).toFixed(1) + "%" : "—"}</td>
        <td>${beat}</td>
      </tr>`;
    })
    .join("");

  const baselineRow = baseline
    ? `<tr class="baseline-row">
        <td>${baseline.name}</td>
        <td><span class="family-pill">Baseline</span></td>
        <td>—</td>
        <td>${(baseline.val_f1 * 100).toFixed(1)}%</td>
        <td>${(baseline.val_accuracy * 100).toFixed(1)}%</td>
        <td></td>
      </tr>`
    : "";

  $("#leaderboard-table").innerHTML = `
    <p class="hint" style="margin-bottom:0.75rem">
      Ranked by <strong>correlation with deforestation rate</strong> (most honest signal).
      Val F1/Acc are held-out (n=6). Binary F1 saturates because the dataset is tiny and imbalanced.
    </p>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Family</th>
          <th>Corr. w/ rate</th>
          <th>Val F1</th>
          <th>Val Acc</th>
          <th></th>
        </tr>
      </thead>
      <tbody>${rows}${baselineRow}</tbody>
    </table>`;

  $("#leaderboard-notes").innerHTML = (data.notes || [])
    .map((n) => `<li>${n}</li>`)
    .join("");
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`#${tab.dataset.tab}-form`).classList.add("active");
    });
  });

  document.querySelectorAll(".viewer-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".viewer-tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".view-pane").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`#view-${tab.dataset.view}`).classList.add("active");
      if (tab.dataset.view === "map" && state.map) {
        setTimeout(() => state.map.invalidateSize(), 50);
      }
    });
  });
}

function bindThresholdToggles() {
  const sync = () => {
    $("#demo-threshold-wrap").hidden = $("#demo-use-auto").checked;
  };
  $("#demo-use-auto").addEventListener("change", sync);
  sync();

  const input = $("#demo-threshold");
  const label = $("#demo-threshold-val");
  const update = () => {
    label.textContent = Number(input.value).toFixed(2);
  };
  input.addEventListener("input", update);
  update();
}

function ensureMap() {
  if (state.map) return state.map;
  state.map = L.map("map", { zoomControl: true }).setView([0, -60], 4);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap",
    maxZoom: 19,
  }).addTo(state.map);
  return state.map;
}

function updateMap(result) {
  const map = ensureMap();
  if (state.overlayLayer) {
    map.removeLayer(state.overlayLayer);
    state.overlayLayer = null;
  }
  if (!result.bounds || result.bounds.length !== 4 || !result.heatmap_url) return;

  const [left, bottom, right, top] = result.bounds;
  const isGeographic = Math.abs(left) <= 180 && Math.abs(right) <= 180 && Math.abs(top) <= 90;
  const bounds = isGeographic
    ? [
        [bottom, left],
        [top, right],
      ]
    : [
        [-5.5, -55.5],
        [-4.5, -54.5],
      ];
  state.overlayLayer = L.imageOverlay(result.heatmap_url, bounds, { opacity: 0.85 }).addTo(map);
  map.fitBounds(bounds, { padding: [20, 20] });
}

function showSingleResult(result) {
  $("#compare-results").hidden = true;
  $("#viewer").style.display = "";

  $("#stat-prob").textContent = pct(result.probability);
  $("#stat-rate").textContent = pct(result.estimated_rate ?? result.deforestation_fraction);
  $("#stat-thr").textContent = result.calibrated
    ? `${Number(result.threshold).toFixed(2)} (auto)`
    : Number(result.threshold).toFixed(2);
  $("#stat-gt").textContent =
    result.ground_truth_deforestation_rate != null
      ? pct(result.ground_truth_deforestation_rate)
      : "—";

  const badge = $("#result-badge");
  badge.textContent = result.label;
  badge.className = `badge ${result.is_deforested ? "danger" : "ok"}`;

  const heatmap = $("#heatmap-img");
  const preview = $("#preview-img");
  if (result.heatmap_url) {
    heatmap.src = `${result.heatmap_url}?t=${Date.now()}`;
    heatmap.classList.add("visible");
    $("#overlay-empty").style.display = "none";
  }
  if (result.preview_url) {
    preview.src = `${result.preview_url}?t=${Date.now()}`;
    preview.classList.add("visible");
    $("#preview-empty").style.display = "none";
  }

  $("#downloads").hidden = false;
  $("#dl-heatmap").href = result.heatmap_url;
  $("#dl-mask").href = result.mask_url;
  updateMap(result);
}

function showCompareResult(data) {
  $("#downloads").hidden = true;
  const box = $("#compare-results");
  box.hidden = false;

  $("#stat-prob").textContent = `${data.agreement}/${data.total_models}`;
  $("#stat-rate").textContent = "agree";
  $("#stat-thr").textContent = "multi";
  $("#stat-gt").textContent = pct(data.ground_truth_deforestation_rate);

  const badge = $("#result-badge");
  badge.textContent = data.ground_truth_label;
  badge.className = `badge ${data.ground_truth_label === "Deforestation" ? "danger" : "ok"}`;

  box.innerHTML = `
    <p class="hint">Ground truth: <strong>${data.ground_truth_label}</strong>
    (${pct(data.ground_truth_deforestation_rate)}) —
    ${data.agreement}/${data.total_models} models correct</p>
    ${data.comparisons
      .map((c) => {
        const ok = c.correct ? "correct" : "wrong";
        const mark = c.correct ? "✓" : "✗";
        return `<div class="compare-card ${ok}">
          <div>
            <div class="compare-name">${c.name}</div>
            <div class="compare-meta">${c.family === "classical_ml" ? "Classical ML" : "Deep Learning"} · ${c.blurb || ""}</div>
          </div>
          <div><div class="compare-meta">Score</div><strong>${pct(c.probability)}</strong></div>
          <div><div class="compare-meta">Est. rate</div><strong>${pct(c.estimated_rate)}</strong></div>
          <div><div class="compare-meta">Decision</div><strong>${c.is_deforested ? "Loss" : "Intact"}</strong></div>
          <div class="badge ${c.correct ? "ok" : "danger"}">${mark}</div>
        </div>`;
      })
      .join("")}
  `;

  const best = data.comparisons.find((c) => c.heatmap_url) || data.comparisons[0];
  if (best) {
    const heatmap = $("#heatmap-img");
    const preview = $("#preview-img");
    if (best.heatmap_url) {
      heatmap.src = `${best.heatmap_url}?t=${Date.now()}`;
      heatmap.classList.add("visible");
      $("#overlay-empty").style.display = "none";
    }
    const previewUrl = data.preview_url || best.preview_url;
    if (previewUrl) {
      preview.src = `${previewUrl}?t=${Date.now()}`;
      preview.classList.add("visible");
      $("#preview-empty").style.display = "none";
    }
    updateMap(best);
  }
}

async function runCompare(event) {
  event.preventDefault();
  const btn = $("#compare-submit");
  btn.disabled = true;
  setStatus("Comparing 7 models on this patch… ~30–60s on CPU.", "info");
  const body = new FormData();
  body.append("patch_id", $("#compare-patch").value);
  try {
    const result = await fetchJSON("/api/v1/compare", { method: "POST", body });
    showCompareResult(result);
    setStatus(
      `Comparison done. ${result.agreement}/${result.total_models} models matched ground truth.`,
      "info"
    );
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    btn.disabled = false;
  }
}

async function runDemo(event) {
  event.preventDefault();
  const btn = $("#demo-submit");
  btn.disabled = true;
  setStatus("Running model…", "info");

  const body = new FormData();
  body.append("patch_id", $("#patch-select").value);
  body.append("model_id", $("#demo-model").value);
  if (!$("#demo-use-auto").checked) {
    body.append("threshold", $("#demo-threshold").value);
  }

  try {
    const result = await fetchJSON("/api/v1/predict/demo", { method: "POST", body });
    showSingleResult(result);
    setStatus(
      `Done · ${result.model_id} · ${result.label} (${pct(result.probability)})`,
      "info"
    );
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    btn.disabled = false;
  }
}

async function runUpload(event) {
  event.preventDefault();
  const s1 = $("#s1-file").files[0];
  const s2 = $("#s2-file").files[0];
  if (!s1 && !s2) {
    setStatus("Choose at least one GeoTIFF to upload.", "error");
    return;
  }

  const btn = $("#upload-submit");
  btn.disabled = true;
  setStatus("Uploading and running inference…", "info");

  const body = new FormData();
  body.append("model_id", $("#upload-model").value);
  if (s1) body.append("sentinel1", s1);
  if (s2) body.append("sentinel2", s2);

  try {
    const result = await fetchJSON("/api/v1/predict", { method: "POST", body });
    showSingleResult(result);
    setStatus(`Done. ${result.label} (${pct(result.probability)}).`, "info");
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    btn.disabled = false;
  }
}

async function init() {
  bindTabs();
  bindThresholdToggles();
  $("#compare-form").addEventListener("submit", runCompare);
  $("#demo-form").addEventListener("submit", runDemo);
  $("#upload-form").addEventListener("submit", runUpload);

  try {
    const [modelsResp, patchesResp, board] = await Promise.all([
      fetchJSON("/api/v1/models"),
      fetchJSON("/api/v1/demo/patches"),
      fetchJSON("/api/v1/leaderboard"),
    ]);
    fillModelSelects(modelsResp.models);
    fillPatchSelects(patchesResp.patches);
    renderLeaderboard(board);
    setStatus("Ready. Open Compare models for a presentation-ready side-by-side run.", "info");
  } catch (err) {
    setStatus(`Failed to initialize: ${err.message}`, "error");
  }
}

init();

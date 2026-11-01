(() => {
  "use strict";

  const MODEL = window.AIRFOIL_MODEL;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  if (!MODEL) {
    document.body.innerHTML =
      '<p style="padding:2rem;font-family:monospace;color:#f78166">model-data.js is missing. Run `uv run airfoil-run` first.</p>';
    return;
  }

  const target = MODEL.target;
  const rmse = MODEL.prediction.rmse;
  const state = { values: {} };

  const RANGE_STEPS = 1000;
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const fmt = (value, decimals) => Number(value).toFixed(decimals);
  const fmtSigned = (value, decimals) => `${value >= 0 ? "+" : ""}${Number(value).toFixed(decimals)}`;

  function normalize(feature, value) {
    const rounded = Number(Number(value).toFixed(feature.decimals));
    return clamp(rounded, feature.min, feature.max);
  }

  function positionFor(feature, value) {
    return ((value - feature.min) / (feature.max - feature.min)) * RANGE_STEPS;
  }

  function valueFromPosition(feature, position) {
    const span = feature.max - feature.min;
    return normalize(feature, feature.min + (Number(position) / RANGE_STEPS) * span);
  }

  function predict(values) {
    let result = MODEL.prediction.intercept;
    for (const feature of MODEL.features) {
      result += feature.coef * ((values[feature.key] - feature.mean) / feature.scale);
    }
    return result;
  }

  function initialValues() {
    const values = {};
    for (const feature of MODEL.features) {
      values[feature.key] = normalize(feature, feature.mean);
    }
    return values;
  }

  // ---------------------------------------------------------------- sliders

  const slidersHost = $("#sliders");

  function buildSliders() {
    const fragment = document.createDocumentFragment();

    for (const feature of MODEL.features) {
      const row = document.createElement("div");
      row.className = "slider-row";

      const top = document.createElement("div");
      top.className = "slider-top";

      const label = document.createElement("label");
      label.htmlFor = `range-${feature.key}`;
      label.innerHTML = `${feature.label} <span class="unit">${feature.unit}</span>`;

      const number = document.createElement("input");
      number.className = "num-input";
      number.type = "number";
      number.id = `number-${feature.key}`;
      number.min = feature.min;
      number.max = feature.max;
      number.step = feature.step;
      number.inputMode = "decimal";
      number.setAttribute("aria-label", `${feature.label} in ${feature.unit}`);

      top.append(label, number);

      const range = document.createElement("input");
      range.type = "range";
      range.id = `range-${feature.key}`;
      range.min = 0;
      range.max = RANGE_STEPS;
      range.step = 1;
      range.setAttribute("aria-label", `${feature.label} slider`);

      const hint = document.createElement("p");
      hint.className = "range-hint";
      hint.textContent = `Observed ${fmt(feature.min, feature.decimals)} – ${fmt(feature.max, feature.decimals)} ${feature.unit}`;

      row.append(top, range, hint);
      fragment.append(row);

      const sync = (value) => {
        state.values[feature.key] = value;
        range.value = positionFor(feature, value);
        range.setAttribute("aria-valuetext", `${fmt(value, feature.decimals)} ${feature.unit}`);
        number.value = fmt(value, feature.decimals);
        refresh();
      };

      range.addEventListener("input", (event) =>
        sync(valueFromPosition(feature, event.target.value))
      );
      number.addEventListener("change", (event) => {
        if (event.target.value === "") return;
        sync(normalize(feature, event.target.value));
      });
      number.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && event.target.value !== "") {
          sync(normalize(feature, event.target.value));
        }
      });
    }

    slidersHost.append(fragment);
  }

  function setValues(values) {
    state.values = { ...values };
    for (const feature of MODEL.features) {
      const value = state.values[feature.key];
      const range = $(`#range-${feature.key}`);
      range.value = positionFor(feature, value);
      range.setAttribute("aria-valuetext", `${fmt(value, feature.decimals)} ${feature.unit}`);
      $(`#number-${feature.key}`).value = fmt(value, feature.decimals);
    }
    refresh();
  }

  // ------------------------------------------------------------- prediction

  function refresh() {
    const value = predict(state.values);
    $("#prediction-value").textContent = fmt(value, 1);

    const delta = value - target.mean;
    const deltaElement = $("#prediction-delta");
    deltaElement.textContent = `${fmtSigned(delta, 1)} dB vs. dataset mean`;
    deltaElement.style.color = delta >= 0 ? "var(--coral)" : "var(--green)";
    $("#prediction-band").textContent = `± ${fmt(rmse, 1)} dB test RMSE`;

    const position = clamp((value - target.min) / (target.max - target.min), 0, 1);
    $("#meter-marker").style.left = `${(position * 100).toFixed(1)}%`;

    drawSensitivity();
  }

  // ------------------------------------------------------ sensitivity chart

  const canvas = $("#sensitivity");
  const featureSelect = $("#sensitivity-feature");
  const palette = {
    grid: "#21262d",
    muted: "#8b949e",
    text: "#e6edf3",
    accent: "#58a6ff",
    band: "rgba(88, 166, 255, 0.12)",
  };

  function selectedFeature() {
    return MODEL.features.find((feature) => feature.key === featureSelect.value) || MODEL.features[0];
  }

  function drawSensitivity() {
    const feature = selectedFeature();
    if (!feature) return;

    const rect = canvas.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 40) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const width = rect.width;
    const height = rect.height;
    const pad = { top: 18, right: 18, bottom: 42, left: 58 };
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;

    const samples = 72;
    const xs = [];
    const ys = [];
    for (let index = 0; index < samples; index += 1) {
      const x = feature.min + ((feature.max - feature.min) * index) / (samples - 1);
      const values = { ...state.values, [feature.key]: x };
      xs.push(x);
      ys.push(predict(values));
    }

    let yLow = Math.min(...ys) - rmse;
    let yHigh = Math.max(...ys) + rmse;
    const padY = Math.max(0.5, (yHigh - yLow) * 0.08);
    yLow -= padY;
    yHigh += padY;

    const xFor = (x) => pad.left + ((x - feature.min) / (feature.max - feature.min)) * plotWidth;
    const yFor = (y) => pad.top + ((yHigh - y) / (yHigh - yLow)) * plotHeight;

    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.strokeStyle = palette.grid;
    ctx.lineWidth = 1;

    ctx.fillStyle = palette.muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yTicks = 5;
    for (let tick = 0; tick <= yTicks; tick += 1) {
      const y = yLow + ((yHigh - yLow) * tick) / yTicks;
      const py = yFor(y);
      ctx.beginPath();
      ctx.moveTo(pad.left, py);
      ctx.lineTo(pad.left + plotWidth, py);
      ctx.stroke();
      ctx.fillText(y.toFixed(1), pad.left - 10, py);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xTicks = 5;
    for (let tick = 0; tick <= xTicks; tick += 1) {
      const x = feature.min + ((feature.max - feature.min) * tick) / xTicks;
      const px = xFor(x);
      ctx.beginPath();
      ctx.moveTo(px, pad.top);
      ctx.lineTo(px, pad.top + plotHeight);
      ctx.stroke();
      ctx.fillText(fmt(x, feature.decimals), px, pad.top + plotHeight + 10);
    }

    ctx.beginPath();
    ys.forEach((y, index) => {
      const px = xFor(xs[index]);
      const py = yFor(y - rmse);
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    for (let index = ys.length - 1; index >= 0; index -= 1) {
      ctx.lineTo(xFor(xs[index]), yFor(ys[index] + rmse));
    }
    ctx.closePath();
    ctx.fillStyle = palette.band;
    ctx.fill();

    ctx.beginPath();
    ys.forEach((y, index) => {
      const px = xFor(xs[index]);
      const py = yFor(y);
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = palette.accent;
    ctx.lineWidth = 2;
    ctx.stroke();

    const markerX = xFor(state.values[feature.key]);
    const markerY = yFor(predict(state.values));
    ctx.strokeStyle = "rgba(230, 237, 243, 0.28)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(markerX, pad.top);
    ctx.lineTo(markerX, pad.top + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.arc(markerX, markerY, 5, 0, Math.PI * 2);
    ctx.fillStyle = palette.text;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(markerX, markerY, 9, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(88, 166, 255, 0.55)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = palette.muted;
    ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${feature.label} (${feature.unit})`, pad.left + plotWidth / 2, height - 12);

    ctx.save();
    ctx.translate(14, pad.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Predicted SPL (dB)", 0, 0);
    ctx.restore();
  }

  // ------------------------------------------------------------------- tabs

  function activateTab(id, updateHash = true) {
    const tabs = $$('[role="tab"]');
    const panels = $$('[role="tabpanel"]');

    for (const tab of tabs) {
      const active = tab.id === id;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    for (const panel of panels) {
      const active = panel.getAttribute("aria-labelledby") === id;
      panel.hidden = !active;
      panel.classList.toggle("is-active", active);
    }

    if (id === "tab-predict") requestAnimationFrame(drawSensitivity);

    if (updateHash) {
      const slug = id.replace(/^tab-/, "");
      try {
        if (location.hash !== `#${slug}`) history.replaceState(null, "", `#${slug}`);
      } catch {
        // Some file:// contexts restrict history updates; tab switching still works.
      }
    }
  }

  $$('[role="tab"]').forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.id));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      const tabs = $$('[role="tab"]');
      const index = tabs.indexOf(tab);
      const next =
        event.key === "ArrowRight"
          ? (index + 1) % tabs.length
          : (index - 1 + tabs.length) % tabs.length;
      tabs[next].focus();
      activateTab(tabs[next].id);
    });
  });

  // --------------------------------------------------------- static sections

  function renderModelSection() {
    const { test, cv_r2_mean: cvMean, cv_r2_std: cvStd } = MODEL.metrics;
    $("#stat-r2").textContent = fmt(test.r2, 3);
    $("#stat-rmse").textContent = fmt(test.rmse, 2);
    $("#stat-mae").textContent = fmt(test.mae, 2);
    $("#stat-cv").textContent = `${fmt(cvMean, 3)} ± ${fmt(cvStd, 3)}`;

    const coefficientBody = $("#coef-table tbody");
    coefficientBody.replaceChildren(
      ...MODEL.coefficients.map((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${row.label} <span class="muted">(${row.unit})</span></td><td class="num">${fmtSigned(row.coefficient, 2)}</td>`;
        return tr;
      })
    );

    const comparisonBody = $("#comparison-table tbody");
    comparisonBody.replaceChildren(
      ...MODEL.comparison.map((row) => {
        const tr = document.createElement("tr");
        const alpha = row.alpha === null ? "—" : fmt(row.alpha, 3);
        tr.innerHTML = `<td>${row.model}</td><td class="num">${alpha}</td><td class="num">${fmt(row.r2, 3)}</td><td class="num">${fmt(row.rmse, 2)}</td><td class="num">${fmt(row.mae, 2)}</td>`;
        return tr;
      })
    );

    const regularized = MODEL.comparison
      .filter((row) => row.model !== "OLS")
      .sort((a, b) => b.r2 - a.r2)[0];
    if (regularized) {
      const delta = regularized.r2 - MODEL.metrics.test.r2;
      $("#comparison-note").textContent =
        `${regularized.model} (α = ${fmt(regularized.alpha, 3)}) changes test R² by ` +
        `${fmtSigned(delta, 4)} versus OLS. Angle of attack and displacement thickness are ` +
        "correlated, but five features on 1,200+ training rows leave little variance to shrink.";
    }

    const logGain = MODEL.log_frequency.r2_gain;
    $("#log-note").textContent =
      `Adding log(frequency) moves test R² from ${fmt(MODEL.metrics.test.r2, 3)} to ` +
      `${fmt(MODEL.log_frequency.test.r2, 3)} (${fmtSigned(logGain, 4)}) and CV R² from ` +
      `${fmt(MODEL.metrics.cv_r2_mean, 3)} to ${fmt(MODEL.log_frequency.cv_r2_mean, 3)}. ` +
      "The linear term already captures most of the log-linear trend.";
  }

  function renderDataSection() {
    const cleaning = MODEL.cleaning;
    $("#clean-raw").textContent = cleaning.raw_rows.toLocaleString();
    $("#clean-dupes").textContent = cleaning.duplicates.toLocaleString();
    $("#clean-invalid").textContent = cleaning.invalid_or_missing.toLocaleString();
    $("#clean-final").textContent = cleaning.final_rows.toLocaleString();
    $("#chip-rows").textContent = MODEL.meta.rows.toLocaleString();
    $("#meta-seed").textContent = `seed ${MODEL.meta.seed}`;
    $("#meter-min").textContent = fmt(target.min, 1);
    $("#meter-max").textContent = fmt(target.max, 1);

    const headerRow = document.createElement("tr");
    for (const column of MODEL.preview.columns) {
      const th = document.createElement("th");
      th.innerHTML = `${column.label} <span class="unit">${column.unit}</span>`;
      if (column.key !== "frequency_hz") th.classList.add("num");
      headerRow.append(th);
    }
    $("#preview-table thead").replaceChildren(headerRow);

    const body = $("#preview-table tbody");
    body.replaceChildren(
      ...MODEL.preview.rows.map((row) => {
        const tr = document.createElement("tr");
        MODEL.preview.columns.forEach((column, index) => {
          const td = document.createElement("td");
          td.className = "num";
          td.textContent = fmt(row[index], column.decimals);
          tr.append(td);
        });
        return tr;
      })
    );
  }

  function renderSelect() {
    featureSelect.replaceChildren(
      ...MODEL.features.map((feature) => {
        const option = document.createElement("option");
        option.value = feature.key;
        option.textContent = `${feature.label} (${feature.unit})`;
        return option;
      })
    );
    featureSelect.value = MODEL.features[0].key;
    featureSelect.addEventListener("change", () => {
      drawSensitivity();
      updateCaption();
    });
  }

  function updateCaption() {
    const feature = selectedFeature();
    $("#sensitivity-caption").textContent =
      `Holding the other four inputs at their current values while sweeping ` +
      `${feature.label.toLowerCase()} from ${fmt(feature.min, feature.decimals)} to ` +
      `${fmt(feature.max, feature.decimals)} ${feature.unit}. The shaded band is ` +
      `± ${fmt(rmse, 1)} dB test RMSE.`;
  }

  // ------------------------------------------------------------------- init

  function tabFromHash() {
    const id = `tab-${location.hash.replace(/^#/, "")}`;
    return $$('[role="tab"]').some((tab) => tab.id === id) ? id : "tab-predict";
  }

  state.values = initialValues();
  buildSliders();
  renderSelect();
  renderModelSection();
  renderDataSection();
  setValues(state.values);
  updateCaption();

  $("#reset").addEventListener("click", () => setValues(initialValues()));
  window.addEventListener("resize", drawSensitivity);
  window.addEventListener("hashchange", () => activateTab(tabFromHash(), false));
  activateTab(tabFromHash(), false);
})();

(() => {
  "use strict";

  const STORAGE_KEY = "heatguard-community-submissions-v1";
  const DEVICE_KEY = "heatguard-community-device-v1";
  const MAP = { width: 1080, height: 650, padX: 52, padY: 46 };
  const BOUNDS = { minLon: 103.58, maxLon: 104.04, minLat: 1.215, maxLat: 1.48 };
  const FLOW_COPY = {
    proposal: { step: "Step 01", title: "Propose a green pocket", photo: "Before photo", requirePhoto: true, impact: "Cooling access estimate after review" },
    confirmation: { step: "Step 02", title: "Confirm it is built", photo: "After photo", requirePhoto: true, impact: "Public map eligibility pending staff confirmation" },
    maintenance: { step: "Step 03", title: "Monthly check-in", photo: "Condition photo", requirePhoto: true, impact: "Green coverage confidence refreshed" },
    comfort: { step: "Step 04", title: "On-site comfort", photo: "Photo not required", requirePhoto: false, impact: "Comfort signal added to aggregated validation" },
  };
  const SUMMARY_LABELS = {
    siteType: "Site type",
    approxSize: "Approximate size",
    shade: "Existing shade",
    seating: "Seating",
    plantTypes: "Suggested planting",
    approvalStatus: "Town council stage",
    proposalReference: "Proposal reference",
    verifierRole: "Confirmed by",
    pocketReference: "Pocket reference",
    conditionRating: "Condition rating",
    issue: "Observed issue",
    qrReference: "Pocket / QR code",
    comfort: "Thermal comfort",
  };
  const SUMMARY_VALUES = {
    "void-deck": "Void deck edge",
    "open-space": "Open space",
    "corridor-edge": "Corridor edge",
    "community-facility": "Community facility",
    "under-25": "Under 25 m²",
    "25-100": "25–100 m²",
    "100-500": "100–500 m²",
    "over-500": "Over 500 m²",
    none: "None",
    partial: "Partial",
    full: "Full",
    yes: "Yes",
    no: "No",
    native: "Native",
    edible: "Edible",
    flowering: "Flowering",
    "shade-tree": "Shade tree",
    "not-requested": "Not requested yet",
    requested: "Approval requested",
    approved: "Approved",
    "town-council": "Town council staff",
    aac: "AAC staff",
    "resident-pending": "Resident · pending staff review",
    "dead-plants": "Dead or damaged plants",
    "standing-water": "Standing water",
    "blocked-access": "Blocked access",
    "damaged-seating": "Damaged seating",
    "too-hot": "Too hot",
    ok: "OK",
    cool: "Cool",
  };
  const CHANNEL_LABELS = { web: "Web form", qr: "On-site QR", aac: "AAC-assisted" };

  const state = {
    kind: "proposal",
    area: "",
    lat: null,
    lon: null,
    photoDataUrl: null,
    pendingPayload: null,
  };

  const byId = (id) => document.getElementById(id);
  const svgEl = (name, attrs = {}) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };
  const uuid = () => crypto.randomUUID?.() || `hg-${Date.now()}-${Math.random().toString(16).slice(2)}`;

  function project([lon, lat]) {
    return [
      MAP.padX + ((lon - BOUNDS.minLon) / (BOUNDS.maxLon - BOUNDS.minLon)) * (MAP.width - MAP.padX * 2),
      MAP.padY + ((BOUNDS.maxLat - lat) / (BOUNDS.maxLat - BOUNDS.minLat)) * (MAP.height - MAP.padY * 2),
    ];
  }

  function unproject([x, y]) {
    return [
      BOUNDS.minLon + ((x - MAP.padX) / (MAP.width - MAP.padX * 2)) * (BOUNDS.maxLon - BOUNDS.minLon),
      BOUNDS.maxLat - ((y - MAP.padY) / (MAP.height - MAP.padY * 2)) * (BOUNDS.maxLat - BOUNDS.minLat),
    ];
  }

  function geometryPath(geometry) {
    const ringPath = (ring) => ring.map((point, index) => {
      const [x, y] = project(point);
      return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ") + " Z";
    const polygons = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
    return polygons.flatMap((polygon) => polygon.map(ringPath)).join(" ");
  }

  function getGeo() {
    return window.__HEATGUARD_GEO__?.features ? window.__HEATGUARD_GEO__ : null;
  }

  function selectArea(name, coordinates = null) {
    const geo = getGeo();
    const feature = geo?.features.find((item) => item.properties.name === name);
    if (!feature) return;
    state.area = name;
    byId("contribution-area").value = name;
    const [lon, lat] = coordinates || feature.properties.centroid;
    setPin(lat, lon);
    document.querySelectorAll("#contribution-map .contribution-area").forEach((path) => {
      path.classList.toggle("selected", path.dataset.area === name);
    });
  }

  function setPin(lat, lon) {
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    state.lat = Math.max(BOUNDS.minLat, Math.min(BOUNDS.maxLat, lat));
    state.lon = Math.max(BOUNDS.minLon, Math.min(BOUNDS.maxLon, lon));
    byId("contribution-latitude").textContent = state.lat.toFixed(5);
    byId("contribution-longitude").textContent = state.lon.toFixed(5);
    const [x, y] = project([state.lon, state.lat]);
    const pin = byId("contribution-pin");
    pin.hidden = false;
    pin.style.left = `${(x / MAP.width) * 100}%`;
    pin.style.top = `${(y / MAP.height) * 100}%`;
    byId("contribution-map-instruction").textContent = `${state.area || "Selected location"} · pin saved`;
  }

  function mapPointFromEvent(event) {
    const svg = byId("contribution-map");
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const matrix = svg.getScreenCTM();
    if (!matrix) return null;
    const local = point.matrixTransform(matrix.inverse());
    return [local.x, local.y];
  }

  function renderMap() {
    const geo = getGeo();
    const svg = byId("contribution-map");
    const areaSelect = byId("contribution-area");
    if (!geo || !svg || !areaSelect) return;
    svg.replaceChildren(svgEl("rect", { width: MAP.width, height: MAP.height, class: "contribution-water" }));
    const names = geo.features.map((feature) => feature.properties.name).sort((a, b) => a.localeCompare(b));
    names.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      areaSelect.appendChild(option);
    });
    geo.features.slice().reverse().forEach((feature) => {
      const name = feature.properties.name;
      const path = svgEl("path", {
        d: geometryPath(feature.geometry),
        class: "contribution-area",
        "data-area": name,
        "fill-rule": "evenodd",
        tabindex: 0,
        role: "button",
        "aria-label": `Select ${name}`,
      });
      const choose = (event) => {
        const point = mapPointFromEvent(event);
        selectArea(name, point ? unproject(point) : feature.properties.centroid);
      };
      path.addEventListener("click", choose);
      path.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectArea(name);
        }
      });
      svg.appendChild(path);
    });
  }

  function setKind(kind) {
    if (!FLOW_COPY[kind]) return;
    state.kind = kind;
    document.querySelectorAll("[data-contribution-kind]").forEach((button) => {
      const active = button.dataset.contributionKind === kind;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelectorAll("[data-contribution-fields]").forEach((fieldset) => {
      fieldset.hidden = fieldset.dataset.contributionFields !== kind;
    });
    const copy = FLOW_COPY[kind];
    byId("contribution-form-kicker").textContent = copy.step;
    byId("contribution-form-title").textContent = copy.title;
    byId("contribution-photo-title").textContent = copy.photo;
    byId("contribution-photo-field").hidden = !copy.requirePhoto;
    byId("contribution-photo").required = copy.requirePhoto;
    byId("contribution-error").hidden = true;
  }

  function readSaved() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      return [];
    }
  }

  function updateSavedCount() {
    const count = readSaved().length;
    byId("contribution-draft-count").textContent = `${count} saved`;
  }

  function saveLocally(payload) {
    const records = readSaved();
    const localRecord = { ...payload, photo: payload.photo ? { ...payload.photo, dataUrl: undefined, stored: false } : null };
    records.push(localRecord);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records.slice(-25)));
    updateSavedCount();
  }

  function getDeviceToken() {
    let token = localStorage.getItem(DEVICE_KEY);
    if (!token) {
      token = uuid();
      localStorage.setItem(DEVICE_KEY, token);
    }
    return token;
  }

  function fieldValues(fieldset) {
    const values = {};
    const formData = new FormData();
    fieldset.querySelectorAll("input, select").forEach((field) => {
      if ((field.type === "checkbox" || field.type === "radio") && !field.checked) return;
      formData.append(field.name, field.value);
    });
    for (const [key, value] of formData.entries()) {
      if (key in values) values[key] = Array.isArray(values[key]) ? [...values[key], value] : [values[key], value];
      else values[key] = value;
    }
    return values;
  }

  function fileAsDataUrl(file) {
    if (!file) return Promise.resolve(null);
    if (file.size > 5 * 1024 * 1024) return Promise.reject(new Error("Photo must be 5 MB or smaller."));
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) return Promise.reject(new Error("Use a JPG, PNG or WebP photo."));
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("The photo could not be read."));
      reader.readAsDataURL(file);
    });
  }

  function showError(message) {
    const error = byId("contribution-error");
    error.textContent = message;
    error.hidden = false;
    error.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function summaryValue(value) {
    if (Array.isArray(value)) return value.map((item) => SUMMARY_VALUES[item] || item).join(", ");
    return SUMMARY_VALUES[value] || String(value || "Not provided");
  }

  function renderSummary(payload) {
    byId("summary-kind").textContent = FLOW_COPY[payload.kind].title;
    byId("summary-step").textContent = FLOW_COPY[payload.kind].step;
    byId("summary-area").textContent = payload.area;
    byId("summary-coordinates").textContent = `${payload.coordinates.lat.toFixed(5)}, ${payload.coordinates.lon.toFixed(5)}`;
    byId("summary-channel").textContent = CHANNEL_LABELS[payload.channel] || payload.channel;
    byId("summary-time").textContent = new Intl.DateTimeFormat("en-SG", {
      dateStyle: "medium",
      timeStyle: "short",
      timeZone: "Asia/Singapore",
    }).format(new Date(payload.observedAt));
    byId("summary-evidence").textContent = payload.photo ? "Photo attached" : "No photo required";

    const details = byId("summary-details");
    details.replaceChildren();
    Object.entries(payload.details).forEach(([key, value]) => {
      if (value === "" || value == null) return;
      const row = document.createElement("div");
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = SUMMARY_LABELS[key] || key;
      description.textContent = summaryValue(value);
      row.append(term, description);
      details.appendChild(row);
    });

    const photo = byId("summary-photo");
    photo.hidden = !payload.photo;
    document.querySelector(".summary-record").classList.toggle("no-photo", !payload.photo);
    if (payload.photo) {
      byId("summary-photo-preview").src = payload.photo.dataUrl;
      byId("summary-photo-caption").textContent = `${payload.photo.name} · ${(payload.photo.size / 1024 / 1024).toFixed(1)} MB`;
    } else {
      byId("summary-photo-preview").removeAttribute("src");
    }

    byId("contribution-summary-status").textContent = "Ready to submit";
    byId("contribution-summary-status").classList.remove("submitted");
    byId("summary-actions").hidden = false;
    byId("contribution-receipt").hidden = true;
    document.querySelector(".contribution-workspace").hidden = true;
    const summary = byId("contribution-summary");
    summary.hidden = false;
    summary.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function submitPayload(payload) {
    if (location.protocol === "file:") return null;
    try {
      const response = await fetch("/api/community-submissions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`Submission service returned ${response.status}`);
      return await response.json();
    } catch (error) {
      console.info("Community API unavailable; keeping the structured record on this device.", error);
      return null;
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    byId("contribution-error").hidden = true;
    if (!state.area || !Number.isFinite(state.lat) || !Number.isFinite(state.lon)) {
      showError("Drop a pin or choose a planning area before submitting.");
      return;
    }
    if (!byId("contribution-consent").checked) {
      showError("Consent is required before this record can enter the review queue.");
      return;
    }
    const file = byId("contribution-photo").files[0];
    if (FLOW_COPY[state.kind].requirePhoto && !file) {
      showError(`Add the ${FLOW_COPY[state.kind].photo.toLowerCase()} before submitting.`);
      return;
    }
    const activeFields = document.querySelector(`[data-contribution-fields="${state.kind}"]`);
    const details = fieldValues(activeFields);
    if (state.kind === "comfort" && !details.comfort) {
      showError("Choose how the place feels before submitting.");
      return;
    }

    const submitButton = event.currentTarget.querySelector(".contribution-submit");
    submitButton.disabled = true;
    submitButton.textContent = "Preparing summary…";
    try {
      const photoDataUrl = await fileAsDataUrl(file);
      const requestId = uuid();
      const payload = {
        requestId,
        kind: state.kind,
        channel: byId("contribution-channel").value,
        area: state.area,
        coordinates: { lat: state.lat, lon: state.lon },
        observedAt: new Date().toISOString(),
        consent: true,
        deviceToken: getDeviceToken(),
        details,
        photo: photoDataUrl ? { name: file.name, type: file.type, size: file.size, dataUrl: photoDataUrl, capturedAt: new Date().toISOString() } : null,
      };
      state.pendingPayload = payload;
      renderSummary(payload);
    } catch (error) {
      showError(error.message || "The contribution could not be prepared.");
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = "Review summary";
    }
  }

  function backToEdit() {
    byId("contribution-summary").hidden = true;
    document.querySelector(".contribution-workspace").hidden = false;
    state.pendingPayload = null;
    document.querySelector(".contribution-workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function confirmSubmission() {
    const payload = state.pendingPayload;
    if (!payload) return;
    const button = byId("summary-confirm");
    button.disabled = true;
    button.textContent = "Submitting…";
    try {
      const remoteReceipt = await submitPayload(payload);
      const receipt = remoteReceipt || { id: `LOCAL-${payload.requestId.slice(0, 8).toUpperCase()}`, status: "saved-on-device" };
      saveLocally({ ...payload, receipt });
      byId("contribution-impact").textContent = FLOW_COPY[payload.kind].impact;
      byId("contribution-receipt-id").textContent = `Receipt ${receipt.id}`;
      byId("contribution-receipt-copy").textContent = remoteReceipt
        ? "Your structured report is in the council-review queue. Only approved, aggregated results can appear on the public map."
        : "This prototype record is saved on this device and is ready to sync when the collection service is connected.";
      byId("contribution-summary-status").textContent = "Submitted";
      byId("contribution-summary-status").classList.add("submitted");
      byId("summary-actions").hidden = true;
      const receiptPanel = byId("contribution-receipt");
      receiptPanel.hidden = false;
      receiptPanel.scrollIntoView({ behavior: "smooth", block: "center" });
      byId("contribution-form").reset();
      byId("contribution-photo-name").textContent = "Choose photo";
      byId("contribution-consent").checked = false;
    } catch (error) {
      showError(error.message || "The contribution could not be submitted.");
    } finally {
      button.disabled = false;
      button.textContent = "Confirm and submit";
    }
  }

  function startAnotherContribution() {
    state.pendingPayload = null;
    state.area = "";
    state.lat = null;
    state.lon = null;
    state.photoDataUrl = null;
    byId("contribution-summary").hidden = true;
    byId("contribution-receipt").hidden = true;
    document.querySelector(".contribution-workspace").hidden = false;
    byId("contribution-area").value = "";
    byId("contribution-latitude").textContent = "—";
    byId("contribution-longitude").textContent = "—";
    byId("contribution-pin").hidden = true;
    byId("contribution-map-instruction").textContent = "Click the map to drop a pin";
    document.querySelectorAll("#contribution-map .contribution-area").forEach((path) => path.classList.remove("selected"));
    setKind("proposal");
    document.querySelector(".contribution-workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function nearestArea(lat, lon) {
    const geo = getGeo();
    if (!geo) return null;
    return geo.features.reduce((nearest, feature) => {
      const [areaLon, areaLat] = feature.properties.centroid;
      const distance = Math.hypot(areaLon - lon, areaLat - lat);
      return !nearest || distance < nearest.distance ? { name: feature.properties.name, distance } : nearest;
    }, null)?.name;
  }

  function useLocation() {
    const button = byId("contribution-use-location");
    if (!navigator.geolocation) {
      showError("Location is unavailable in this browser. Choose an area and click the map instead.");
      return;
    }
    button.disabled = true;
    button.textContent = "Locating…";
    navigator.geolocation.getCurrentPosition((position) => {
      const { latitude, longitude } = position.coords;
      const area = nearestArea(latitude, longitude);
      if (!area || longitude < BOUNDS.minLon || longitude > BOUNDS.maxLon || latitude < BOUNDS.minLat || latitude > BOUNDS.maxLat) {
        showError("The detected location is outside the Singapore contribution map.");
      } else {
        selectArea(area, [longitude, latitude]);
      }
      button.disabled = false;
      button.textContent = "Use my location";
    }, () => {
      showError("Location permission was not available. Choose an area and click the map instead.");
      button.disabled = false;
      button.textContent = "Use my location";
    }, { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 });
  }

  function init() {
    if (!byId("contribution-form")) return;
    renderMap();
    updateSavedCount();
    setKind("proposal");
    document.querySelectorAll("[data-contribution-kind]").forEach((button) => {
      button.addEventListener("click", () => setKind(button.dataset.contributionKind));
    });
    byId("contribution-area").addEventListener("change", (event) => event.target.value && selectArea(event.target.value));
    byId("contribution-use-location").addEventListener("click", useLocation);
    byId("contribution-photo").addEventListener("change", (event) => {
      const file = event.target.files[0];
      byId("contribution-photo-name").textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB` : "Choose photo";
    });
    byId("contribution-form").addEventListener("submit", handleSubmit);
    byId("summary-back").addEventListener("click", backToEdit);
    byId("summary-confirm").addEventListener("click", confirmSubmission);
    byId("contribution-another").addEventListener("click", startAnotherContribution);
  }

  init();
})();

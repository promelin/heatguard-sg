const state = {
  data: null,
  localHeat: null,
  localRadius: 500,
  localSubzone: null,
  geo: null,
  station: null,
  area: null,
  index: 0,
  layer: "risk",
  showCooling: true,
  areaRows: [],
  coolingScenario: { additionalCentres: 2, greenShadePct: 15 },
  spatial: null,
  activeBlocks: null,
  activeBlocksAreaId: null,
  blockLoadToken: 0,
  green: null,
  greenPaths: null,
  greenLayers: { parks: true, cycling: true, parkConnectors: true },
  nextRefreshAt: null,
};

const LIVE_DATA_URL = window.HEATGUARD_STATIC_PUBLIC ? "heatguard-data.json" : "/api/heatguard-data.json";
const LIVE_POLL_MS = 5 * 60 * 1000;

const STATION_META = {
  S104: { name: "Woodlands Avenue 9", lon: 103.78538, lat: 1.44387 },
  S108: { name: "Marina Gardens Drive", lon: 103.8703, lat: 1.2799 },
  S109: { name: "Ang Mo Kio Avenue 5", lon: 103.8492, lat: 1.3764 },
  S111: { name: "Scotts Road", lon: 103.8365, lat: 1.31055 },
  S115: { name: "Tuas South Avenue 3", lon: 103.61843, lat: 1.29377 },
  S117: { name: "Banyan Road", lon: 103.679, lat: 1.256 },
  S121: { name: "Old Choa Chu Kang Road", lon: 103.72244, lat: 1.37288 },
  S24: { name: "Upper Changi Road North", lon: 103.9826, lat: 1.3678 },
  S43: { name: "Kim Chuan Road", lon: 103.8878, lat: 1.3399 },
  S44: { name: "Nanyang Avenue", lon: 103.68166, lat: 1.34583 },
  S50: { name: "Clementi Road", lon: 103.7768, lat: 1.3337 },
  S60: { name: "Sentosa", lon: 103.8279, lat: 1.25 },
};

const LABEL_AREAS = new Set([
  "Woodlands", "Yishun", "Sembawang", "Punggol", "Sengkang", "Pasir Ris", "Tampines",
  "Changi", "Bedok", "Hougang", "Serangoon", "Ang Mo Kio", "Toa Payoh", "Kallang",
  "Downtown Core", "Bukit Merah", "Queenstown", "Clementi", "Jurong East", "Jurong West",
  "Bukit Batok", "Bukit Panjang", "Choa Chu Kang", "Boon Lay", "Bukit Timah",
]);

const OVERVIEW_LABEL_AREAS = new Set([
  "Woodlands", "Yishun", "Punggol", "Tampines", "Changi", "Bedok", "Ang Mo Kio",
  "Downtown Core", "Bukit Merah", "Queenstown", "Jurong East", "Jurong West",
]);

const BOUNDS = { minLon: 103.58, maxLon: 104.04, minLat: 1.215, maxLat: 1.48 };
const MAP = { width: 1080, height: 650, padX: 52, padY: 46 };
const els = {};
const mapViewportState = {
  community: { zoom: 1, x: 0, y: 0 },
  home: { zoom: 1, x: 0, y: 0 },
  forecast: { zoom: 1, x: 0, y: 0 },
  cooling: { zoom: 1, x: 0, y: 0 },
};
const byId = (id) => document.getElementById(id);
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const currentLocale = () => window.HeatGuardI18n?.locale?.() || "en-SG";
const translateUI = (source, vars = {}) => window.HeatGuardI18n?.t?.(source, vars) || source;
const fmtNumber = { format: (value) => new Intl.NumberFormat(currentLocale()).format(value) };
const fmtCompact = { format: (value) => new Intl.NumberFormat(currentLocale(), { notation: "compact", maximumFractionDigits: 1 }).format(value) };

const VIEW_TITLES = {
  home: "Home",
  forecast: "Local Heat Estimate",
  "community-risk": "Community Risk",
  cooling: "Cooling Simulation",
  contribute: "Community Data",
  model: "Model & Evidence",
  readiness: "B1 Readiness",
};

const LEGACY_ROUTES = { dashboard: "forecast", communities: "community-risk", evidence: "model" };

function routeNameFromHash() {
  const raw = (location.hash || "#home").slice(1);
  return LEGACY_ROUTES[raw] || (VIEW_TITLES[raw] ? raw : "home");
}

function showView(viewName = routeNameFromHash()) {
  const target = VIEW_TITLES[viewName] ? viewName : "home";
  document.querySelectorAll("[data-view]").forEach((view) => {
    view.hidden = view.dataset.view !== target;
  });
  const moduleNav = document.querySelector(".module-entry-grid");
  const moduleAnchor = target === "home"
    ? document.querySelector(".home-view .workspace-heading")
    : document.querySelector(`#${target} > .view-header`);
  if (moduleNav && moduleAnchor && moduleNav.previousElementSibling !== moduleAnchor) {
    moduleAnchor.insertAdjacentElement("afterend", moduleNav);
  }
  document.querySelectorAll("[data-nav-view]").forEach((link) => {
    const active = link.dataset.navView === target;
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.title = `HeatGuard SG · ${translateUI(VIEW_TITLES[target])}`;
  if (state.data && (target === "community-risk" || target === "home" || target === "forecast" || target === "cooling")) {
    if (target === "cooling") {
      updateCoolingSimulation();
      ensureGreenData();
    }
    renderMap(networkAtTime(getStation().series[state.index].time));
    if (target === "forecast") updateForecastAreaPanel();
  }
}

function project([lon, lat]) {
  return [
    MAP.padX + ((lon - BOUNDS.minLon) / (BOUNDS.maxLon - BOUNDS.minLon)) * (MAP.width - MAP.padX * 2),
    MAP.padY + ((BOUNDS.maxLat - lat) / (BOUNDS.maxLat - BOUNDS.minLat)) * (MAP.height - MAP.padY * 2),
  ];
}

function areaIdForName(areaName) {
  return state.geo?.features?.find((feature) => feature.properties.name === areaName)?.properties.code || null;
}

function featureCentroid(feature) {
  return feature?.properties?.centroid || [0, 0];
}

function riskForTemperature(value) {
  if (value >= 40) return { key: "extreme", label: "Very high" };
  if (value >= 37) return { key: "high", label: "High" };
  if (value >= 34) return { key: "elevated", label: "Elevated" };
  return { key: "watch", label: "Watch" };
}

function riskForScore(score) {
  const thresholds = state.data?.publishedRisk?.thresholds || { low_medium: 45, medium_high: 70 };
  if (score >= thresholds.medium_high) return {
    key: "high", label: "High",
    title: "Activate targeted outreach before the heat peak",
    copy: "Prioritise older residents, confirm staffed cooling points, and prepare hydration and transport support.",
  };
  if (score >= thresholds.low_medium) return {
    key: "medium", label: "Medium",
    title: "Prepare local support and increase monitoring",
    copy: "Review elderly-dense blocks, check nearby cooling capacity, and ready heat-safety messaging.",
  };
  return {
    key: "low", label: "Low",
    title: "Maintain routine heat monitoring",
    copy: "Continue routine sensor review and community checks.",
  };
}

function publishedRiskPrediction(properties, heat, additionalCentres) {
  const id = properties.publicationRiskId || `area:${properties.code}`;
  const key = `${heat.toFixed(3)}:${Math.round(additionalCentres)}`;
  const score = state.data?.publishedRisk?.scores?.[id]?.[key];
  return Number.isFinite(score) ? { score } : null;
}

function fmtTime(iso) {
  return new Intl.DateTimeFormat(currentLocale(), {
    timeZone: "Asia/Singapore",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(iso));
}

function updateHomeDataTiming() {
  if (!state.data || !els.homeReadingTime || !els.homeNextUpdateTime) return;
  const live = hasLiveCurrent() || isLiveMode();
  const readingAt = state.data.current?.observedAt
    || state.data.current?.generatedAt
    || state.data.generatedAt
    || getStation()?.series?.[state.index]?.time;
  if (els.homeReadingLabel) els.homeReadingLabel.textContent = live ? "Latest data reading" : "Replay data time";
  els.homeReadingTime.textContent = readingAt ? fmtTime(readingAt) : "Data unavailable";

  const refreshSeconds = Number(state.data.refreshSeconds) || 3600;
  const generatedAt = state.data.generatedAt ? new Date(state.data.generatedAt).getTime() : NaN;
  const payloadNext = state.data.nextRefreshAt ? new Date(state.data.nextRefreshAt).getTime() : NaN;
  const nextUpdate = Number.isFinite(payloadNext)
    ? payloadNext
    : Number.isFinite(generatedAt)
      ? generatedAt + refreshSeconds * 1000
      : state.nextRefreshAt;
  els.homeNextUpdateTime.textContent = live && Number.isFinite(nextUpdate)
    ? fmtTime(new Date(nextUpdate).toISOString())
    : "Hourly refresh schedule";
  if (window.HEATGUARD_STATIC_PUBLIC) {
    els.homeReadingLabel.textContent = "Latest station observation";
    els.homeNextUpdateTime.textContent = nextUpdate > Date.now()
      ? `${fmtTime(new Date(nextUpdate).toISOString())} · scheduled`
      : "Hourly refresh active · latest result shown";
  }
  updatePublicationStatus();
}

function updatePublicationStatus() {
  const node = byId("publication-status-text");
  if (!node || !state.data) return;
  const origin = new Date(state.data.issueTime || state.data.generatedAt).getTime();
  const stale = !Number.isFinite(origin) || Date.now()-origin > 3*60*60*1000;
  const generated = state.data.generatedAt ? fmtTime(state.data.generatedAt) : "unknown time";
  node.textContent = stale
    ? `Latest available heat data: ${generated} SGT · Hourly refresh enabled.`
    : `Heat data updated ${generated} SGT · Local estimates use the published spatial model.`;
  node.closest(".publication-status")?.classList.toggle("is-stale", stale);
}

function getStation() {
  return state.data.stations.find((item) => item.id === state.station) || state.data.stations[0];
}

function isLiveMode() {
  return state.data?.mode === "live forecast";
}

function hasLiveCurrent() {
  return Boolean(state.data?.current?.stations?.length);
}

function seriesHasActual(series) {
  return series.some((point) => Number.isFinite(point.actual));
}

function networkAtTime(time) {
  return state.data.stations.flatMap((station) => {
    const point = station.series.find((item) => item.time === time);
    return point ? [{ station: station.id, ...point, risk: riskForTemperature(point.predicted) }] : [];
  }).sort((a, b) => b.predicted - a.predicted);
}

function currentNetwork() {
  if (!hasLiveCurrent()) return [];
  return state.data.current.stations.flatMap((station) => {
    const heatIndex = Number(station.heatIndex);
    if (!Number.isFinite(heatIndex)) return [];
    return [{
      station: station.id,
      predicted: heatIndex,
      actual: heatIndex,
      time: station.time || state.data.current.observedAt,
      temperature: Number(station.temperature),
      humidity: Number(station.humidity),
      risk: riskForTemperature(heatIndex),
      live: true,
    }];
  }).sort((a, b) => b.predicted - a.predicted);
}

function inverseDistanceHeat(centroid, stationRows) {
  let weighted = 0;
  let weights = 0;
  for (const row of stationRows) {
    const meta = STATION_META[row.station];
    if (!meta) continue;
    const dx = (centroid[0] - meta.lon) * Math.cos(1.35 * Math.PI / 180);
    const dy = centroid[1] - meta.lat;
    const weight = 1 / (dx * dx + dy * dy + 0.000025);
    weighted += row.predicted * weight;
    weights += weight;
  }
  return weights ? weighted / weights : 0;
}

function areaRiskModel(properties, heat, additionalCentres = 0) {
  const vulnerability = properties.seniors
    ? clamp(
      (properties.seniorDensity / 4200) * 45
      + (properties.seniorShare / 30) * 35
      + ((properties.elderly75Share || 0) / 15) * 20,
      0,
      100,
    )
    : 0;
  const centreCount = properties.coolingCentres + additionalCentres;
  const accessPer10k = properties.seniors ? (centreCount / properties.seniors) * 10000 : 0;
  const accessGap = properties.seniors ? 100 - clamp((accessPer10k / 2.2) * 100, 0, 100) : 25;
  const heatScore = clamp(((heat - 31.5) / 9.5) * 100, 0, 100);
  const prediction = publishedRiskPrediction(properties, heat, additionalCentres);
  const fallbackScore = clamp(0.55 * heatScore + 0.30 * vulnerability + 0.15 * accessGap, 0, 100);
  const score = prediction?.score ?? fallbackScore;
  return {
    score,
    heatScore,
    vulnerability,
    accessGap,
    accessPer10k,
    centreCount,
    heatContribution: heatScore,
    elderlyContribution: vulnerability,
    accessContribution: accessGap,
    classProbabilities: prediction?.probabilities || null,
    risk: riskForScore(score),
  };
}

function computeAreas(stationRows) {
  return state.geo.features.map((feature) => {
    const p = feature.properties;
    const heat = inverseDistanceHeat(p.centroid, stationRows);
    const model = areaRiskModel(p, heat);
    return {
      feature,
      name: p.name,
      heat,
      ...model,
      ...p,
    };
  }).sort((a, b) => b.score - a.score);
}

function computeSubzones(stationRows, areaId = null) {
  if (!state.spatial?.subzones?.features) return [];
  return state.spatial.subzones.features
    .filter((feature) => !areaId || feature.properties.planning_area_id === areaId)
    .map((feature) => {
      const p = feature.properties;
      const parent = state.geo.features.find((item) => item.properties.code === p.planning_area_id)?.properties || {};
      const seniors = p.elderly_65plus_2026 || 0;
      const residents = p.population_2026 || 0;
      const elderly75 = p.elderly_75plus_2026 || 0;
      const areaKm2 = Math.max(.01, p.area_km2 || parent.areaKm2 * ((p.population_2026 || 0) / Math.max(1, parent.residents || 1)));
      const heat = inverseDistanceHeat(featureCentroid(feature), stationRows);
      const model = areaRiskModel({
        publicationRiskId: `subzone:${p.id}`,
        seniors,
        seniorDensity: seniors / areaKm2,
        seniorShare: p.elderly_65plus_share_pct || 0,
        elderly75Share: residents ? elderly75 / residents * 100 : 0,
        coolingCentres: p.cooling_centres || 0,
      }, heat);
      return {
        feature,
        name: p.display_name,
        planningArea: p.planning_area,
        heat,
        residents,
        seniors,
        seniorShare: p.elderly_65plus_share_pct || 0,
        hdbBlocks: p.hdb_building_count || 0,
        confidence: p.block_estimate_confidence,
        ...model,
      };
    });
}

function marginalShadeCooling(row) {
  const greenScore = clamp(Number(row.greenBaselineScore) || 0, 0, 100);
  const degreesPerPct = .01 + .02 * (1 - greenScore / 100);
  return state.coolingScenario.greenShadePct * degreesPerPct;
}

function simulatedCoolingRows() {
  return state.areaRows.map((row) => {
    if (row.name !== state.area) return { ...row, simulated: false, simulatedScore: row.score, simulatedRisk: row.risk };
    const heatReduction = marginalShadeCooling(row);
    const simulatedHeat = Math.max(0, row.heat - heatReduction);
    const model = areaRiskModel(row, simulatedHeat, state.coolingScenario.additionalCentres);
    return {
      ...row,
      simulated: true,
      simulatedHeat,
      simulatedScore: model.score,
      simulatedRisk: model.risk,
      simulatedAccessGap: model.accessGap,
      simulatedCentreCount: model.centreCount,
      heatReduction,
    };
  }).sort((a, b) => b.simulatedScore - a.simulatedScore);
}

function forecastWindowPoints() {
  const series = getStation()?.series || [];
  if (series.length <= 24) return series.slice();
  const start = clamp(state.index, 0, Math.max(0, series.length - 24));
  return series.slice(start, start + 24);
}

function areaForecastSeries(areaName) {
  const feature = state.geo?.features?.find((item) => item.properties.name === areaName);
  if (!feature) return [];
  return forecastWindowPoints().map((point) => ({
    time: point.time,
    predicted: inverseDistanceHeat(feature.properties.centroid, networkAtTime(point.time)),
    actual: null,
  }));
}

function computeForecastPeakAreas() {
  if (!state.geo?.features) return [];
  if (state.localHeat?.areas) {
    const byName = new Map(state.localHeat.areas.map((row) => [row.name, row]));
    const radius = String(state.localRadius);
    return state.geo.features.map((feature) => {
      const p = feature.properties;
      const local = byName.get(p.name)?.estimates?.[radius];
      const heat = Number(local?.estimate_c ?? state.localHeat.background_heat_index_c ?? 34);
      return {
        feature,
        name: p.name,
        heat,
        local,
        risk: riskForTemperature(heat),
        score: clamp(((heat - 31.5) / 9.5) * 100, 0, 100),
        ...p,
      };
    }).sort((a, b) => b.heat - a.heat);
  }
  return state.geo.features.map((feature) => {
    const p = feature.properties;
    const series = areaForecastSeries(p.name);
    const peak = series.reduce((best, point) => point.predicted > (best?.predicted ?? -Infinity) ? point : best, null);
    const heat = peak?.predicted || 0;
    return {
      feature,
      name: p.name,
      heat,
      peakTime: peak?.time,
      risk: riskForTemperature(heat),
      score: clamp(((heat - 31.5) / 9.5) * 100, 0, 100),
      ...p,
    };
  }).sort((a, b) => b.heat - a.heat);
}

function localSubzoneRows() {
  if (!state.localHeat?.subzones || !state.spatial?.subzones?.features) return [];
  const radius = String(state.localRadius);
  const byId = new Map(state.localHeat.subzones.map((row) => [row.id, row]));
  return state.spatial.subzones.features.map((feature) => {
    const source = byId.get(feature.properties.id);
    const local = source?.estimates?.[radius];
    if (!source || !local) return null;
    const heat = Number(local.estimate_c);
    return {
      feature,
      name: source.name,
      planningArea: source.planning_area,
      planningAreaId: source.planning_area_id,
      heat,
      local,
      risk: riskForTemperature(heat),
      score: clamp(((heat - 31.5) / 9.5) * 100, 0, 100),
    };
  }).filter(Boolean).sort((a, b) => b.heat - a.heat);
}

function guidanceForHeat(value) {
  if (value >= 40) return "Very high heat risk: avoid strenuous outdoor activity around the peak, use cooled spaces, and prioritise checks on vulnerable residents.";
  if (value >= 37) return "High heat risk: plan shaded breaks, hydration, and earlier outreach before the forecast peak.";
  if (value >= 34) return "Elevated heat risk: stay hydrated, reduce prolonged sun exposure, and monitor conditions through the day.";
  return "Lower heat risk: normal precautions remain appropriate; continue monitoring the hourly outlook.";
}

function guidanceForLocalHeat(value) {
  if (value >= 40) return "Very high possible local heat: prioritise shade, cooled spaces and checks on vulnerable residents; verify with on-site observations.";
  if (value >= 37) return "High possible local heat: plan shaded breaks, hydration and targeted local checks; verify conditions before action.";
  if (value >= 34) return "Elevated possible local heat: reduce prolonged sun exposure, use nearby shade and compare the estimate with current observations.";
  return "Lower possible local heat: normal precautions remain appropriate; continue checking current station conditions.";
}

function updateForecastAreaPanel() {
  if (!els.forecastAreaTitle || !state.data || !state.geo) return;
  const areaName = state.area || computeForecastPeakAreas()[0]?.name;
  const subzone = state.localSubzone
    ? localSubzoneRows().find((item) => item.name === state.localSubzone)
    : null;
  const row = subzone || computeForecastPeakAreas().find((item) => item.name === areaName);
  if (!areaName || !row) return;
  state.area = subzone?.planningArea || areaName;
  const risk = riskForTemperature(row.heat);
  const spatial = row.local?.spatial || {};
  els.forecastAreaTitle.textContent = subzone ? `${subzone.name} · ${subzone.planningArea}` : areaName;
  els.forecastAreaRiskChip.textContent = `${risk.label} risk`;
  els.forecastAreaRiskChip.className = `risk-chip ${risk.key}`;
  els.forecastAreaPeak.textContent = `${row.heat.toFixed(1)}°C`;
  els.forecastAreaPeakTime.textContent = `${state.localRadius >= 1000 ? `${state.localRadius / 1000} km` : `${state.localRadius} m`} surroundings · ${Number(state.localHeat?.background_heat_index_c || 34).toFixed(1)}°C network background`;
  els.forecastAreaGuidance.textContent = guidanceForLocalHeat(row.heat);
  els.forecastRiskAlert.className = `forecast-risk-alert ${risk.key}`;
  if (els.localParkCover) els.localParkCover.textContent = `${Number(spatial.park_cover_proxy_pct || 0).toFixed(1)}%`;
  if (els.localBuildingCover) els.localBuildingCover.textContent = `${Number(spatial.building_coverage_pct || 0).toFixed(1)}%`;
  if (els.localBuildingDensity) els.localBuildingDensity.textContent = `${Number(spatial.building_density_per_km2 || 0).toFixed(0)} / km²`;
  if (els.localCoastDistance) {
    const distance = Number(spatial.distance_to_coast_m || 0);
    els.localCoastDistance.textContent = distance >= 1000 ? `${(distance / 1000).toFixed(1)} km` : `${distance.toFixed(0)} m`;
  }
}

function selectForecastArea(areaName) {
  state.area = areaName;
  state.localSubzone = null;
  updateForecastAreaPanel();
  renderMap(networkAtTime(getStation().series[state.index].time));
  document.querySelector(".forecast-area-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function selectForecastSubzone(row) {
  state.area = row.planningArea;
  state.localSubzone = row.name;
  updateForecastAreaPanel();
  renderMap(networkAtTime(getStation().series[state.index].time));
  document.querySelector(".forecast-area-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function selectStation(stationId, time) {
  state.station = stationId;
  els.stationSelect.value = stationId;
  const station = getStation();
  const matchingIndex = station.series.findIndex((item) => item.time === time);
  state.index = matchingIndex >= 0 ? matchingIndex : Math.floor(station.series.length / 2);
  syncSlider();
  updateDashboard();
}

function selectArea(areaName, scroll = false) {
  state.area = areaName;
  ensureBlocksForArea(areaName);
  updateAreaDetails();
  updateCoolingSimulation();
  renderMap(networkAtTime(getStation().series[state.index].time));
  renderAreaList();
  if (scroll) document.querySelector(".risk-score-panel")?.scrollIntoView({ behavior: "smooth", block: "center" });
}

function updateDashboard() {
  const station = getStation();
  state.index = clamp(state.index, 0, station.series.length - 1);
  const point = station.series[state.index];
  const temperatureRisk = riskForTemperature(point.predicted);
  const network = networkAtTime(point.time);
  const mean = network.reduce((sum, item) => sum + item.predicted, 0) / Math.max(1, network.length);
  state.areaRows = computeAreas(network);
  if (!state.area || !state.areaRows.some((row) => row.name === state.area)) state.area = state.areaRows[0]?.name;
  const priority = state.areaRows.filter((row) => row.risk.key === "high" && row.seniors >= 1000);
  const exposed = priority.reduce((sum, row) => sum + row.seniors, 0);
  const observedNetwork = currentNetwork();
  const homeNetwork = observedNetwork.length ? observedNetwork : network;
  const homeMean = homeNetwork.reduce((sum, item) => sum + item.predicted, 0) / Math.max(1, homeNetwork.length);
  const homeHighCount = homeNetwork.filter((item) => item.predicted >= 37).length;
  const homeObservedAt = hasLiveCurrent() ? state.data.current.observedAt : point.time;
  const meta = STATION_META[station.id];
  const hasActual = Number.isFinite(point.actual);
  const error = hasActual ? point.predicted - point.actual : null;

  els.stationTitle.textContent = meta ? `${station.id} · ${meta.name}` : `Station ${station.id}`;
  els.predicted.textContent = point.predicted.toFixed(1);
  els.actual.textContent = hasActual ? point.actual.toFixed(1) : "—";
  els.rmse.textContent = isLiveMode()
    ? state.data.model.high37Rmse?.toFixed(2) ?? "—"
    : station.rmse?.toFixed(2) ?? "—";
  els.delta.textContent = hasActual
    ? `${error >= 0 ? "+" : ""}${error.toFixed(2)}°C vs observed replay`
    : "Live forecast · observation pending";
  els.riskChip.textContent = `${temperatureRisk.label} heat`;
  els.riskChip.className = `risk-chip ${temperatureRisk.key}`;
  if (els.networkMean) els.networkMean.textContent = `${mean.toFixed(1)}°C`;
  if (els.priorityAreaCount) els.priorityAreaCount.textContent = String(priority.length);
  if (els.seniorsExposed) els.seniorsExposed.textContent = fmtCompact.format(exposed);
  if (els.mapStationCount) els.mapStationCount.textContent = `${network.length} forecast stations · ${state.geo.features.length} planning areas`;
  if (els.homeMapStationCount) {
    els.homeMapStationCount.textContent = hasLiveCurrent()
      ? `${homeNetwork.length} live stations · ${state.geo.features.length} planning areas`
      : `${homeNetwork.length} forecast replay stations · ${state.geo.features.length} planning areas`;
  }
  if (els.headerRecall) els.headerRecall.textContent = `${(state.data.model.high37Recall * 100).toFixed(1)}%`;
  document.querySelectorAll("[data-model-recall]").forEach((item) => {
    item.textContent = `${(state.data.model.high37Recall * 100).toFixed(1)}%`;
  });
  document.querySelectorAll("[data-forecast-status]").forEach((item) => {
    item.textContent = isLiveMode() ? "Forecast target" : "Replay target";
  });
  document.querySelectorAll("[data-mode-label]").forEach((item) => {
    item.textContent = isLiveMode() ? "Live state" : "Demo state";
  });
  document.querySelectorAll("[data-mode-value]").forEach((item) => {
    item.textContent = isLiveMode() ? "data.gov.sg · V4 scheduled hourly" : state.data.mode;
  });
  els.actualLabel.textContent = hasActual ? "Observed replay" : "Observed value";
  els.actualNote.textContent = hasActual ? "°C · evaluation only" : "Pending after target time";
  els.rmseLabel.textContent = isLiveMode() ? "V4 high-heat RMSE" : "Station RMSE";
  els.rmseNote.textContent = isLiveMode() ? "°C · ≥37°C test set" : "°C · full 2024";
  els.actualLegend.hidden = !seriesHasActual(station.series);
  if (els.forecastContextTime) els.forecastContextTime.textContent = fmtTime(point.time);
  if (els.communityContextTime) els.communityContextTime.textContent = fmtTime(point.time);
  if (els.homeLiveCount) els.homeLiveCount.textContent = hasLiveCurrent() ? String(homeNetwork.length) : "Replay";
  if (els.homeLiveMean) els.homeLiveMean.textContent = `${homeMean.toFixed(1)}°C`;
  if (els.homeForecastPeak) els.homeForecastPeak.textContent = `${Number(state.data.networkMean).toFixed(1)}°C`;
  if (els.homeMapNetworkMean) els.homeMapNetworkMean.textContent = `${homeMean.toFixed(1)}°C`;
  if (els.homeMapPriorityCount) els.homeMapPriorityCount.textContent = String(homeHighCount);
  if (els.homeMapSeniorsExposed) els.homeMapSeniorsExposed.textContent = fmtTime(homeObservedAt);
  document.querySelectorAll("[data-home-source-label]").forEach((item) => {
    item.textContent = hasLiveCurrent() ? "Network heat index" : "Forecast replay mean";
  });
  updateHomeDataTiming();

  updateCoolingSimulation();
  renderMap(network);
  updateForecastAreaPanel();
  renderAreaList();
  updateAreaDetails();
}

function geometryPath(geometry) {
  const ringPath = (ring) => ring.map((point, index) => {
    const [x, y] = project(point);
    return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ") + " Z";
  const polygons = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
  return polygons.flatMap((polygon) => polygon.map(ringPath)).join(" ");
}

function lineGeometryPath(geometry) {
  const lines = geometry.type === "LineString" ? [geometry.coordinates] : geometry.coordinates;
  return lines.map((line) => line.map((point, index) => {
    const [x, y] = project(point);
    return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ")).join(" ");
}

async function ensureBlocksForArea(areaName) {
  if (!state.spatial || !areaName || !window.HeatGuardSpatial) return;
  const areaId = areaIdForName(areaName);
  if (!areaId || (state.activeBlocksAreaId === areaId && state.activeBlocks)) return;
  const token = ++state.blockLoadToken;
  state.activeBlocks = null;
  state.activeBlocksAreaId = areaId;
  updateSpatialDetailStatus(getMapViews().find((view) => view.key === "community"));
  try {
    const blocks = await window.HeatGuardSpatial.loadBlocks(areaId);
    if (token !== state.blockLoadToken) return;
    state.activeBlocks = blocks;
    updateSpatialDetailStatus(getMapViews().find((view) => view.key === "community"));
    if (state.data) renderMap(networkAtTime(getStation().series[state.index].time));
  } catch (error) {
    console.warn(`HDB buildings could not be loaded for ${areaName}.`, error);
  }
}

function buildGreenPaths(green) {
  return {
    parks: green.parks.features.map((feature) => geometryPath(feature.geometry)).join(" "),
    cycling: green.cycling.features.map((feature) => lineGeometryPath(feature.geometry)).join(" "),
    parkConnectors: green.parkConnectors.features.map((feature) => lineGeometryPath(feature.geometry)).join(" "),
  };
}

async function ensureGreenData() {
  if (state.green || !window.HeatGuardSpatial) return;
  try {
    state.green = await window.HeatGuardSpatial.loadGreen();
    state.greenPaths = buildGreenPaths(state.green);
    if (state.data) renderMap(networkAtTime(getStation().series[state.index].time));
  } catch (error) {
    console.warn("Mapped green infrastructure could not be loaded.", error);
  }
}

function appendSubzoneDetail(svg, view, stationRows, mapLayer) {
  if (!state.spatial || !["community", "cooling", "forecast"].includes(view.key)) return;
  const rows = view.key === "forecast" ? localSubzoneRows() : computeSubzones(stationRows);
  const group = svgEl("g", { class: "subzone-layer" });
  for (const row of rows) {
    const parent = state.geo.features.find((feature) => feature.properties.code === row.feature.properties.planning_area_id)?.properties;
    const selected = view.key === "forecast"
      ? row.name === state.localSubzone
      : parent?.name === state.area;
    const path = svgEl("path", {
      d: geometryPath(row.feature.geometry),
      class: `subzone-shape ${selected ? "selected-area" : ""}`,
      fill: areaFill(row, mapLayer),
      "fill-rule": "evenodd",
      tabindex: view.key === "forecast" ? 0 : -1,
      role: view.key === "forecast" ? "button" : "img",
      "aria-label": view.key === "forecast"
        ? `${row.name}, ${row.planningArea}, local heat estimate ${row.heat.toFixed(1)} degrees Celsius`
        : `${row.name}, ${parent?.name || row.planningArea} subzone`,
    });
    const detail = view.key === "forecast"
      ? `<strong>${row.name}</strong><span>${row.planningArea} subzone</span><span>${row.heat.toFixed(1)}°C local estimate · ${state.localRadius} m surroundings</span><span>Click for local environment inputs</span>`
      : `<strong>${row.name}</strong><span>${parent?.name || row.planningArea} subzone</span><span>${row.heat.toFixed(1)}°C heat estimate · score ${row.score.toFixed(0)}</span><span>${fmtNumber.format(row.seniors)} residents aged 65+ · official 2026</span><span>${fmtNumber.format(row.hdbBlocks)} HDB buildings</span>`;
    path.addEventListener("pointermove", (event) => showMapTooltip(event, detail, view));
    path.addEventListener("pointerleave", () => hideMapTooltip(view));
    const activate = () => view.key === "forecast"
      ? selectForecastSubzone(row)
      : parent?.name && selectArea(parent.name);
    path.addEventListener("click", activate);
    if (view.key === "forecast") {
      path.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); }
      });
    }
    group.appendChild(path);
    if ((view.key === "forecast" ? row.planningArea === state.area : parent?.name === state.area)) {
      const [x, y] = project(featureCentroid(row.feature));
      const label = svgEl("text", { x, y, class: "subzone-label", "text-anchor": "middle", "pointer-events": "none" });
      label.textContent = row.name;
      group.appendChild(label);
    }
  }
  svg.appendChild(group);
}

function appendBlockDetail(svg, view, areaRows) {
  if (!state.activeBlocks?.features || (view.key !== "community" && view.key !== "cooling")) return;
  const areaId = areaIdForName(state.area);
  if (state.activeBlocksAreaId !== areaId) return;
  const parent = areaRows.find((row) => row.name === state.area);
  const group = svgEl("g", { class: "hdb-block-layer" });
  for (const feature of state.activeBlocks.features) {
    const p = feature.properties;
    const exposure = Number.isFinite(p.elderly_exposure_score) ? p.elderly_exposure_score : 0;
    const blockScore = clamp((parent?.heatScore || 0) * .55 + exposure * .45, 0, 100);
    const path = svgEl("path", {
      d: geometryPath(feature.geometry),
      class: "hdb-block",
      fill: rampColor(blockScore, [[0, "#fff4ec"], [45, "#f7c49e"], [65, "#ed765f"], [80, "#cf2e48"], [95, "#78132d"]]),
      "fill-rule": "evenodd",
    });
    path.addEventListener("pointermove", (event) => showMapTooltip(event,
      `<strong>${p.display_name}${p.street ? ` · ${p.street}` : ""}</strong><span>${p.subzone ? `${p.subzone} · ` : ""}${p.postal_code || "Postal code unavailable"}</span><span>${fmtNumber.format(Math.round(p.estimated_elderly_65plus_2026 || 0))} estimated residents aged 65+</span><span>${fmtNumber.format(p.total_dwelling_units || 0)} dwellings · ${p.estimate_confidence || "Unrated"} estimate confidence</span>`, view
    ));
    path.addEventListener("pointerleave", () => hideMapTooltip(view));
    group.appendChild(path);
  }
  svg.appendChild(group);
}

function appendGreenInfrastructure(svg, view) {
  if (view.key !== "cooling" || !state.greenPaths) return;
  const group = svgEl("g", { class: "green-infrastructure-layer", "pointer-events": "none" });
  if (state.greenLayers.parks) group.appendChild(svgEl("path", { d: state.greenPaths.parks, class: "mapped-parks", "fill-rule": "evenodd" }));
  if (state.greenLayers.cycling) group.appendChild(svgEl("path", { d: state.greenPaths.cycling, class: "mapped-cycling" }));
  if (state.greenLayers.parkConnectors) group.appendChild(svgEl("path", { d: state.greenPaths.parkConnectors, class: "mapped-park-connectors" }));
  svg.appendChild(group);
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16));
}

function rgbToHex(rgb) {
  return `#${rgb.map((value) => Math.round(value).toString(16).padStart(2, "0")).join("")}`;
}

function mixColor(a, b, t) {
  const start = hexToRgb(a);
  const end = hexToRgb(b);
  return rgbToHex(start.map((value, index) => value + (end[index] - value) * clamp(t, 0, 1)));
}

function rampColor(value, stops) {
  if (value <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i += 1) {
    if (value <= stops[i][0]) {
      const ratio = (value - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
      return mixColor(stops[i - 1][1], stops[i][1], ratio);
    }
  }
  return stops.at(-1)[1];
}

function areaFill(row, layer = state.layer) {
  if (layer === "heat") {
    return rampColor(row.heat, [[31, "#fff6ef"], [34, "#ffd8c9"], [37, "#f58a72"], [40, "#d9344d"], [43, "#861128"]]);
  }
  if (layer === "elderly") {
    return rampColor(row.seniorDensity, [[0, "#f5f2ed"], [800, "#f5d8d3"], [1800, "#e59a98"], [3000, "#c94355"], [4400, "#78132d"]]);
  }
  return rampColor(row.score, [[20, "#f4f1ea"], [45, "#f6cf84"], [65, "#ef8066"], [80, "#cf2e48"], [95, "#7a1029"]]);
}

function getMapViews() {
  return [
    {
      key: "community",
      svg: els.map,
      tooltip: els.mapTooltip,
      legend: els.mapLegend,
      stationCount: els.mapStationCount,
      stage: els.mapStage,
      scale: els.mapScale,
      scaleLabel: els.mapScaleLabel,
      scaleLine: els.mapScaleLine,
    },
    {
      key: "home",
      svg: els.homeMap,
      tooltip: els.homeMapTooltip,
      legend: els.homeMapLegend,
      stationCount: els.homeMapStationCount,
      stage: els.homeMapStage,
      scale: els.homeMapScale,
      scaleLabel: els.homeMapScaleLabel,
      scaleLine: els.homeMapScaleLine,
    },
    {
      key: "forecast",
      svg: els.forecastMap,
      tooltip: els.forecastMapTooltip,
      legend: els.forecastMapLegend,
      stationCount: null,
      stage: els.forecastMapStage,
      scale: els.forecastMapScale,
      scaleLabel: els.forecastMapScaleLabel,
      scaleLine: els.forecastMapScaleLine,
    },
    {
      key: "cooling",
      svg: els.coolingMap,
      tooltip: els.coolingMapTooltip,
      legend: els.coolingMapLegend,
      stationCount: null,
      stage: els.coolingMapStage,
      scale: els.coolingMapScale,
      scaleLabel: els.coolingMapScaleLabel,
      scaleLine: els.coolingMapScaleLine,
    },
  ].filter((view) => view.svg);
}

function getViewBoxForMap(key) {
  const viewport = mapViewportState[key] || mapViewportState.community;
  const zoom = clamp(viewport.zoom, 1, 7);
  const width = MAP.width / zoom;
  const height = MAP.height / zoom;
  const maxX = MAP.width - width;
  const maxY = MAP.height - height;
  viewport.zoom = zoom;
  viewport.x = clamp(viewport.x, 0, maxX);
  viewport.y = clamp(viewport.y, 0, maxY);
  return { x: viewport.x, y: viewport.y, width, height };
}

function applyMapViewport(view) {
  if (!view?.svg) return;
  const box = getViewBoxForMap(view.key);
  const zoom = mapViewportState[view.key].zoom;
  view.svg.setAttribute("viewBox", `${box.x.toFixed(2)} ${box.y.toFixed(2)} ${box.width.toFixed(2)} ${box.height.toFixed(2)}`);
  view.stage?.classList.toggle("is-zoomed", zoom > 1.01);
  view.stage?.classList.toggle("labels-detailed", zoom >= 1.45);
  view.stage?.classList.toggle("labels-local", zoom >= 2.7);
  view.stage?.classList.toggle("labels-stations", zoom >= 3.2);
  view.stage?.style.setProperty("--map-label-scale", (1 / zoom).toFixed(4));
  updateMapScale(view);
  updateSpatialDetailStatus(view);
}

function updateSpatialDetailStatus(view) {
  if (view?.key !== "community" || !els.spatialDetailStatus) return;
  const zoom = mapViewportState.community.zoom;
  if (zoom >= 2.7) {
    const count = state.activeBlocksAreaId === areaIdForName(state.area) ? state.activeBlocks?.features?.length : null;
    els.spatialDetailStatus.textContent = count == null
      ? `Loading HDB buildings · ${state.area || "selected area"}`
      : `${fmtNumber.format(count)} HDB buildings · ${state.area} · population estimates`;
  } else if (zoom >= 1.45) {
    els.spatialDetailStatus.textContent = `${state.spatial?.metadata?.subzones || 332} subzones · official 2026 demographics`;
  } else {
    els.spatialDetailStatus.textContent = `${state.spatial?.metadata?.planningAreas || 55} planning areas · official 2026 demographics`;
  }
}

function updateMapScale(view) {
  if (!view?.svg || !view.scale || !view.scaleLine || !view.scaleLabel) return;
  const rect = view.svg.getBoundingClientRect();
  if (!rect.width) return;
  const box = getViewBoxForMap(view.key);
  const midLat = (BOUNDS.minLat + BOUNDS.maxLat) / 2;
  const geoWidthKm = (BOUNDS.maxLon - BOUNDS.minLon) * 111.32 * Math.cos(midLat * Math.PI / 180);
  const projectedWidth = MAP.width - MAP.padX * 2;
  const svgUnitsPerKm = projectedWidth / geoWidthKm;
  const pixelsPerKm = svgUnitsPerKm * (rect.width / box.width);
  const candidates = [20, 10, 5, 2, 1, .5, .2];
  let distance = candidates.find((km) => km * pixelsPerKm <= 145 && km * pixelsPerKm >= 60);
  if (!distance) distance = candidates.find((km) => km * pixelsPerKm <= 145) || .2;
  const px = clamp(distance * pixelsPerKm, 42, 150);
  view.scale.style.setProperty("--scale-width", `${px.toFixed(0)}px`);
  view.scaleLabel.textContent = distance >= 1 ? `${distance} km` : `${Math.round(distance * 1000)} m`;
}

function setMapZoom(view, nextZoom, anchor = null) {
  if (!view?.svg) return;
  const viewport = mapViewportState[view.key];
  const oldBox = getViewBoxForMap(view.key);
  const newZoom = clamp(nextZoom, 1, 7);
  const newWidth = MAP.width / newZoom;
  const newHeight = MAP.height / newZoom;
  let anchorX = .5;
  let anchorY = .5;
  if (anchor) {
    const rect = view.svg.getBoundingClientRect();
    if (rect.width && rect.height) {
      anchorX = clamp((anchor.clientX - rect.left) / rect.width, 0, 1);
      anchorY = clamp((anchor.clientY - rect.top) / rect.height, 0, 1);
    }
  }
  const focusX = oldBox.x + oldBox.width * anchorX;
  const focusY = oldBox.y + oldBox.height * anchorY;
  viewport.zoom = newZoom;
  viewport.x = focusX - newWidth * anchorX;
  viewport.y = focusY - newHeight * anchorY;
  applyMapViewport(view);
  if ((view.key === "community" || view.key === "cooling") && newZoom >= 2.7) ensureBlocksForArea(state.area);
}

function resetMapZoom(view) {
  const viewport = mapViewportState[view.key];
  viewport.zoom = 1;
  viewport.x = 0;
  viewport.y = 0;
  applyMapViewport(view);
}

function initMapInteractions() {
  getMapViews().forEach((view) => {
    const drag = { active: false, moved: false, pointerId: null, clientX: 0, clientY: 0, startX: 0, startY: 0 };
    view.stage?.querySelectorAll("[data-map-zoom]").forEach((button) => {
      button.addEventListener("click", () => {
        const action = button.dataset.mapZoom;
        if (action === "reset") resetMapZoom(view);
        else if (action === "in") setMapZoom(view, mapViewportState[view.key].zoom * 1.5);
        else setMapZoom(view, mapViewportState[view.key].zoom / 1.5);
      });
    });
    view.svg.addEventListener("dblclick", (event) => {
      event.preventDefault();
      setMapZoom(view, mapViewportState[view.key].zoom * 1.5, event);
    });
    view.svg.addEventListener("pointerdown", (event) => {
      if (mapViewportState[view.key].zoom <= 1.01 || event.button !== 0) return;
      drag.active = true;
      drag.moved = false;
      drag.pointerId = event.pointerId;
      drag.clientX = event.clientX;
      drag.clientY = event.clientY;
      drag.startX = mapViewportState[view.key].x;
      drag.startY = mapViewportState[view.key].y;
    });
    view.svg.addEventListener("pointermove", (event) => {
      if (!drag.active || event.pointerId !== drag.pointerId) return;
      const rect = view.svg.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const box = getViewBoxForMap(view.key);
      const dx = event.clientX - drag.clientX;
      const dy = event.clientY - drag.clientY;
      if (Math.hypot(dx, dy) > 3 && !drag.moved) {
        drag.moved = true;
        view.svg.setPointerCapture?.(event.pointerId);
        view.stage?.classList.add("is-panning");
      }
      if (!drag.moved) return;
      mapViewportState[view.key].x = drag.startX - dx * (box.width / rect.width);
      mapViewportState[view.key].y = drag.startY - dy * (box.height / rect.height);
      applyMapViewport(view);
    });
    const finishDrag = (event) => {
      if (!drag.active || event.pointerId !== drag.pointerId) return;
      drag.active = false;
      if (view.svg.hasPointerCapture?.(event.pointerId)) view.svg.releasePointerCapture(event.pointerId);
      view.stage?.classList.remove("is-panning");
    };
    view.svg.addEventListener("pointerup", finishDrag);
    view.svg.addEventListener("pointercancel", finishDrag);
    view.svg.addEventListener("click", (event) => {
      if (!drag.moved) return;
      event.preventDefault();
      event.stopPropagation();
      drag.moved = false;
    }, true);
  });
  window.addEventListener("resize", () => getMapViews().forEach(updateMapScale));
}

function appendMapDefinitions(svg, key) {
  const defs = svgEl("defs");
  const patternId = `terrain-hatch-${key}`;
  const shadowId = `marker-shadow-${key}`;
  const pattern = svgEl("pattern", { id: patternId, width: 9, height: 9, patternUnits: "userSpaceOnUse", patternTransform: "rotate(32)" });
  pattern.appendChild(svgEl("line", { x1: 0, y1: 0, x2: 0, y2: 9, stroke: "#2f7b62", "stroke-width": 2, opacity: .25 }));
  defs.appendChild(pattern);
  const shadow = svgEl("filter", { id: shadowId, x: "-80%", y: "-80%", width: "260%", height: "260%" });
  shadow.appendChild(svgEl("feDropShadow", { dx: 0, dy: 4, stdDeviation: 4, "flood-color": "#651826", "flood-opacity": .28 }));
  defs.appendChild(shadow);
  svg.appendChild(defs);
  return { patternId, shadowId };
}

function appendTerrain(svg, patternId, areaRows) {
  const terrainAreas = ["Central Water Catchment", "Western Water Catchment", "Bukit Timah"];
  for (const name of terrainAreas) {
    const row = areaRows.find((item) => item.name === name);
    if (!row) continue;
    svg.appendChild(svgEl("path", {
      d: geometryPath(row.feature.geometry), fill: `url(#${patternId})`, class: "terrain-overlay", "pointer-events": "none",
    }));
  }
  const contourCoordinates = [
    [[103.759, 1.337], [103.777, 1.348], [103.797, 1.353], [103.816, 1.347]],
    [[103.766, 1.344], [103.785, 1.359], [103.807, 1.365], [103.826, 1.355]],
    [[103.777, 1.354], [103.796, 1.374], [103.817, 1.38], [103.835, 1.37]],
  ];
  for (const line of contourCoordinates) {
    const d = line.map((point, index) => {
      const [x, y] = project(point);
      return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    svg.appendChild(svgEl("path", { d, class: "contour-line", "pointer-events": "none" }));
  }
  const roads = [
    { name: "PIE", points: [[103.665, 1.342], [103.75, 1.335], [103.84, 1.34], [103.94, 1.35], [104.00, 1.36]] },
    { name: "CTE", points: [[103.834, 1.444], [103.835, 1.39], [103.844, 1.335], [103.852, 1.288]] },
    { name: "KJE", points: [[103.69, 1.372], [103.72, 1.39], [103.76, 1.412], [103.79, 1.43]] },
  ];
  for (const road of roads) {
    const d = road.points.map((point, index) => {
      const [x, y] = project(point);
      return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    svg.appendChild(svgEl("path", { d, class: "map-road", "pointer-events": "none" }));
  }
}

function appendCoolingScenario(svg, row) {
  const [x, y] = project(row.centroid);
  const shadeLength = 24 + state.coolingScenario.greenShadePct * 1.8;
  if (state.coolingScenario.greenShadePct > 0) {
    const corridor = svgEl("path", {
      d: `M${(x - shadeLength).toFixed(1)},${(y + 8).toFixed(1)} C${(x - shadeLength / 3).toFixed(1)},${(y - 17).toFixed(1)} ${(x + shadeLength / 3).toFixed(1)},${(y + 21).toFixed(1)} ${(x + shadeLength).toFixed(1)},${(y - 7).toFixed(1)}`,
      class: "scenario-corridor",
      "pointer-events": "none",
    });
    svg.appendChild(corridor);
  }
  for (let index = 0; index < state.coolingScenario.additionalCentres; index += 1) {
    const angle = -Math.PI / 2 + (index / Math.max(1, state.coolingScenario.additionalCentres)) * Math.PI * 2;
    const radius = 22 + (index % 2) * 9;
    const marker = svgEl("g", {
      class: "scenario-cooling-point",
      transform: `translate(${(x + Math.cos(angle) * radius).toFixed(1)} ${(y + Math.sin(angle) * radius).toFixed(1)})`,
      "pointer-events": "none",
    });
    marker.appendChild(svgEl("circle", { r: 8 }));
    const plus = svgEl("text", { x: 0, y: 3.2, "text-anchor": "middle" });
    plus.textContent = "+";
    marker.appendChild(plus);
    svg.appendChild(marker);
  }
}

function renderMap(stationRows) {
  const observedRows = currentNetwork();
  const forecastPeakAreas = computeForecastPeakAreas();
  const forecastPeakRows = networkAtTime(state.data?.peakTime || forecastWindowPoints().at(-1)?.time);
  const coolingRows = simulatedCoolingRows();
  getMapViews().forEach((view) => {
    const useCurrent = view.key === "home" && observedRows.length > 0;
    const rows = view.key === "forecast" ? forecastPeakRows : useCurrent ? observedRows : stationRows;
    const areas = view.key === "forecast"
      ? forecastPeakAreas
      : view.key === "cooling"
        ? coolingRows
        : useCurrent ? computeAreas(rows) : state.areaRows;
    renderMapInstance(rows, view, areas, useCurrent);
  });
}

function renderMapInstance(stationRows, view, areaRows, useCurrent) {
  const svg = view.svg;
  const mapLayer = view.key === "home" || view.key === "forecast" ? "heat" : view.key === "cooling" ? "risk" : state.layer;
  svg.replaceChildren();
  svg.appendChild(svgEl("rect", { width: MAP.width, height: MAP.height, class: "map-water" }));
  const { patternId, shadowId } = appendMapDefinitions(svg, view.key);

  for (const row of areaRows.slice().reverse()) {
    const areaSelected = view.key !== "home" && row.name === state.area;
    const areaInteractive = view.key !== "home";
    const region = svgEl("path", {
      d: geometryPath(row.feature.geometry),
      class: `planning-area ${areaSelected ? "selected" : ""}`,
      fill: areaFill(view.key === "cooling" ? { ...row, score: row.simulatedScore } : row, mapLayer),
      "fill-rule": "evenodd",
      tabindex: areaInteractive ? 0 : -1,
      role: areaInteractive ? "button" : "img",
      "aria-label": view.key === "forecast"
        ? `${row.name}, local environment heat index estimate ${row.heat.toFixed(1)} degrees Celsius within ${state.localRadius} metres`
        : areaInteractive
          ? `${row.name}, estimated heat ${row.heat.toFixed(1)} degrees Celsius, community risk ${row.score.toFixed(0)}`
        : `${row.name}, live heat index estimate ${row.heat.toFixed(1)} degrees Celsius`,
    });
    const activate = () => view.key === "forecast" ? selectForecastArea(row.name) : selectArea(row.name);
    if (areaInteractive) {
      region.addEventListener("click", activate);
      region.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); }
      });
    }
    region.addEventListener("pointermove", (event) => {
      const detail = view.key === "home"
        ? `<strong>${row.name}</strong><span>${row.heat.toFixed(1)}°C live heat index estimate</span>`
        : view.key === "forecast"
          ? `<strong>${row.name}</strong><span>${row.heat.toFixed(1)}°C local estimate</span><span>${row.risk.label} heat risk · ${state.localRadius} m surroundings</span><span>Click for environment details</span>`
          : view.key === "cooling"
            ? `<strong>${row.name}</strong><span>${row.simulatedRisk.label} · simulated score ${row.simulatedScore.toFixed(0)}</span><span>Green baseline ${Number(row.greenBaselineScore || 0).toFixed(0)}/100 · ${Number((row.cyclingKm || 0) + (row.parkConnectorKm || 0)).toFixed(1)} km network</span><span>${row.simulated ? `${row.score.toFixed(0)} before intervention · click to adjust` : "Click to test an intervention"}</span>`
          : `<strong>${row.name}</strong><span>${row.risk.label} · score ${row.score.toFixed(0)}</span><span>${row.heat.toFixed(1)}°C ${useCurrent ? "current" : "forecast"} estimate</span><span>${fmtNumber.format(row.seniors)} residents aged 65+</span>`;
      showMapTooltip(event, detail, view);
    });
    region.addEventListener("pointerleave", () => hideMapTooltip(view));
    svg.appendChild(region);
  }

  appendTerrain(svg, patternId, areaRows);
  appendSubzoneDetail(svg, view, stationRows, mapLayer);
  appendGreenInfrastructure(svg, view);
  appendBlockDetail(svg, view, areaRows);

  if (view.key === "cooling") {
    const selected = areaRows.find((row) => row.name === state.area);
    if (selected) appendCoolingScenario(svg, selected);
  }

  if ((state.showCooling && view.key === "community") || view.key === "cooling") {
    for (const centre of state.geo.coolingCentres) {
      const [x, y] = project([centre.lon, centre.lat]);
      const marker = svgEl("circle", { cx: x, cy: y, r: 3.2, class: "cooling-point" });
      marker.addEventListener("pointermove", (event) => showMapTooltip(event,
        `<strong>${centre.name}</strong><span>Eldercare access proxy</span><span>${centre.area || "Planning area unavailable"}</span>`, view
      ));
      marker.addEventListener("pointerleave", () => hideMapTooltip(view));
      svg.appendChild(marker);
    }
  }

  for (const row of areaRows) {
    const labelTier = OVERVIEW_LABEL_AREAS.has(row.name)
      ? "map-label-overview"
      : LABEL_AREAS.has(row.name)
        ? "map-label-district"
        : "map-label-detail";
    const [x, y] = project(row.centroid);
    const label = svgEl("text", {
      x, y, class: `place-label ${labelTier} ${view.key !== "home" && row.name === state.area ? "selected" : ""}`,
      "text-anchor": "middle", "pointer-events": "none",
    });
    label.textContent = row.name;
    svg.appendChild(label);
  }

  const geoLabels = [
    { text: "JOHOR STRAIT", point: [103.82, 1.468], cls: "water-label" },
    { text: "SINGAPORE STRAIT", point: [103.80, 1.226], cls: "water-label" },
    { text: "Central Catchment", point: [103.80, 1.414], cls: "terrain-label" },
    { text: "Bukit Timah · 163 m", point: [103.776, 1.356], cls: "terrain-label" },
    { text: "Marina Bay", point: [103.866, 1.274], cls: "water-label small" },
  ];
  for (const item of geoLabels) {
    const [x, y] = project(item.point);
    const label = svgEl("text", { x, y, class: item.cls, "text-anchor": "middle", "pointer-events": "none" });
    label.textContent = item.text;
    svg.appendChild(label);
  }

  for (const row of view.key === "forecast" || view.key === "cooling" ? [] : stationRows) {
    const meta = STATION_META[row.station];
    if (!meta) continue;
    const [x, y] = project([meta.lon, meta.lat]);
    const marker = svgEl("g", {
      class: `station-marker ${row.risk.key} ${row.station === state.station ? "selected" : ""}`,
      transform: `translate(${x.toFixed(2)} ${y.toFixed(2)})`,
      role: "button", tabindex: 0,
      "aria-label": `${row.station}, ${meta.name}, ${row.predicted.toFixed(1)} degrees Celsius`,
    });
    marker.appendChild(svgEl("circle", { class: "marker-halo", r: 24 }));
    marker.appendChild(svgEl("circle", { class: "marker-dot", r: 15, style: `filter:url(#${shadowId})` }));
    const value = svgEl("text", { class: "marker-value", x: 0, y: 3.5, "text-anchor": "middle" });
    value.textContent = row.predicted.toFixed(0);
    marker.appendChild(value);
    const stationName = svgEl("text", {
      class: "station-name-label",
      x: 20,
      y: -12,
      "text-anchor": "start",
      "pointer-events": "none",
    });
    stationName.textContent = meta.name;
    marker.appendChild(stationName);
    const activate = () => selectStation(row.station, useCurrent ? state.data.peakTime : row.time);
    marker.addEventListener("click", activate);
    marker.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); }
    });
    marker.addEventListener("pointermove", (event) => {
      const detail = useCurrent
        ? `<span>${row.predicted.toFixed(1)}°C observed heat index</span><span>${row.temperature.toFixed(1)}°C air · ${row.humidity.toFixed(0)}% humidity</span>`
        : `<span>${row.predicted.toFixed(1)}°C predicted heat index</span>`;
      showMapTooltip(event,
        `<strong>${row.station} · ${meta.name}</strong>${detail}<span>${row.risk.label} temperature band</span>`, view
      );
    });
    marker.addEventListener("pointerleave", () => hideMapTooltip(view));
    svg.appendChild(marker);
  }
  renderLegend(view, useCurrent, mapLayer);
  applyMapViewport(view);
}

function showMapTooltip(event, html, view) {
  if (!view?.tooltip) return;
  const rect = view.svg.getBoundingClientRect();
  view.tooltip.hidden = false;
  view.tooltip.style.left = `${Math.min(rect.width - 205, Math.max(8, event.clientX - rect.left + 12))}px`;
  view.tooltip.style.top = `${Math.max(8, event.clientY - rect.top - 92)}px`;
  view.tooltip.innerHTML = html;
}

function hideMapTooltip(view) {
  if (view?.tooltip) view.tooltip.hidden = true;
}

function renderLegend(view, useCurrent = false, layer = state.layer) {
  if (!view?.legend) return;
  const details = layer === "heat"
    ? { title: view.key === "forecast" ? "Local environment heat estimate" : useCurrent ? "Live observed heat index" : "24h heat index estimate", min: "31°C", max: "43°C", cls: "heat" }
    : layer === "elderly"
      ? { title: "Residents aged 65+ per km²", min: "Low", max: "High", cls: "elderly" }
      : { title: view.key === "cooling" ? "Simulated Heat Risk Score" : "Community Heat Risk Score", min: "Low", max: "High", cls: "risk" };
  const coolingKey = view.key === "community" && state.showCooling ? `<em><u></u> Eldercare point</em>` : "";
  const scenarioKey = view.key === "cooling" ? `<em><u class="scenario-key"></u> Proposed intervention</em>` : "";
  const greenKeys = view.key === "cooling" && state.greenPaths
    ? `${state.greenLayers.parks ? `<em><u class="park-key"></u> Existing parks</em>` : ""}${state.greenLayers.cycling || state.greenLayers.parkConnectors ? `<em><u class="cycling-key"></u> Cycling paths / park connectors</em>` : ""}`
    : "";
  view.legend.innerHTML = `<strong>${details.title}</strong><i class="legend-ramp ${details.cls}"></i><span><b>${details.min}</b><b>${details.max}</b></span>${coolingKey}${greenKeys}${scenarioKey}`;
}

function updateAreaDetails() {
  const row = state.areaRows.find((item) => item.name === state.area) || state.areaRows[0];
  if (!row) return;
  els.areaTitle.textContent = `${row.name} · ${row.region}`;
  els.areaRisk.textContent = row.score.toFixed(0);
  els.areaRiskChip.textContent = row.risk.label;
  els.areaRiskChip.className = `risk-chip ${row.risk.key}`;
  els.areaHeat.textContent = `${row.heat.toFixed(1)}°C`;
  els.areaSeniors.textContent = row.seniors ? fmtNumber.format(row.seniors) : "No resident count";
  els.areaShare.textContent = row.residents ? `${row.seniorShare.toFixed(1)}%` : "—";
  els.areaCooling.textContent = row.seniors ? `${row.coolingCentres} mapped · ${row.accessPer10k.toFixed(1)} / 10k seniors` : `${row.coolingCentres} mapped`;
  if (els.areaHdbBlocks) els.areaHdbBlocks.textContent = `${fmtNumber.format(row.hdbBlocks || 0)} mapped`;
  els.riskHeatBar.style.width = `${row.heatScore.toFixed(1)}%`;
  els.riskElderlyBar.style.width = `${row.vulnerability.toFixed(1)}%`;
  els.riskAccessBar.style.width = `${row.accessGap.toFixed(1)}%`;
  els.riskHeatValue.textContent = `${row.heatContribution.toFixed(0)}%`;
  els.riskElderlyValue.textContent = `${row.elderlyContribution.toFixed(0)}%`;
  els.riskAccessValue.textContent = `${row.accessContribution.toFixed(0)}%`;
  els.actionTitle.textContent = row.risk.title;
  els.actionCopy.textContent = `${row.risk.copy} ${row.name} has ${fmtNumber.format(row.seniors)} official 2026 residents aged 65+, ${fmtNumber.format(row.hdbBlocks || 0)} mapped HDB buildings and ${row.coolingCentres} eldercare point${row.coolingCentres === 1 ? "" : "s"}. Prototype guidance only.`;
  els.actionCard.className = `panel alert-tier-panel ${row.risk.key}`;
  document.querySelectorAll("[data-risk-tier]").forEach((tier) => tier.classList.toggle("active", tier.dataset.riskTier === row.risk.key));
}

function renderAreaList() {
  els.areaList.replaceChildren();
  state.areaRows.filter((row) => row.seniors >= 1000).slice(0, 8).forEach((row, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `area-row ${row.name === state.area ? "active" : ""}`;
    button.setAttribute("role", "listitem");
    button.innerHTML = `
      <span class="area-rank">${String(index + 1).padStart(2, "0")}</span>
      <strong>${row.name}<small>${row.coolingCentres} eldercare points</small></strong>
      <span>${fmtCompact.format(row.seniors)}</span>
      <span>${row.heat.toFixed(1)}°C</span>
      <b class="score-pill ${row.risk.key}">${row.score.toFixed(0)}</b>`;
    button.addEventListener("click", () => selectArea(row.name, true));
    els.areaList.appendChild(button);
  });
}

function updateCoolingSimulation() {
  if (!els.coolingAreaSelect || !state.areaRows.length || !state.area) return;
  const populatedRows = state.areaRows.filter((row) => row.seniors >= 1000);
  if (els.coolingAreaSelect.options.length !== populatedRows.length) {
    els.coolingAreaSelect.replaceChildren();
    populatedRows.slice().sort((a, b) => a.name.localeCompare(b.name)).forEach((row) => {
      const option = document.createElement("option");
      option.value = row.name;
      option.textContent = row.name;
      els.coolingAreaSelect.appendChild(option);
    });
  }
  const base = state.areaRows.find((row) => row.name === state.area) || populatedRows[0];
  if (!base) return;
  if (!populatedRows.some((row) => row.name === base.name)) {
    state.area = populatedRows[0].name;
    return updateCoolingSimulation();
  }
  const simulated = simulatedCoolingRows().find((row) => row.name === state.area);
  if (!simulated) return;
  els.coolingAreaSelect.value = base.name;
  els.simulationAreaName.textContent = base.name;
  els.coolingPointsOutput.textContent = String(state.coolingScenario.additionalCentres);
  els.greenShadeOutput.textContent = `${state.coolingScenario.greenShadePct}%`;
  els.simulationBeforeScore.textContent = base.score.toFixed(0);
  els.simulationAfterScore.textContent = simulated.simulatedScore.toFixed(0);
  els.simulationBeforeTier.textContent = `${base.risk.label} risk`;
  els.simulationAfterTier.textContent = `${simulated.simulatedRisk.label} risk`;
  els.simulationRiskChip.textContent = `${simulated.simulatedRisk.label} after`;
  els.simulationRiskChip.className = `risk-chip ${simulated.simulatedRisk.key}`;
  const scoreChange = simulated.simulatedScore - base.score;
  els.simulationScoreChange.textContent = `${scoreChange > 0 ? "+" : ""}${scoreChange.toFixed(1)} pts`;
  els.simulationHeatChange.textContent = `−${simulated.heatReduction.toFixed(2)}°C`;
  els.simulationCentreChange.textContent = `${base.coolingCentres} → ${simulated.simulatedCentreCount}`;
  if (els.simulationGreenScore) els.simulationGreenScore.textContent = `${Number(base.greenBaselineScore || 0).toFixed(0)} / 100`;
  if (els.simulationParkCover) els.simulationParkCover.textContent = `${Number(base.parkCoveragePct || 0).toFixed(1)}%`;
  if (els.simulationGreenNetwork) els.simulationGreenNetwork.textContent = `${Number((base.cyclingKm || 0) + (base.parkConnectorKm || 0)).toFixed(1)} km`;
  els.simulationGuidance.textContent = simulated.simulatedRisk.key === "high"
    ? `The area remains High risk. Its existing green-infrastructure score is ${Number(base.greenBaselineScore || 0).toFixed(0)}/100; consider a stronger package or targeted outreach alongside infrastructure changes.`
    : simulated.simulatedRisk.key === "medium"
      ? "The scenario moves or keeps the area in Medium risk. Prepare local support and monitor the next forecast cycle."
      : "The scenario places the area in Low risk. Maintain routine monitoring and verify local conditions before standing down support."
  renderCoolingPriorityList();
}

function renderCoolingPriorityList() {
  if (!els.coolingPriorityList) return;
  els.coolingPriorityList.replaceChildren();
  state.areaRows.filter((row) => row.seniors >= 1000).slice(0, 6).forEach((row, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `cooling-priority-row ${row.name === state.area ? "active" : ""}`;
    button.setAttribute("role", "listitem");
    button.innerHTML = `<span>${String(index + 1).padStart(2, "0")}</span><strong>${row.name}</strong><small>${row.heat.toFixed(1)}°C · ${fmtCompact.format(row.seniors)} seniors · green ${Number(row.greenBaselineScore || 0).toFixed(0)}</small><b class="score-pill ${row.risk.key}">${row.score.toFixed(0)}</b>`;
    button.addEventListener("click", () => selectArea(row.name));
    els.coolingPriorityList.appendChild(button);
  });
}

function svgEl(name, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function renderChart(series, activeIndex, options = {}) {
  const showActual = options.showActual !== false;
  const allowSelection = options.allowSelection !== false;
  const svg = els.chart;
  svg.replaceChildren();
  const width = 900;
  const height = 340;
  const margin = { top: 22, right: 18, bottom: 44, left: 50 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const values = series.flatMap((d) => showActual ? [d.predicted, d.actual] : [d.predicted]).filter(Number.isFinite);
  const minY = Math.min(29, Math.floor(Math.min(...values) - 1));
  const maxY = Math.max(42, Math.ceil(Math.max(...values) + 1));
  const x = (index) => margin.left + (index / Math.max(1, series.length - 1)) * innerW;
  const y = (value) => margin.top + ((maxY - value) / (maxY - minY)) * innerH;

  if (maxY > 37) {
    svg.appendChild(svgEl("rect", {
      x: margin.left, y: margin.top, width: innerW, height: Math.max(0, y(37) - margin.top),
      fill: "rgba(201,31,55,.055)",
    }));
    const focusLabel = svgEl("text", { x: margin.left + 8, y: margin.top + 15, class: "chart-zone-label" });
    focusLabel.textContent = "HIGH-HEAT FOCUS";
    svg.appendChild(focusLabel);
  }

  for (let value = Math.ceil(minY / 2) * 2; value <= maxY; value += 2) {
    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value), class: "chart-grid" }));
    const label = svgEl("text", { x: margin.left - 10, y: y(value) + 4, class: "chart-axis", "text-anchor": "end" });
    label.textContent = value;
    svg.appendChild(label);
  }

  series.map((d, i) => ({ d, i })).filter(({ d, i }) => options.compact24
    ? i === 0 || i === series.length - 1 || i % 6 === 0
    : i === 0 || new Date(d.time).getHours() === 0
  ).forEach(({ d, i }) => {
    const label = svgEl("text", { x: x(i), y: height - 12, class: "chart-axis", "text-anchor": "middle" });
    label.textContent = new Intl.DateTimeFormat(currentLocale(), options.compact24
      ? { timeZone: "Asia/Singapore", hour: "2-digit", minute: "2-digit", hour12: false }
      : { timeZone: "Asia/Singapore", day: "numeric", month: "short" }
    ).format(new Date(d.time));
    svg.appendChild(label);
  });

  const linePath = (key) => {
    let penDown = false;
    return series.map((d, i) => {
      if (!Number.isFinite(d[key])) { penDown = false; return ""; }
      const command = penDown ? "L" : "M";
      penDown = true;
      return `${command}${x(i).toFixed(2)},${y(d[key]).toFixed(2)}`;
    }).filter(Boolean).join(" ");
  };
  const actualPath = showActual ? linePath("actual") : "";
  if (actualPath) svg.appendChild(svgEl("path", { d: actualPath, class: "chart-line actual-line" }));
  svg.appendChild(svgEl("path", { d: linePath("predicted"), class: "chart-line predicted-line" }));
  const activeX = x(activeIndex);
  svg.appendChild(svgEl("line", { x1: activeX, x2: activeX, y1: margin.top, y2: height - margin.bottom, class: "chart-active-line" }));
  svg.appendChild(svgEl("circle", { cx: activeX, cy: y(series[activeIndex].predicted), r: 6, class: "chart-active-dot" }));

  const capture = svgEl("rect", { x: margin.left, y: margin.top, width: innerW, height: innerH, fill: "transparent", tabindex: 0 });
  const indexFromEvent = (event) => {
    const rect = svg.getBoundingClientRect();
    const position = ((event.clientX - rect.left) / rect.width) * width;
    return clamp(Math.round(((position - margin.left) / innerW) * (series.length - 1)), 0, series.length - 1);
  };
  capture.addEventListener("pointermove", (event) => {
    const rect = svg.getBoundingClientRect();
    const point = series[indexFromEvent(event)];
    els.chartTooltip.hidden = false;
    els.chartTooltip.style.left = `${Math.min(rect.width - 170, Math.max(8, event.clientX - rect.left + 10))}px`;
    els.chartTooltip.style.top = `${Math.max(8, event.clientY - rect.top - 72)}px`;
    const observed = showActual
      ? Number.isFinite(point.actual) ? `<br>Observed ${point.actual.toFixed(1)}°C` : "<br>Observation pending"
      : "";
    const interval = Number.isFinite(point.lower) && Number.isFinite(point.upper)
      ? `<br>80% interval ${point.lower.toFixed(1)}–${point.upper.toFixed(1)}°C` : "";
    els.chartTooltip.innerHTML = `<strong>${fmtTime(point.time)}</strong>Predicted ${point.predicted.toFixed(1)}°C${observed}${interval}`;
  });
  capture.addEventListener("pointerleave", () => { els.chartTooltip.hidden = true; });
  if (allowSelection) {
    capture.addEventListener("click", (event) => {
      state.index = indexFromEvent(event);
      syncSlider();
      updateDashboard();
    });
  }
  svg.appendChild(capture);
}

function syncSlider() {
  const station = getStation();
  els.timeSlider.max = String(station.series.length - 1);
  els.timeSlider.value = String(state.index);
}

async function init() {
  Object.assign(els, {
    stationSelect: byId("station-select"),
    timeSlider: byId("time-slider"),
    headerRecall: byId("header-recall"),
    stationTitle: byId("station-title"),
    riskChip: byId("risk-chip"),
    predicted: byId("predicted-hi"),
    actual: byId("actual-hi"),
    actualLabel: byId("actual-label"),
    actualNote: byId("actual-note"),
    rmse: byId("station-rmse"),
    rmseLabel: byId("rmse-label"),
    rmseNote: byId("rmse-note"),
    delta: byId("forecast-delta"),
    chart: byId("forecast-chart"),
    chartTooltip: byId("chart-tooltip"),
    actualLegend: byId("actual-legend"),
    networkMean: byId("network-mean"),
    priorityAreaCount: byId("priority-area-count"),
    seniorsExposed: byId("seniors-exposed"),
    map: byId("sg-map"),
    mapTooltip: byId("map-tooltip"),
    mapLegend: byId("map-legend"),
    mapStationCount: byId("map-station-count"),
    mapStage: document.querySelector('[data-map-key="community"]'),
    mapScale: document.querySelector('[data-map-key="community"] .map-scale'),
    mapScaleLabel: document.querySelector('[data-map-key="community"] .map-scale-label'),
    mapScaleLine: document.querySelector('[data-map-key="community"] .map-scale-line'),
    spatialDetailStatus: byId("spatial-detail-status"),
    coolingToggle: byId("cooling-toggle"),
    homeMap: byId("home-sg-map"),
    homeMapTooltip: byId("home-map-tooltip"),
    homeMapLegend: byId("home-map-legend"),
    homeMapStationCount: byId("home-map-station-count"),
    homeMapStage: document.querySelector('[data-map-key="home"]'),
    homeMapScale: document.querySelector('[data-map-key="home"] .map-scale'),
    homeMapScaleLabel: document.querySelector('[data-map-key="home"] .map-scale-label'),
    homeMapScaleLine: document.querySelector('[data-map-key="home"] .map-scale-line'),
    homeCoolingToggle: byId("home-cooling-toggle"),
    homeReadingLabel: byId("home-reading-label"),
    homeReadingTime: byId("home-reading-time"),
    homeNextUpdateTime: byId("home-next-update-time"),
    forecastMap: byId("forecast-sg-map"),
    forecastMapTooltip: byId("forecast-map-tooltip"),
    forecastMapLegend: byId("forecast-map-legend"),
    forecastMapStage: document.querySelector('[data-map-key="forecast"]'),
    forecastMapScale: document.querySelector('[data-map-key="forecast"] .map-scale'),
    forecastMapScaleLabel: document.querySelector('[data-map-key="forecast"] .map-scale-label'),
    forecastMapScaleLine: document.querySelector('[data-map-key="forecast"] .map-scale-line'),
    coolingMap: byId("cooling-sg-map"),
    coolingMapTooltip: byId("cooling-map-tooltip"),
    coolingMapLegend: byId("cooling-map-legend"),
    coolingMapStage: document.querySelector('[data-map-key="cooling"]'),
    coolingMapScale: document.querySelector('[data-map-key="cooling"] .map-scale'),
    coolingMapScaleLabel: document.querySelector('[data-map-key="cooling"] .map-scale-label'),
    coolingMapScaleLine: document.querySelector('[data-map-key="cooling"] .map-scale-line'),
    forecastAreaTitle: byId("forecast-area-title"),
    forecastAreaRiskChip: byId("forecast-area-risk-chip"),
    forecastAreaPeak: byId("forecast-area-peak"),
    forecastAreaPeakTime: byId("forecast-area-peak-time"),
    forecastAreaGuidance: byId("forecast-area-guidance"),
    forecastRiskAlert: byId("forecast-risk-alert"),
    localRadiusSelect: byId("local-radius-select"),
    localParkCover: byId("local-park-cover"),
    localBuildingCover: byId("local-building-cover"),
    localBuildingDensity: byId("local-building-density"),
    localCoastDistance: byId("local-coast-distance"),
    areaTitle: byId("area-title"),
    areaRiskChip: byId("area-risk-chip"),
    areaRisk: byId("area-risk"),
    areaHeat: byId("area-heat"),
    areaSeniors: byId("area-seniors"),
    areaShare: byId("area-share"),
    areaCooling: byId("area-cooling"),
    areaHdbBlocks: byId("area-hdb-blocks"),
    riskHeatBar: byId("risk-heat-bar"),
    riskElderlyBar: byId("risk-elderly-bar"),
    riskAccessBar: byId("risk-access-bar"),
    riskHeatValue: byId("risk-heat-value"),
    riskElderlyValue: byId("risk-elderly-value"),
    riskAccessValue: byId("risk-access-value"),
    actionCard: byId("action-card"),
    actionTitle: byId("action-title"),
    actionCopy: byId("action-copy"),
    mappedSeniors: byId("mapped-seniors"),
    areaList: byId("area-list"),
    forecastContextTime: byId("forecast-context-time"),
    communityContextTime: byId("community-context-time"),
    homeLiveCount: byId("home-live-count"),
    homeLiveMean: byId("home-live-mean"),
    homeForecastPeak: byId("home-forecast-peak"),
    homeMapNetworkMean: byId("home-map-network-mean"),
    homeMapPriorityCount: byId("home-map-priority-count"),
    homeMapSeniorsExposed: byId("home-map-seniors-exposed"),
    coolingAreaSelect: byId("cooling-area-select"),
    coolingPointsSlider: byId("cooling-points-slider"),
    greenShadeSlider: byId("green-shade-slider"),
    coolingPointsOutput: byId("cooling-points-output"),
    greenShadeOutput: byId("green-shade-output"),
    simulationAreaName: byId("simulation-area-name"),
    simulationRiskChip: byId("simulation-risk-chip"),
    simulationBeforeScore: byId("simulation-before-score"),
    simulationAfterScore: byId("simulation-after-score"),
    simulationBeforeTier: byId("simulation-before-tier"),
    simulationAfterTier: byId("simulation-after-tier"),
    simulationScoreChange: byId("simulation-score-change"),
    simulationHeatChange: byId("simulation-heat-change"),
    simulationCentreChange: byId("simulation-centre-change"),
    simulationGreenScore: byId("simulation-green-score"),
    simulationParkCover: byId("simulation-park-cover"),
    simulationGreenNetwork: byId("simulation-green-network"),
    simulationGuidance: byId("simulation-guidance"),
    coolingPriorityList: byId("cooling-priority-list"),
  });

  initMapInteractions();
  showView();
  window.addEventListener("hashchange", () => {
    showView();
    window.scrollTo({ top: 0, behavior: "auto" });
  });

  try {
    const packagedGeo = window.__HEATGUARD_GEO__ || await fetch("sg-planning-areas.json").then((response) => response.json());
    state.geo = packagedGeo;
    if (window.HeatGuardSpatial && window.location.protocol !== "file:") {
      try {
        state.spatial = await window.HeatGuardSpatial.loadBase();
        state.geo = window.HeatGuardSpatial.enrichPlanningAreas(packagedGeo, state.spatial);
      } catch (error) {
        console.warn("Detailed 2026 spatial data could not be loaded; using packaged planning-area data.", error);
      }
    }
    state.localHeat = await loadLocalHeatResults();
    applyForecastData(await loadLatestData(), false);
    els.mappedSeniors.textContent = fmtNumber.format(state.geo.features.reduce((sum, feature) => sum + feature.properties.seniors, 0));
    ensureBlocksForArea(state.area);
    if (routeNameFromHash() === "cooling") ensureGreenData();

    els.stationSelect.addEventListener("change", (event) => {
      selectStation(event.target.value, getStation().series[state.index]?.time);
    });
    els.timeSlider.addEventListener("input", (event) => {
      state.index = Number(event.target.value);
      updateDashboard();
    });
    document.querySelectorAll("[data-module-jump]").forEach((link) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        byId(link.dataset.moduleJump)?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
    document.querySelectorAll("[data-layer]").forEach((button) => {
      button.addEventListener("click", () => {
        state.layer = button.dataset.layer;
        document.querySelectorAll("[data-layer]").forEach((item) => item.classList.toggle("active", item.dataset.layer === state.layer));
        renderMap(networkAtTime(getStation().series[state.index].time));
      });
    });
    document.querySelectorAll("[data-cooling-toggle]").forEach((toggle) => {
      toggle.addEventListener("change", () => {
        state.showCooling = toggle.checked;
        document.querySelectorAll("[data-cooling-toggle]").forEach((item) => { item.checked = state.showCooling; });
        renderMap(networkAtTime(getStation().series[state.index].time));
      });
    });
    els.coolingAreaSelect.addEventListener("change", (event) => selectArea(event.target.value));
    els.coolingPointsSlider.addEventListener("input", (event) => {
      state.coolingScenario.additionalCentres = Number(event.target.value);
      updateCoolingSimulation();
      renderMap(networkAtTime(getStation().series[state.index].time));
    });
    els.greenShadeSlider.addEventListener("input", (event) => {
      state.coolingScenario.greenShadePct = Number(event.target.value);
      updateCoolingSimulation();
      renderMap(networkAtTime(getStation().series[state.index].time));
    });
    els.localRadiusSelect?.addEventListener("change", (event) => {
      state.localRadius = Number(event.target.value);
      updateForecastAreaPanel();
      renderMap(networkAtTime(getStation().series[state.index].time));
    });
    document.querySelectorAll("[data-green-layer]").forEach((input) => {
      input.addEventListener("change", () => {
        state.greenLayers[input.dataset.greenLayer] = input.checked;
        renderMap(networkAtTime(getStation().series[state.index].time));
      });
    });
    if (window.location.protocol !== "file:") {
      window.setInterval(refreshForecastData, LIVE_POLL_MS);
      window.setInterval(updateHomeDataTiming, 60*1000);
    }
    byId("publication-refresh")?.addEventListener("click", async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      button.textContent = "Checking…";
      const success = await refreshForecastData();
      button.textContent = success ? "Latest published data loaded" : "Could not update · retry";
      window.setTimeout(() => {
        button.disabled = false;
        button.textContent = "Check for latest data";
      }, 2500);
    });
  } catch (error) {
    console.error(error);
    const activeView = document.querySelector("[data-view]:not([hidden])") || document.querySelector("[data-view]");
    activeView.innerHTML = `<div class="panel error-panel"><p class="eyebrow">Data unavailable</p><h1>The decision dashboard could not load.</h1><p>Please refresh the page or try again shortly.</p></div>`;
  }
}

async function loadLatestData() {
  if (window.location.protocol === "file:" && window.__HEATGUARD_DATA__) {
    return window.__HEATGUARD_DATA__;
  }
  if (window.HEATGUARD_STATIC_PUBLIC) {
    const response = await fetch(`${LIVE_DATA_URL}?t=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Published V4 forecast is unavailable");
    return response.json();
  }
  if (window.location.protocol !== "file:") {
    try {
      const response = await fetch(`${LIVE_DATA_URL}?t=${Date.now()}`, { cache: "no-store" });
      if (response.ok) return await response.json();
    } catch (error) {
      console.info("Live forecast endpoint is not ready; using packaged replay data.", error);
    }
  }
  if (window.__HEATGUARD_DATA__) return window.__HEATGUARD_DATA__;
  const response = await fetch("heatguard-data.json", { cache: "no-store" });
  if (!response.ok) throw new Error("Dashboard data unavailable");
  return response.json();
}

async function loadLocalHeatResults() {
  if (window.location.protocol === "file:") return null;
  try {
    const response = await fetch("data/local-heat-results.json", { cache: "no-store" });
    if (!response.ok) throw new Error("Local heat results are unavailable");
    return await response.json();
  } catch (error) {
    console.warn("Local environment estimates could not be loaded; retaining the forecast fallback.", error);
    return null;
  }
}

function applyForecastData(data, preserveSelection = true) {
  const previousStation = preserveSelection ? state.station : null;
  const previousTime = preserveSelection && state.data ? getStation()?.series?.[state.index]?.time : null;
  state.data = data;
  state.nextRefreshAt = Date.now() + LIVE_POLL_MS;
  const stationIds = new Set(data.stations.map((station) => station.id));
  state.station = previousStation && stationIds.has(previousStation) ? previousStation : data.defaultStation;
  els.stationSelect.replaceChildren();
  data.stations.forEach((station) => {
    const option = document.createElement("option");
    option.value = station.id;
    option.textContent = `${station.id} · ${STATION_META[station.id]?.name || "Weather station"}`;
    els.stationSelect.appendChild(option);
  });
  els.stationSelect.value = state.station;
  const station = getStation();
  const retainedIndex = previousTime ? station.series.findIndex((point) => point.time === previousTime) : -1;
  const peakIndex = station.series.findIndex((point) => point.time === data.peakTime);
  state.index = retainedIndex >= 0 ? retainedIndex : peakIndex >= 0 ? peakIndex : 0;
  syncSlider();
  updateDashboard();
}

async function refreshForecastData() {
  try {
    const data = await loadLatestData();
    const changed = data.generatedAt && data.generatedAt !== state.data?.generatedAt;
    if (changed || data.mode !== state.data?.mode) applyForecastData(data, true);
    updateHomeDataTiming();
    return true;
  } catch (error) {
    console.warn("The latest forecast could not be loaded; keeping the current dashboard.", error);
    return false;
  }
}

init();

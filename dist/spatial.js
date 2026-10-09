(function () {
  const ROOT = "data/spatial";
  const blockCache = new Map();
  const plannerBuildingCache = new Map();
  let basePromise = null;
  let greenPromise = null;
  let plannerBuildingIndexPromise = null;

  async function fetchJson(path) {
    const response = await fetch(path, { cache: "force-cache" });
    if (!response.ok) throw new Error(`Spatial asset unavailable: ${path}`);
    return response.json();
  }

  function loadBase() {
    if (!basePromise) {
      basePromise = Promise.all([
        fetchJson(`${ROOT}/areas.geojson`),
        fetchJson(`${ROOT}/subzones.geojson`),
        fetchJson(`${ROOT}/blocks_index.json`),
        fetchJson(`${ROOT}/green-metrics.json`),
        fetchJson(`${ROOT}/metadata.json`),
      ]).then(([areas, subzones, blocksIndex, greenMetrics, metadata]) => ({
        areas,
        subzones,
        blocksIndex,
        greenMetrics,
        metadata,
      }));
    }
    return basePromise;
  }

  function loadGreen() {
    if (!greenPromise) {
      greenPromise = Promise.all([
        fetchJson(`${ROOT}/parks.geojson`),
        fetchJson(`${ROOT}/cycling.geojson`),
        fetchJson(`${ROOT}/park-connectors.geojson`),
      ]).then(([parks, cycling, parkConnectors]) => ({ parks, cycling, parkConnectors }));
    }
    return greenPromise;
  }

  async function loadBlocks(areaId) {
    if (!areaId) return null;
    if (blockCache.has(areaId)) return blockCache.get(areaId);
    const base = await loadBase();
    const item = base.blocksIndex[areaId];
    if (!item) return null;
    const request = fetchJson(`${ROOT}/${item.file}`).catch((error) => {
      blockCache.delete(areaId);
      throw error;
    });
    blockCache.set(areaId, request);
    return request;
  }

  async function loadPlannerBuildings(areaId) {
    if (!areaId) return null;
    if (plannerBuildingCache.has(areaId)) return plannerBuildingCache.get(areaId);
    if (!plannerBuildingIndexPromise) {
      plannerBuildingIndexPromise = fetchJson(`${ROOT}/planner-buildings-index.json`);
    }
    const index = await plannerBuildingIndexPromise;
    const item = index[areaId];
    if (!item) return null;
    const request = fetchJson(`${ROOT}/${item.file}`).then((collection) => ({ ...collection, metadata: item })).catch((error) => {
      plannerBuildingCache.delete(areaId);
      throw error;
    });
    plannerBuildingCache.set(areaId, request);
    return request;
  }

  function enrichPlanningAreas(original, spatial) {
    const oldByName = new Map(original.features.map((feature) => [feature.properties.name.toUpperCase(), feature.properties]));
    const greenById = new Map(spatial.greenMetrics.areas.map((row) => [row.id, row]));
    const features = spatial.areas.features.map((feature) => {
      const source = feature.properties;
      const old = oldByName.get(source.display_name.toUpperCase()) || {};
      const areaKm2 = source.area_km2 || old.areaKm2 || 0;
      const seniors = source.elderly_65plus_2026 || 0;
      const elderly75 = source.elderly_75plus_2026 || 0;
      const residents = source.population_2026 || 0;
      return {
        ...feature,
        properties: {
          ...old,
          name: source.display_name,
          code: source.id,
          region: old.region || source.region.replace(/ REGION$/i, ""),
          centroid: source.centroid,
          areaKm2,
          residents,
          seniors,
          seniorShare: source.elderly_65plus_share_pct || 0,
          seniorDensity: areaKm2 ? seniors / areaKm2 : 0,
          elderly75,
          elderly75Share: residents ? elderly75 / residents * 100 : 0,
          coolingCentres: source.cooling_centres || 0,
          hdbResidents: source.hdb_residents_2026 || 0,
          hdbSeniors: source.hdb_elderly_65plus_2026 || 0,
          hdbBlocks: source.hdb_building_count || 0,
          demographicsYear: 2026,
          ...greenById.get(source.id),
        },
      };
    });
    return { ...original, features };
  }

  function getAreaId(spatial, areaName) {
    return spatial.areas.features.find((feature) => feature.properties.display_name === areaName)?.properties.id || null;
  }

  function getSubzones(spatial, areaId) {
    return spatial.subzones.features.filter((feature) => feature.properties.planning_area_id === areaId);
  }

  window.HeatGuardSpatial = { loadBase, loadGreen, loadBlocks, loadPlannerBuildings, enrichPlanningAreas, getAreaId, getSubzones };
})();

#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const [planningPath, populationPath, eldercarePath, outputPath] = process.argv.slice(2);
if (!planningPath || !populationPath || !eldercarePath || !outputPath) {
  throw new Error("Usage: build-geodata.mjs <planning.geojson> <population.json> <eldercare.geojson> <output.json>");
}

const planning = JSON.parse(fs.readFileSync(planningPath, "utf8"));
const population = JSON.parse(fs.readFileSync(populationPath, "utf8"));
const eldercare = JSON.parse(fs.readFileSync(eldercarePath, "utf8"));

const number = (value) => value === "-" || value == null ? 0 : Number(String(value).replaceAll(",", "")) || 0;
const titleCase = (value) => value.toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());
const areaFields = ["Total_65_69", "Total_70_74", "Total_75_79", "Total_80_84", "Total_85_89", "Total_90andOver"];

const populationByArea = new Map();
for (const row of population.result.records) {
  if (!/\s*-\s*Total$/i.test(row.Number)) continue;
  const name = row.Number.replace(/\s*-\s*Total$/i, "").trim().toUpperCase();
  populationByArea.set(name, {
    residents: number(row.Total_Total),
    seniors: areaFields.reduce((sum, field) => sum + number(row[field]), 0),
  });
}

function perpendicularDistance(point, start, end) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  if (dx === 0 && dy === 0) return Math.hypot(point[0] - start[0], point[1] - start[1]);
  const t = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy));
}

function simplifyLine(points, tolerance) {
  if (points.length <= 4) return points;
  const open = points.slice(0, -1);
  const keep = new Uint8Array(open.length);
  keep[0] = 1;
  keep[open.length - 1] = 1;
  const stack = [[0, open.length - 1]];
  while (stack.length) {
    const [start, end] = stack.pop();
    let maxDistance = 0;
    let index = -1;
    for (let i = start + 1; i < end; i += 1) {
      const distance = perpendicularDistance(open[i], open[start], open[end]);
      if (distance > maxDistance) {
        maxDistance = distance;
        index = i;
      }
    }
    if (maxDistance > tolerance && index > start) {
      keep[index] = 1;
      stack.push([start, index], [index, end]);
    }
  }
  const result = open.filter((_, index) => keep[index]);
  if (result.length < 3) return points;
  result.push(result[0]);
  return result;
}

function simplifyGeometry(geometry) {
  const simplifyPolygon = (polygon) => polygon.map((ring) => simplifyLine(ring, 0.00022));
  return {
    type: geometry.type,
    coordinates: geometry.type === "Polygon"
      ? simplifyPolygon(geometry.coordinates)
      : geometry.coordinates.map(simplifyPolygon),
  };
}

function eachOuterRing(geometry) {
  return geometry.type === "Polygon" ? [geometry.coordinates[0]] : geometry.coordinates.map((polygon) => polygon[0]);
}

function polygonCentroid(ring) {
  let area = 0;
  let x = 0;
  let y = 0;
  for (let i = 0; i < ring.length - 1; i += 1) {
    const cross = ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
    area += cross;
    x += (ring[i][0] + ring[i + 1][0]) * cross;
    y += (ring[i][1] + ring[i + 1][1]) * cross;
  }
  const divisor = area * 3;
  if (!divisor) return { point: ring[0], weight: 0 };
  return { point: [x / divisor, y / divisor], weight: Math.abs(area / 2) };
}

function geometryCentroid(geometry) {
  const parts = eachOuterRing(geometry).map(polygonCentroid);
  const weight = parts.reduce((sum, part) => sum + part.weight, 0) || 1;
  return [
    parts.reduce((sum, part) => sum + part.point[0] * part.weight, 0) / weight,
    parts.reduce((sum, part) => sum + part.point[1] * part.weight, 0) / weight,
  ];
}

function pointInRing(point, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const intersects = ((yi > point[1]) !== (yj > point[1]))
      && (point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

function pointInGeometry(point, geometry) {
  const polygons = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
  return polygons.some((polygon) => pointInRing(point, polygon[0]) && !polygon.slice(1).some((hole) => pointInRing(point, hole)));
}

function decodeHtml(value) {
  return value
    .replace(/<[^>]*>/g, "")
    .replaceAll("&amp;", "&")
    .replaceAll("&#39;", "'")
    .replaceAll("&quot;", '"')
    .trim();
}

const centres = [];
const centreKeys = new Set();
for (const feature of eldercare.features) {
  const coordinates = feature.geometry?.coordinates?.slice(0, 2);
  if (!coordinates || coordinates.some((value) => !Number.isFinite(value))) continue;
  const match = feature.properties?.Description?.match(/<th>NAME<\/th>\s*<td>(.*?)<\/td>/i);
  const name = match ? decodeHtml(match[1]) : "Eldercare service";
  const key = `${coordinates.map((value) => value.toFixed(6)).join(",")}|${name}`;
  if (centreKeys.has(key)) continue;
  centreKeys.add(key);
  centres.push({ name, lon: coordinates[0], lat: coordinates[1] });
}

const features = planning.features.map((feature) => {
  const name = feature.properties.PLN_AREA_N;
  const demographic = populationByArea.get(name) || { residents: 0, seniors: 0 };
  const areaKm2 = Number(feature.properties["SHAPE.AREA"] || 0) / 1_000_000;
  const localCentres = centres.filter((centre) => pointInGeometry([centre.lon, centre.lat], feature.geometry));
  return {
    type: "Feature",
    properties: {
      name: titleCase(name),
      code: feature.properties.PLN_AREA_C,
      region: titleCase(feature.properties.REGION_N.replace(" REGION", "")),
      centroid: geometryCentroid(feature.geometry).map((value) => Number(value.toFixed(6))),
      areaKm2: Number(areaKm2.toFixed(2)),
      residents: demographic.residents,
      seniors: demographic.seniors,
      seniorShare: demographic.residents ? Number((demographic.seniors / demographic.residents * 100).toFixed(1)) : 0,
      seniorDensity: areaKm2 ? Number((demographic.seniors / areaKm2).toFixed(1)) : 0,
      coolingCentres: localCentres.length,
    },
    geometry: simplifyGeometry(feature.geometry),
  };
});

const centrePoints = centres.map((centre) => {
  const area = features.find((feature) => pointInGeometry([centre.lon, centre.lat], feature.geometry));
  return { ...centre, area: area?.properties.name || null };
});

const payload = {
  type: "FeatureCollection",
  source: {
    boundaries: "URA Master Plan 2019 Planning Area Boundary (No Sea)",
    population: "SingStat Census of Population 2020, planning area/subzone and age group",
    cooling: "MOH Eldercare Services directory (dataset last updated 2024)",
  },
  features,
  coolingCentres: centrePoints,
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(payload));
console.log(JSON.stringify({ outputPath, bytes: fs.statSync(outputPath).size, planningAreas: features.length, coolingCentres: centrePoints.length }, null, 2));

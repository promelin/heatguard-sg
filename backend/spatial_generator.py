"""Model-backed candidate generation for user-drawn Singapore sites."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import Polygon, box
from shapely.ops import transform as transform_geometry

from spatial_cvae.model import model_from_checkpoint


class SpatialGeneratorError(RuntimeError):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


class SpatialLayoutGenerator:
    ALLOWED_COUNTS = (10, 25, 50, 100, 250, 500, 1000)
    PREVIEW_LIMIT = 12
    PREVIEW_SIZE = 256
    TILE_SIZE = 256
    LATENT_TEMPERATURE = 1.8
    SIGNATURE_SIZE = 16

    def __init__(self, project_root: Path) -> None:
        model_root = project_root / "backend" / "model"
        self.checkpoint_path = Path(
            os.getenv("HEATGUARD_SPATIAL_CVAE", model_root / "spatial_cvae_inference.pt")
        )
        self.context_path = Path(
            os.getenv("HEATGUARD_SPATIAL_CONTEXT", model_root / "spatial_context.tif")
        )
        self.context_metadata_path = self.context_path.with_suffix(".json")
        self._model = None
        self._context_metadata = None
        self._device = None
        self._generation_lock = threading.Lock()
        self._to_svy21 = Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)

    def _read_context_metadata(self) -> dict:
        if self._context_metadata is None:
            if not self.context_metadata_path.exists():
                raise SpatialGeneratorError("Singapore spatial context is unavailable", 503)
            self._context_metadata = json.loads(
                self.context_metadata_path.read_text(encoding="utf-8")
            )
        return self._context_metadata

    def status(self) -> dict:
        metadata = self._read_context_metadata() if self.context_metadata_path.exists() else {}
        ready = self.checkpoint_path.exists() and self.context_path.exists() and bool(metadata)
        return {
            "ready": ready,
            "model": "Conditional Spatial VAE",
            "modelVersion": "2026-10-09",
            "selectionMode": "user-drawn boundary",
            "candidateCounts": list(self.ALLOWED_COUNTS),
            "outputLayers": 13,
            "modelTile": "256 × 256",
            "previewTile": "256 × 256",
            "candidateSelection": "constraint-aware diverse subset",
            "latentTemperature": self.LATENT_TEMPERATURE,
            "sourceResolutionMetres": metadata.get("pixel_size_m", 10),
            "contextLayers": metadata.get("band_names", []),
            "device": str(self._device) if self._device is not None else "loaded on first generation",
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.exists() or not self.context_path.exists():
            raise SpatialGeneratorError("Spatial CVAE model files are unavailable", 503)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cpu":
            torch.set_num_threads(max(1, int(os.getenv("HEATGUARD_TORCH_THREADS", "4"))))
        checkpoint = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)
        self._model = model_from_checkpoint(checkpoint, device=self._device).eval()

    def _site_geometry(self, boundary: list[list[float]]) -> tuple[Polygon, Polygon]:
        if not isinstance(boundary, list) or not 3 <= len(boundary) <= 100:
            raise SpatialGeneratorError("boundary must contain between 3 and 100 longitude/latitude points")
        points: list[tuple[float, float]] = []
        for item in boundary:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise SpatialGeneratorError("Each boundary point must be [longitude, latitude]")
            lon, lat = float(item[0]), float(item[1])
            if not 103.55 <= lon <= 104.10 or not 1.15 <= lat <= 1.50:
                raise SpatialGeneratorError("The drawn site must lie within Singapore's map extent")
            points.append((lon, lat))
        geographic = Polygon(points)
        if not geographic.is_valid:
            geographic = geographic.buffer(0)
        if geographic.is_empty or geographic.geom_type != "Polygon":
            raise SpatialGeneratorError("The drawn site boundary is self-intersecting or invalid")
        projected = transform_geometry(self._to_svy21.transform, geographic)
        area_ha = projected.area / 10_000.0
        if area_ha < 1.0:
            raise SpatialGeneratorError("Draw a site of at least 1 hectare")
        if area_ha > 7_500.0:
            raise SpatialGeneratorError("Draw a site smaller than 7,500 hectares")
        return geographic, projected

    def _condition(
        self,
        boundary: list[list[float]],
        target_density: float,
        target_coverage: float,
    ) -> tuple[np.ndarray, dict]:
        geographic, site_geometry = self._site_geometry(boundary)
        minx, miny, maxx, maxy = site_geometry.bounds
        side_m = max(2_560.0, max(maxx - minx, maxy - miny) + 640.0)
        side_m = min(side_m, 10_000.0)
        center_x, center_y = site_geometry.centroid.coords[0]
        bounds = (
            center_x - side_m / 2,
            center_y - side_m / 2,
            center_x + side_m / 2,
            center_y + side_m / 2,
        )
        transform = from_bounds(*bounds, self.TILE_SIZE, self.TILE_SIZE)
        with rasterio.open(self.context_path) as source:
            singapore = box(*source.bounds)
            if not site_geometry.intersects(singapore):
                raise SpatialGeneratorError("The drawn site does not intersect the Singapore data extent")
            window = window_from_bounds(*bounds, transform=source.transform)
            context = source.read(
                window=window,
                out_shape=(source.count, self.TILE_SIZE, self.TILE_SIZE),
                boundless=True,
                fill_value=0,
                resampling=Resampling.nearest,
            ).astype(np.float32) / 255.0
        site = rasterize(
            [(site_geometry, 1.0)],
            out_shape=(self.TILE_SIZE, self.TILE_SIZE),
            transform=transform,
            fill=0.0,
            all_touched=True,
            dtype="float32",
        )
        if site.sum() < 25:
            raise SpatialGeneratorError("The drawn site is too small at the selected map scale")
        conserved, water, nature, hard, building, road, walk_cycle, green, context_water, land_use = context
        editable = site * (1.0 - np.clip(hard, 0, 1))
        outside = 1.0 - site
        condition = np.stack([
            site,
            editable,
            conserved * site,
            water * site,
            nature * site,
            hard * site,
            building * outside,
            road * outside,
            walk_cycle * outside,
            green * outside,
            np.maximum(water, context_water) * outside,
            land_use,
            site * np.clip(target_density / 250.0, 0, 1),
            site * np.clip(target_coverage / 0.35, 0, 1),
        ]).astype(np.float32)
        centroid = geographic.centroid
        metadata = {
            "name": "Drawn Singapore site",
            "areaHectares": round(site_geometry.area / 10_000.0, 2),
            "centroid": [round(centroid.x, 6), round(centroid.y, 6)],
            "boundary": [[round(lon, 6), round(lat, 6)] for lon, lat in geographic.exterior.coords[:-1]],
            "targetDwellingDensity": target_density,
            "targetBuildingCoverage": target_coverage,
            "effectiveResolutionMetres": round(side_m / self.TILE_SIZE, 2),
            "contextWidthMetres": round(side_m, 1),
        }
        return condition, metadata

    @staticmethod
    def _blend(canvas: np.ndarray, colour: tuple[int, int, int], alpha: np.ndarray) -> None:
        clipped = np.clip(alpha[..., None], 0.0, 1.0)
        canvas[:] = canvas * (1.0 - clipped) + np.asarray(colour, dtype=np.float32) * clipped

    def _preview(self, prediction: torch.Tensor, condition: torch.Tensor) -> str:
        small = F.interpolate(
            prediction[None], size=(self.PREVIEW_SIZE, self.PREVIEW_SIZE), mode="bilinear", align_corners=False
        )[0].detach().cpu().numpy()
        small_condition = F.interpolate(
            condition[None], size=(self.PREVIEW_SIZE, self.PREVIEW_SIZE), mode="nearest"
        )[0].detach().cpu().numpy()
        site = small_condition[0]
        canvas = np.empty((self.PREVIEW_SIZE, self.PREVIEW_SIZE, 3), dtype=np.float32)
        canvas[:] = (235, 241, 239)
        outside = 1.0 - site
        self._blend(canvas, (108, 126, 116), small_condition[9] * outside * 0.34)
        self._blend(canvas, (100, 157, 176), small_condition[10] * outside * 0.44)
        self._blend(canvas, (103, 91, 92), small_condition[6] * outside * 0.30)
        self._blend(canvas, (64, 129, 132), np.maximum(small_condition[7], small_condition[8]) * outside * 0.40)
        canvas[site > 0.5] = (250, 247, 244)
        masks = (small >= 0.5).astype(np.float32) * site[None]
        self._blend(canvas, (61, 126, 166), masks[5] * 0.94)
        self._blend(canvas, (66, 119, 78), masks[3] * 0.92)
        self._blend(canvas, (116, 164, 94), masks[2] * 0.88)
        self._blend(canvas, (157, 188, 103), masks[4] * 0.84)
        self._blend(canvas, (92, 88, 91), masks[6] * 0.86)
        self._blend(canvas, (220, 159, 64), masks[7] * 0.88)
        self._blend(canvas, (38, 127, 128), masks[8] * 0.92)
        building_colour = np.stack([
            205.0 - small[1] * 93.0,
            82.0 - small[1] * 54.0,
            83.0 - small[1] * 35.0,
        ], axis=-1)
        alpha = np.clip(masks[0] * 0.96, 0.0, 1.0)[..., None]
        canvas[:] = canvas * (1.0 - alpha) + building_colour * alpha
        rgba = np.concatenate([
            np.rint(np.clip(canvas, 0, 255)).astype(np.uint8),
            np.full((self.PREVIEW_SIZE, self.PREVIEW_SIZE, 1), 255, np.uint8),
        ], axis=-1)
        return base64.b64encode(rgba.tobytes()).decode("ascii")

    def _signature(self, prediction: torch.Tensor, site: torch.Tensor) -> np.ndarray:
        """Create a compact multi-layer fingerprint for diversity selection."""
        green = torch.maximum(torch.maximum(prediction[2], prediction[3]), prediction[4])
        layers = torch.stack([
            prediction[0], prediction[1], green, prediction[5],
            prediction[6], prediction[7], prediction[8], prediction[9:].amax(dim=0),
        ])
        occupied = torch.nonzero(site >= 0.5, as_tuple=False)
        if occupied.numel():
            y0, x0 = occupied.amin(dim=0)
            y1, x1 = occupied.amax(dim=0)
            layers = layers[:, y0:y1 + 1, x0:x1 + 1]
        pooled = F.adaptive_avg_pool2d(
            layers[None], (self.SIGNATURE_SIZE, self.SIGNATURE_SIZE)
        )[0]
        return pooled.detach().cpu().numpy().astype(np.float16).reshape(-1)

    def _select_diverse(self, candidates: list[dict]) -> list[dict]:
        """Keep feasible candidates, then greedily maximise spatial separation."""
        if len(candidates) <= self.PREVIEW_LIMIT:
            selected = sorted(candidates, key=lambda row: (row["coverageError"], row["id"]))
        else:
            ordered = sorted(candidates, key=lambda row: (row["coverageError"], row["id"]))
            feasible = [row for row in ordered if row["coverageError"] <= 0.10]
            if len(feasible) < self.PREVIEW_LIMIT:
                feasible = ordered[: max(self.PREVIEW_LIMIT, min(len(ordered), 128))]
            else:
                feasible = feasible[: min(len(feasible), 400)]

            selected = [feasible.pop(0)]
            while feasible and len(selected) < self.PREVIEW_LIMIT:
                max_error = max(0.01, max(row["coverageError"] for row in feasible))
                best_index = 0
                best_score = -1.0
                for index, row in enumerate(feasible):
                    signature = row["signature"].astype(np.float32)
                    novelty = min(
                        float(np.mean(np.abs(signature - item["signature"].astype(np.float32))))
                        for item in selected
                    )
                    quality = 1.0 - min(1.0, row["coverageError"] / max_error)
                    score = 0.88 * novelty + 0.12 * quality
                    if score > best_score:
                        best_index, best_score = index, score
                selected.append(feasible.pop(best_index))

        reference = selected[0]["signature"].astype(np.float32)
        for index, row in enumerate(selected):
            signature = row["signature"].astype(np.float32)
            row["layoutDifference"] = 0.0 if index == 0 else float(np.mean(np.abs(signature - reference)))
        return selected

    def generate(
        self,
        boundary: list[list[float]],
        count: int,
        seed: int,
        target_density: float = 80.0,
        target_coverage: float = 0.18,
    ) -> dict:
        if count not in self.ALLOWED_COUNTS:
            raise SpatialGeneratorError(
                f"candidate_count must be one of {', '.join(map(str, self.ALLOWED_COUNTS))}"
            )
        if not 20 <= target_density <= 250:
            raise SpatialGeneratorError("target_dwelling_density must be between 20 and 250 dwellings/ha")
        if not 0.05 <= target_coverage <= 0.35:
            raise SpatialGeneratorError("target_building_coverage must be between 0.05 and 0.35")
        if not self._generation_lock.acquire(blocking=False):
            raise SpatialGeneratorError("Another candidate-generation run is in progress", 409)
        try:
            self._load()
            started = time.perf_counter()
            condition_np, site_metadata = self._condition(boundary, target_density, target_coverage)
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32 - 1))
            condition = torch.from_numpy(condition_np).unsqueeze(0).to(self._device)
            site = condition_np[0] >= 0.5
            hard = condition_np[5] >= 0.5
            batch_size = int(os.getenv(
                "HEATGUARD_CVAE_BATCH_SIZE", "20" if self._device.type == "cuda" else "2"
            ))
            coverages: list[float] = []
            green_coverages: list[float] = []
            water_coverages: list[float] = []
            violations: list[float] = []
            candidates: list[dict] = []
            generated = 0
            with torch.inference_mode():
                while generated < count:
                    batch = min(batch_size, count - generated)
                    batch_condition = condition.expand(batch, -1, -1, -1)
                    z = torch.randn(batch, self._model.latent_dim, device=self._device)
                    z *= self.LATENT_TEMPERATURE
                    prediction = torch.sigmoid(self._model.decode(batch_condition, z))
                    prediction *= batch_condition[:, 0:1]
                    prediction[:, 0, hard] = 0.0
                    binary = prediction >= 0.5
                    for offset in range(batch):
                        item = prediction[offset]
                        building_coverage = float(binary[offset, 0].detach().cpu().numpy()[site].mean())
                        green = torch.maximum(torch.maximum(item[2], item[3]), item[4])
                        green_coverage = float((green.detach().cpu().numpy()[site] >= 0.5).mean())
                        water_coverage = float((item[5].detach().cpu().numpy()[site] >= 0.5).mean())
                        violation = float(binary[offset, 0].detach().cpu().numpy()[hard].mean()) if hard.any() else 0.0
                        candidate_index = generated + offset + 1
                        candidates.append({
                            "id": candidate_index,
                            "coverageError": abs(building_coverage - target_coverage),
                            "buildingCoverage": building_coverage,
                            "greenCoverage": green_coverage,
                            "waterCoverage": water_coverage,
                            "hardExclusionOverlap": violation,
                            "signature": self._signature(item, condition[0, 0]),
                            "latent": z[offset].detach().cpu().numpy().astype(np.float32),
                        })
                        coverages.append(building_coverage)
                        green_coverages.append(green_coverage)
                        water_coverages.append(water_coverage)
                        violations.append(violation)
                    generated += batch
                previews = self._select_diverse(candidates)
                for start in range(0, len(previews), batch_size):
                    rows = previews[start:start + batch_size]
                    preview_z = torch.from_numpy(
                        np.stack([row["latent"] for row in rows])
                    ).to(self._device)
                    preview_condition = condition.expand(len(rows), -1, -1, -1)
                    preview_prediction = torch.sigmoid(self._model.decode(preview_condition, preview_z))
                    preview_prediction *= preview_condition[:, 0:1]
                    preview_prediction[:, 0, hard] = 0.0
                    for row, item in zip(rows, preview_prediction):
                        row["rgbaBase64"] = self._preview(item, condition[0])
                        row["width"] = self.PREVIEW_SIZE
                        row["height"] = self.PREVIEW_SIZE
                        row.pop("signature", None)
                        row.pop("latent", None)
            elapsed = time.perf_counter() - started
            coverage_array = np.asarray(coverages)
            diversity_values = [row["layoutDifference"] for row in previews[1:]]
            return {
                "site": site_metadata,
                "candidateCount": count,
                "seed": seed,
                "elapsedSeconds": round(elapsed, 3),
                "device": str(self._device),
                "summary": {
                    "buildingCoverageMean": float(coverage_array.mean()),
                    "buildingCoverageMin": float(coverage_array.min()),
                    "buildingCoverageMax": float(coverage_array.max()),
                    "withinTarget05": int((np.abs(coverage_array - target_coverage) <= 0.05).sum()),
                    "greenCoverageMean": float(np.mean(green_coverages)),
                    "waterCoverageMean": float(np.mean(water_coverages)),
                    "hardExclusionOverlapMax": float(np.max(violations)),
                    "selectedLayoutDifferenceMean": float(np.mean(diversity_values)) if diversity_values else 0.0,
                },
                "previews": previews,
            }
        finally:
            self._generation_lock.release()

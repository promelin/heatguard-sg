"""Orchestrate ingestion, feature building, V4 inference and JSON publication."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .data_gov import DataGovClient, WeatherCache
from .local_environment import LocalHeatIndexModel
from .predictor import V4Predictor


LOGGER = logging.getLogger(__name__)


class ForecastService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.data_dir = Path(os.getenv("HEATGUARD_DATA_DIR", project_root / "backend" / "data"))
        self.model_path = Path(os.getenv(
            "HEATGUARD_MODEL_PATH", project_root / "backend" / "model" / "best_model.pt"
        ))
        self.metadata_path = Path(os.getenv(
            "HEATGUARD_METADATA_PATH", project_root / "backend" / "model" / "v4_runtime_metadata.json"
        ))
        self.local_heat_model_path = Path(os.getenv(
            "HEATGUARD_LOCAL_HEAT_MODEL_PATH",
            project_root / "backend" / "model" / "local_heat_model.joblib",
        ))
        self.output_path = self.data_dir / "heatguard-data-live.json"
        self.fallback_path = project_root / "dist" / "heatguard-data.json"
        self.cache = WeatherCache(self.data_dir / "weather_cache.sqlite3")
        self.client = DataGovClient(self.cache)
        self.predictor: V4Predictor | None = None
        self.local_heat_model: LocalHeatIndexModel | None = None
        self.lock = asyncio.Lock()
        self.last_run: str | None = None
        self.last_success: str | None = None
        self.last_current_success: str | None = None
        self.last_error: str | None = None
        self.last_ingestion: dict[str, Any] | None = None
        self.refresh_seconds = max(300, int(os.getenv("HEATGUARD_REFRESH_SECONDS", "3600")))

    def _get_predictor(self) -> V4Predictor:
        if self.predictor is None:
            self.predictor = V4Predictor(self.model_path, self.metadata_path)
        return self.predictor

    def _get_local_heat_model(self) -> LocalHeatIndexModel:
        if self.local_heat_model is None:
            self.local_heat_model = LocalHeatIndexModel(
                self.local_heat_model_path,
                self.project_root / "dist" / "data" / "spatial",
            )
        return self.local_heat_model

    def estimate_local_heat(
        self,
        longitude: float,
        latitude: float,
        radius_m: int,
        background_heat_index_c: float | None,
    ) -> dict[str, Any]:
        if background_heat_index_c is None:
            payload, _ = self.read_payload()
            current = payload.get("current") or {}
            background_heat_index_c = (
                current.get("networkMean")
                or payload.get("networkActualMean")
                or payload.get("networkMean")
            )
        if background_heat_index_c is None:
            raise ValueError("A background heat index is required when live station data is unavailable")
        return self._get_local_heat_model().predict(
            longitude=longitude,
            latitude=latitude,
            radius_m=radius_m,
            background_heat_index_c=float(background_heat_index_c),
        )

    def _write_atomic(self, payload: dict[str, Any]) -> None:
        written_at = datetime.now(timezone.utc)
        payload["refreshSeconds"] = self.refresh_seconds
        payload["nextRefreshAt"] = (written_at + timedelta(seconds=self.refresh_seconds)).isoformat()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.output_path)

    def _publish_current(self, current: dict[str, Any]) -> dict[str, Any]:
        """Merge current observations into the last forecast or packaged replay."""
        source = self.output_path if self.output_path.exists() else self.fallback_path
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["current"] = current
        payload["generatedAt"] = datetime.now(timezone.utc).isoformat()
        if payload.get("mode") != "live forecast":
            payload["mode"] = "live observations · forecast replay"
        self._write_atomic(payload)
        return payload

    async def refresh(self) -> dict[str, Any]:
        async with self.lock:
            started = datetime.now(timezone.utc)
            self.last_run = started.isoformat()
            try:
                latest_ingestion = await self.client.refresh_latest()
                recent_rows = await asyncio.to_thread(
                    self.cache.observations_since, datetime.now(timezone.utc) - timedelta(days=2)
                )
                predictor = await asyncio.to_thread(self._get_predictor)
                current = await asyncio.to_thread(predictor.current_conditions, recent_rows)
                if current:
                    await asyncio.to_thread(self._publish_current, current)
                    self.last_current_success = datetime.now(timezone.utc).isoformat()
                    LOGGER.info(
                        "Published current heat index for %s stations at %s",
                        len(current["stations"]), current["observedAt"],
                    )

                history_ingestion = await self.client.backfill(backfill_days=8)
                self.last_ingestion = {
                    "inserted_or_updated": (
                        latest_ingestion["inserted_or_updated"]
                        + history_ingestion["inserted_or_updated"]
                    ),
                    "latest_observation": self.cache.latest_timestamp(),
                    "errors": [*latest_ingestion["errors"], *history_ingestion["errors"]],
                }
                rows = await asyncio.to_thread(
                    self.cache.observations_since, datetime.now(timezone.utc) - timedelta(days=9)
                )
                payload = await asyncio.to_thread(predictor.predict, rows)
                await asyncio.to_thread(self._write_atomic, payload)
                self.last_success = datetime.now(timezone.utc).isoformat()
                self.last_error = None
                LOGGER.info(
                    "Published live forecast for %s from %s cached observations",
                    payload["issueTime"], len(rows),
                )
                return payload
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                LOGGER.exception("Forecast refresh failed")
                raise

    async def run_periodically(self) -> None:
        while True:
            try:
                await self.refresh()
            except Exception:
                pass
            await asyncio.sleep(self.refresh_seconds)

    def read_payload(self) -> tuple[dict[str, Any], bool]:
        path = self.output_path if self.output_path.exists() else self.fallback_path
        return json.loads(path.read_text(encoding="utf-8")), path == self.output_path

    def status(self) -> dict[str, Any]:
        live_current_available = False
        live_forecast_available = False
        if self.output_path.exists():
            try:
                payload = json.loads(self.output_path.read_text(encoding="utf-8"))
                live_current_available = bool((payload.get("current") or {}).get("stations"))
                live_forecast_available = payload.get("mode") == "live forecast"
            except (OSError, json.JSONDecodeError):
                pass
        model_ready = self.model_path.exists() and self.metadata_path.exists()
        local_heat_model_ready = self.local_heat_model_path.exists()
        return {
            "status": "ok" if (live_current_available or live_forecast_available) else "starting",
            "liveObservationsAvailable": live_current_available,
            "liveForecastAvailable": live_forecast_available,
            "modelReady": model_ready,
            "localHeatModelReady": local_heat_model_ready,
            "modelVersion": "V4-history-only",
            "refreshSeconds": self.refresh_seconds,
            "lastRun": self.last_run,
            "lastSuccess": self.last_success,
            "lastCurrentSuccess": self.last_current_success,
            "lastError": self.last_error,
            "ingestion": self.last_ingestion,
        }

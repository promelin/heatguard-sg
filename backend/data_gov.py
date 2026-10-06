"""Rate-limit-aware ingestion and local caching for data.gov.sg weather APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx


LOGGER = logging.getLogger(__name__)
API_ROOT = "https://api-open.data.gov.sg/v2/real-time/api"
ENDPOINTS = {
    "temperature": "air-temperature",
    "humidity": "relative-humidity",
    "rainfall": "rainfall",
}


class WeatherCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS observations (
                    metric TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (metric, timestamp, station_id)
                );
                CREATE TABLE IF NOT EXISTS stations (
                    metric TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    name TEXT,
                    latitude REAL,
                    longitude REAL,
                    PRIMARY KEY (metric, station_id)
                );
                CREATE TABLE IF NOT EXISTS fetched_dates (
                    metric TEXT NOT NULL,
                    date TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (metric, date)
                );
                """
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def has_date(self, metric: str, target_date: date) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM fetched_dates WHERE metric = ? AND date = ?",
                (metric, target_date.isoformat()),
            ).fetchone()
        return row is not None

    def store_payload(self, metric: str, payload: dict[str, Any]) -> int:
        data = payload.get("data") or {}
        stations = data.get("stations") or []
        readings = data.get("readings") or []
        station_rows = []
        for station in stations:
            location = station.get("location") or {}
            station_rows.append((
                metric, str(station.get("id")), station.get("name"),
                location.get("latitude"), location.get("longitude"),
            ))
        observation_rows = []
        for reading in readings:
            timestamp = reading.get("timestamp")
            if not timestamp:
                continue
            try:
                normalized_timestamp = datetime.fromisoformat(
                    str(timestamp).replace("Z", "+00:00")
                ).astimezone(timezone.utc).isoformat()
            except ValueError:
                continue
            for item in reading.get("data") or []:
                station_id = item.get("stationId")
                value = item.get("value")
                if station_id is None or value is None:
                    continue
                try:
                    observation_rows.append((metric, normalized_timestamp, str(station_id), float(value)))
                except (TypeError, ValueError):
                    continue
        with self.connect() as connection:
            connection.executemany(
                """INSERT INTO stations(metric, station_id, name, latitude, longitude)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(metric, station_id) DO UPDATE SET
                     name=excluded.name, latitude=excluded.latitude, longitude=excluded.longitude""",
                station_rows,
            )
            connection.executemany(
                """INSERT OR REPLACE INTO observations(metric, timestamp, station_id, value)
                   VALUES (?, ?, ?, ?)""",
                observation_rows,
            )
        return len(observation_rows)

    def mark_date(self, metric: str, target_date: date) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO fetched_dates(metric, date, fetched_at) VALUES (?, ?, ?)",
                (metric, target_date.isoformat(), datetime.now(timezone.utc).isoformat()),
            )

    def observations_since(self, since: datetime) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT o.metric, o.timestamp, o.station_id, o.value,
                          s.name, s.latitude, s.longitude
                   FROM observations o
                   LEFT JOIN stations s
                     ON s.metric = o.metric AND s.station_id = o.station_id
                   WHERE o.timestamp >= ?
                   ORDER BY o.timestamp, o.metric, o.station_id""",
                (since.astimezone(timezone.utc).isoformat(),),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_timestamp(self) -> str | None:
        with self.connect() as connection:
            row = connection.execute("SELECT MAX(timestamp) AS value FROM observations").fetchone()
        return row["value"] if row else None

    def prune(self, before: datetime) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM observations WHERE timestamp < ?",
                (before.astimezone(timezone.utc).isoformat(),),
            )


class DataGovClient:
    def __init__(self, cache: WeatherCache) -> None:
        self.cache = cache
        self.api_key = os.getenv("DATA_GOV_SG_API_KEY", "").strip()
        self.request_gap_seconds = 0.05 if self.api_key else 1.8
        self.timeout = float(os.getenv("DATA_GOV_SG_TIMEOUT_SECONDS", "45"))

    async def _fetch(self, client: httpx.AsyncClient, metric: str, params: dict[str, str]) -> dict[str, Any]:
        headers = {
            "accept": "application/json",
            "user-agent": "HeatGuard-SG/4.0 (+https://daisi.online/guide)",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        endpoint = ENDPOINTS[metric]
        for attempt in range(5):
            response = await client.get(f"{API_ROOT}/{endpoint}", params=params, headers=headers)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = float(response.headers.get("retry-after", 0) or 0)
                await asyncio.sleep(max(retry_after, min(30.0, 2.0 ** attempt)))
                continue
            response.raise_for_status()
            payload = response.json()
            if payload.get("code", 0) != 0:
                raise RuntimeError(
                    f"data.gov.sg returned code {payload.get('code')}: {payload.get('errorMsg')}"
                )
            return payload
        response.raise_for_status()
        raise RuntimeError("data.gov.sg request failed after retries")

    async def fetch_metric(self, metric: str, target_date: date | None = None) -> int:
        params = {"date": target_date.isoformat()} if target_date else {}
        total = 0
        seen_tokens: set[str] = set()
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            while True:
                payload = await self._fetch(client, metric, params)
                total += self.cache.store_payload(metric, payload)
                token = (payload.get("data") or {}).get("paginationToken")
                if not token:
                    break
                if str(token) in seen_tokens:
                    raise RuntimeError("data.gov.sg returned a repeated pagination token")
                seen_tokens.add(str(token))
                params = {**params, "paginationToken": str(token)}
                await asyncio.sleep(self.request_gap_seconds)
        if target_date is not None and total:
            self.cache.mark_date(metric, target_date)
        return total

    async def refresh_latest(self) -> dict[str, Any]:
        """Fetch the three latest streams first so the live map can update quickly."""
        inserted = 0
        errors: list[str] = []
        for metric in ENDPOINTS:
            try:
                inserted += await self.fetch_metric(metric)
            except Exception as error:
                message = f"{metric} latest: {error}"
                LOGGER.exception("Latest weather refresh failed: %s", message)
                errors.append(message)
            await asyncio.sleep(self.request_gap_seconds)
        return {
            "inserted_or_updated": inserted,
            "latest_observation": self.cache.latest_timestamp(),
            "errors": errors,
        }

    async def backfill(self, backfill_days: int = 8) -> dict[str, Any]:
        """Fill the rolling model history after the latest readings are available."""
        today_sgt = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
        inserted = 0
        errors: list[str] = []
        for offset in range(backfill_days - 1, -1, -1):
            target_date = today_sgt - timedelta(days=offset)
            for metric in ENDPOINTS:
                # Today's endpoint is deliberately re-read so a service restart
                # cannot leave a gap between the earlier backfill and "latest".
                if target_date < today_sgt and self.cache.has_date(metric, target_date):
                    continue
                try:
                    inserted += await self.fetch_metric(metric, target_date)
                except Exception as error:  # retain other streams and cached history
                    message = f"{metric} {target_date}: {error}"
                    LOGGER.exception("Weather backfill failed: %s", message)
                    errors.append(message)
                await asyncio.sleep(self.request_gap_seconds)
        self.cache.prune(datetime.now(timezone.utc) - timedelta(days=14))
        return {
            "inserted_or_updated": inserted,
            "latest_observation": self.cache.latest_timestamp(),
            "errors": errors,
        }

    async def refresh(self, backfill_days: int = 8) -> dict[str, Any]:
        """Compatibility entry point: latest readings first, then model history."""
        latest = await self.refresh_latest()
        history = await self.backfill(backfill_days=backfill_days)
        return {
            "inserted_or_updated": latest["inserted_or_updated"] + history["inserted_or_updated"],
            "latest_observation": self.cache.latest_timestamp(),
            "errors": [*latest["errors"], *history["errors"]],
        }


def cache_summary(path: Path) -> str:
    """Small operator-facing diagnostic used by command-line checks."""
    cache = WeatherCache(path)
    return json.dumps({"path": str(path), "latest": cache.latest_timestamp()})

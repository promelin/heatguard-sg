"""Privacy-minimised storage for structured community cooling reports."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ALLOWED_KINDS = {"proposal", "confirmation", "maintenance", "comfort"}
ALLOWED_CHANNELS = {"web", "qr", "aac"}
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
REQUIRED_PHOTO_KINDS = {"proposal", "confirmation", "maintenance"}
MAX_PHOTO_BYTES = 5 * 1024 * 1024
SINGAPORE_BOUNDS = {"min_lon": 103.55, "max_lon": 104.10, "min_lat": 1.15, "max_lat": 1.50}
DETAIL_FIELDS = {
    "proposal": {"siteType", "approxSize", "shade", "seating", "plantTypes", "approvalStatus"},
    "confirmation": {"proposalReference", "verifierRole"},
    "maintenance": {"pocketReference", "conditionRating", "issue"},
    "comfort": {"qrReference", "comfort"},
}


class CommunitySubmissionError(ValueError):
    """Validation or rate-control error with an HTTP-friendly status code."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class CommunityDataStore:
    """Store private submissions without names or public exact-location output."""

    def __init__(self, project_root: Path):
        runtime_dir = project_root / "runtime"
        self.database_path = Path(os.getenv("HEATGUARD_COMMUNITY_DB", runtime_dir / "community-submissions.sqlite3"))
        self.upload_dir = Path(os.getenv("HEATGUARD_COMMUNITY_UPLOAD_DIR", runtime_dir / "community-uploads"))
        self.weekly_cap = max(1, int(os.getenv("HEATGUARD_COMMUNITY_WEEKLY_CAP", "10")))
        self._lock = threading.Lock()
        self._prepare()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def _prepare(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS community_submissions (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    area TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    observed_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    device_hash TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    photo_path TEXT,
                    status TEXT NOT NULL DEFAULT 'received',
                    duplicate_key TEXT
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS community_device_time ON community_submissions(device_hash, created_at)")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS community_duplicate_issue ON community_submissions(duplicate_key) WHERE duplicate_key IS NOT NULL")

    @staticmethod
    def _clean_reference(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "", str(value or ""))[:32]

    def _validate(self, record: dict[str, Any]) -> dict[str, Any]:
        kind = str(record.get("kind", ""))
        channel = str(record.get("channel", ""))
        if kind not in ALLOWED_KINDS:
            raise CommunitySubmissionError("Unknown contribution type.")
        if channel not in ALLOWED_CHANNELS:
            raise CommunitySubmissionError("Unknown submission channel.")
        if record.get("consent") is not True:
            raise CommunitySubmissionError("Consent is required.")

        area = str(record.get("area", "")).strip()[:80]
        coordinates = record.get("coordinates") or {}
        try:
            latitude = float(coordinates["lat"])
            longitude = float(coordinates["lon"])
        except (KeyError, TypeError, ValueError) as error:
            raise CommunitySubmissionError("Valid coordinates are required.") from error
        if not (SINGAPORE_BOUNDS["min_lat"] <= latitude <= SINGAPORE_BOUNDS["max_lat"] and SINGAPORE_BOUNDS["min_lon"] <= longitude <= SINGAPORE_BOUNDS["max_lon"]):
            raise CommunitySubmissionError("The contribution location must be within Singapore.")
        if not area:
            raise CommunitySubmissionError("Planning area is required.")

        details = record.get("details")
        if not isinstance(details, dict):
            raise CommunitySubmissionError("Structured contribution details are required.")
        allowed = DETAIL_FIELDS[kind]
        clean_details = {key: value for key, value in details.items() if key in allowed}
        for reference_key in ("proposalReference", "pocketReference", "qrReference"):
            if reference_key in clean_details:
                clean_details[reference_key] = self._clean_reference(clean_details[reference_key])

        request_id = self._clean_reference(record.get("requestId")) or uuid.uuid4().hex
        device_token = str(record.get("deviceToken", ""))
        if len(device_token) < 8:
            raise CommunitySubmissionError("A valid anonymous device token is required.")

        photo = record.get("photo")
        if kind in REQUIRED_PHOTO_KINDS and not isinstance(photo, dict):
            raise CommunitySubmissionError("A location-and-time tagged photo is required.")

        return {
            "request_id": request_id,
            "kind": kind,
            "channel": channel,
            "area": area,
            "latitude": latitude,
            "longitude": longitude,
            "observed_at": str(record.get("observedAt") or datetime.now(timezone.utc).isoformat())[:40],
            "device_hash": hashlib.sha256(device_token.encode("utf-8")).hexdigest(),
            "details": clean_details,
            "photo": photo,
        }

    def _save_photo(self, submission_id: str, photo: dict[str, Any] | None) -> str | None:
        if not photo:
            return None
        data_url = str(photo.get("dataUrl", ""))
        match = re.fullmatch(r"data:([^;]+);base64,(.+)", data_url, flags=re.DOTALL)
        if not match or match.group(1) not in ALLOWED_IMAGE_TYPES:
            raise CommunitySubmissionError("Photo must be a JPG, PNG or WebP image.")
        try:
            content = base64.b64decode(match.group(2), validate=True)
        except (ValueError, binascii.Error) as error:
            raise CommunitySubmissionError("Photo encoding is invalid.") from error
        if not content or len(content) > MAX_PHOTO_BYTES:
            raise CommunitySubmissionError("Photo must be between 1 byte and 5 MB.", 413)
        mime_type = match.group(1)
        valid_signature = (
            (mime_type == "image/jpeg" and content.startswith(b"\xff\xd8\xff"))
            or (mime_type == "image/png" and content.startswith(b"\x89PNG\r\n\x1a\n"))
            or (mime_type == "image/webp" and content.startswith(b"RIFF") and content[8:12] == b"WEBP")
        )
        if not valid_signature:
            raise CommunitySubmissionError("Photo content does not match its declared image type.")
        extension = ALLOWED_IMAGE_TYPES[mime_type]
        path = self.upload_dir / f"{submission_id}{extension}"
        path.write_bytes(content)
        return str(path.relative_to(self.database_path.parent))

    @staticmethod
    def _duplicate_key(clean: dict[str, Any], created_at: datetime) -> str | None:
        if clean["kind"] != "maintenance" or clean["details"].get("issue") in (None, "none"):
            return None
        week = created_at.strftime("%G-W%V")
        source = "|".join([
            clean["kind"],
            str(clean["details"].get("issue")),
            f"{clean['latitude']:.3f}",
            f"{clean['longitude']:.3f}",
            week,
        ])
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def submit(self, record: dict[str, Any]) -> dict[str, Any]:
        clean = self._validate(record)
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=7)).isoformat()
        submission_id = f"HG-{now:%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
        duplicate_key = self._duplicate_key(clean, now)
        photo_path: str | None = None

        with self._lock, self._connect() as connection:
            prior = connection.execute(
                "SELECT id, status, created_at FROM community_submissions WHERE request_id = ?",
                (clean["request_id"],),
            ).fetchone()
            if prior:
                return {"id": prior["id"], "status": prior["status"], "createdAt": prior["created_at"], "duplicate": True}
            recent_count = connection.execute(
                "SELECT COUNT(*) FROM community_submissions WHERE device_hash = ? AND created_at >= ?",
                (clean["device_hash"], cutoff),
            ).fetchone()[0]
            if recent_count >= self.weekly_cap:
                raise CommunitySubmissionError("Weekly contribution limit reached.", 429)
            if duplicate_key and connection.execute(
                "SELECT 1 FROM community_submissions WHERE duplicate_key = ?",
                (duplicate_key,),
            ).fetchone():
                raise CommunitySubmissionError("This issue has already been reported for the current week.", 409)

            photo_path = self._save_photo(submission_id, clean["photo"])
            try:
                connection.execute(
                    """
                    INSERT INTO community_submissions (
                        id, request_id, kind, channel, area, latitude, longitude,
                        observed_at, created_at, device_hash, details_json,
                        photo_path, status, duplicate_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'received', ?)
                    """,
                    (
                        submission_id,
                        clean["request_id"],
                        clean["kind"],
                        clean["channel"],
                        clean["area"],
                        clean["latitude"],
                        clean["longitude"],
                        clean["observed_at"],
                        now.isoformat(),
                        clean["device_hash"],
                        json.dumps(clean["details"], ensure_ascii=False, separators=(",", ":")),
                        photo_path,
                        duplicate_key,
                    ),
                )
                connection.commit()
            except Exception:
                if photo_path:
                    (self.database_path.parent / photo_path).unlink(missing_ok=True)
                raise

        return {"id": submission_id, "status": "received", "createdAt": now.isoformat(), "duplicate": False}

    def receipt_status(self, submission_id: str) -> dict[str, Any] | None:
        safe_id = re.sub(r"[^A-Z0-9-]", "", submission_id.upper())[:40]
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, kind, status, created_at FROM community_submissions WHERE id = ?",
                (safe_id,),
            ).fetchone()
        if not row:
            return None
        return {"id": row["id"], "kind": row["kind"], "status": row["status"], "createdAt": row["created_at"]}

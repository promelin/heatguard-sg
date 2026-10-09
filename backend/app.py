"""FastAPI entry point for the live HeatGuard SG website."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .service import ForecastService
from .community_data import CommunityDataStore, CommunitySubmissionError


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
service = ForecastService(PROJECT_ROOT)
community_store = CommunityDataStore(PROJECT_ROOT)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(service.run_periodically(), name="heatguard-live-refresh")
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="HeatGuard SG live heat and forecast", version="3.3", lifespan=lifespan)
origins = [origin.strip() for origin in os.getenv("HEATGUARD_CORS_ORIGINS", "").split(",") if origin.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


@app.get("/api/health")
async def health() -> dict:
    return service.status()


@app.get("/api/heatguard-data.json")
async def heatguard_data() -> JSONResponse:
    payload, is_live = await asyncio.to_thread(service.read_payload)
    return JSONResponse(
        payload,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-HeatGuard-Data-Mode": "live" if is_live else "fallback-replay",
        },
    )


@app.post("/api/community-submissions", status_code=201)
async def create_community_submission(payload: dict = Body(...)) -> JSONResponse:
    """Accept a privacy-minimised structured report for the council-review queue."""
    try:
        receipt = await asyncio.to_thread(community_store.submit, payload)
    except CommunitySubmissionError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    return JSONResponse(receipt, status_code=201, headers={"Cache-Control": "no-store"})


@app.get("/api/community-submissions/{submission_id}")
async def community_submission_status(submission_id: str) -> JSONResponse:
    """Return receipt status without exposing coordinates, photos or report details."""
    receipt = await asyncio.to_thread(community_store.receipt_status, submission_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail="Submission receipt not found.")
    return JSONResponse(receipt, headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "dist", html=True), name="site")

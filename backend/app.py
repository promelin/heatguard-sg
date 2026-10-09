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
from .spatial_generator import SpatialGeneratorError, SpatialLayoutGenerator
from .spatial_jobs import SpatialGenerationJobs


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
service = ForecastService(PROJECT_ROOT)
community_store = CommunityDataStore(PROJECT_ROOT)
spatial_generator = SpatialLayoutGenerator(PROJECT_ROOT)
spatial_jobs = SpatialGenerationJobs(spatial_generator)


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


@app.post("/api/local-heat-index")
async def local_heat_index(payload: dict = Body(...)) -> JSONResponse:
    """Estimate local heat from nearby vegetation, buildings, coast and location."""
    try:
        longitude = float(payload["longitude"])
        latitude = float(payload["latitude"])
        radius_m = int(payload.get("radius_m", 500))
        background = payload.get("background_heat_index_c")
        if background is not None:
            background = float(background)
        if not 103.55 <= longitude <= 104.10 or not 1.15 <= latitude <= 1.50:
            raise ValueError("Location must be within Singapore's map extent")
        if not 200 <= radius_m <= 2_000:
            raise ValueError("radius_m must be between 200 and 2000")
        if background is not None and not 20.0 <= background <= 55.0:
            raise ValueError("background_heat_index_c must be between 20 and 55")
        result = await asyncio.to_thread(
            service.estimate_local_heat,
            longitude,
            latitude,
            radius_m,
            background,
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store, max-age=0"})
    except KeyError as error:
        raise HTTPException(status_code=422, detail=f"Missing field: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


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


@app.get("/api/spatial-layout/status")
async def spatial_layout_status() -> JSONResponse:
    """Return arbitrary-site spatial model readiness."""
    try:
        return JSONResponse(spatial_generator.status(), headers={"Cache-Control": "no-store"})
    except SpatialGeneratorError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@app.post("/api/spatial-layout/generate")
async def generate_spatial_layouts(payload: dict = Body(...)) -> JSONResponse:
    """Generate nature-positive candidates for a user-drawn Singapore site."""
    try:
        boundary = payload["boundary"]
        candidate_count = int(payload.get("candidate_count", 100))
        seed = int(payload.get("seed", 20261009))
        target_density = float(payload.get("target_dwelling_density", 80))
        target_coverage = float(payload.get("target_building_coverage", 0.18))
        result = await asyncio.to_thread(
            spatial_generator.generate,
            boundary,
            candidate_count,
            seed,
            target_density,
            target_coverage,
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except KeyError as error:
        raise HTTPException(status_code=422, detail=f"Missing field: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except SpatialGeneratorError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@app.post("/api/spatial-layout/jobs", status_code=202)
async def create_spatial_layout_job(payload: dict = Body(...)) -> JSONResponse:
    """Queue a generation run so 100–1,000 candidate requests can complete safely."""
    try:
        boundary = payload["boundary"]
        candidate_count = int(payload.get("candidate_count", 100))
        seed = int(payload.get("seed", 20261009))
        target_density = float(payload.get("target_dwelling_density", 80))
        target_coverage = float(payload.get("target_building_coverage", 0.18))
        if candidate_count not in spatial_generator.ALLOWED_COUNTS:
            raise ValueError(
                f"candidate_count must be one of {', '.join(map(str, spatial_generator.ALLOWED_COUNTS))}"
            )
        spatial_generator._site_geometry(boundary)
        if not 20 <= target_density <= 250:
            raise ValueError("target_dwelling_density must be between 20 and 250 dwellings/ha")
        if not 0.05 <= target_coverage <= 0.35:
            raise ValueError("target_building_coverage must be between 0.05 and 0.35")
        job = spatial_jobs.create({
            "boundary": boundary,
            "candidate_count": candidate_count,
            "seed": seed,
            "target_density": target_density,
            "target_coverage": target_coverage,
        })
        return JSONResponse(job, status_code=202, headers={"Cache-Control": "no-store"})
    except KeyError as error:
        raise HTTPException(status_code=422, detail=f"Missing field: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except SpatialGeneratorError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@app.get("/api/spatial-layout/jobs/{job_id}")
async def spatial_layout_job(job_id: str) -> JSONResponse:
    try:
        return JSONResponse(spatial_jobs.get(job_id), headers={"Cache-Control": "no-store"})
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Generation job not found") from error


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "dist", html=True), name="site")

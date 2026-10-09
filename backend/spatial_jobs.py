"""Small in-process queue for longer spatial candidate-generation runs."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .spatial_generator import SpatialLayoutGenerator


class SpatialGenerationJobs:
    def __init__(self, generator: SpatialLayoutGenerator) -> None:
        self.generator = generator
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spatial-cvae")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _prune(self) -> None:
        cutoff = time.time() - 3600
        stale = [job_id for job_id, row in self._jobs.items() if row["updatedAt"] < cutoff]
        for job_id in stale:
            self._jobs.pop(job_id, None)

    def create(self, request: dict) -> dict:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._prune()
            self._jobs[job_id] = {
                "jobId": job_id,
                "status": "queued",
                "candidateCount": request["candidate_count"],
                "targetDwellingDensity": request["target_density"],
                "targetBuildingCoverage": request["target_coverage"],
                "createdAt": now,
                "updatedAt": now,
            }
        self._executor.submit(self._run, job_id, request)
        return self.get(job_id)

    def _run(self, job_id: str, request: dict) -> None:
        with self._lock:
            self._jobs[job_id].update(status="running", updatedAt=time.time())
        try:
            result = self.generator.generate(
                request["boundary"],
                request["candidate_count"],
                request["seed"],
                request["target_density"],
                request["target_coverage"],
            )
            with self._lock:
                self._jobs[job_id].update(status="complete", result=result, updatedAt=time.time())
        except Exception as error:  # background boundary; converted to a safe job failure
            with self._lock:
                self._jobs[job_id].update(status="failed", error=str(error), updatedAt=time.time())

    def get(self, job_id: str) -> dict:
        with self._lock:
            row = self._jobs.get(job_id)
            if row is None:
                raise KeyError(job_id)
            return dict(row)

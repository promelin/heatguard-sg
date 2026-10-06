"""Compute and atomically publish an actual V4 24-hour forecast for Pages."""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import math
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from backend.data_gov import DataGovClient, WeatherCache
from backend.predictor import V4Predictor, parse_timestamp
from scripts.publish_risk_results import attach_risk_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
CACHE = ROOT/'.cache'
DATABASE = CACHE/'weather.sqlite3'
COMPRESSED = CACHE/'weather.sqlite3.gz'


def restore_cache():
    CACHE.mkdir(exist_ok=True)
    if DATABASE.exists():
        return
    archive = COMPRESSED if COMPRESSED.exists() else ROOT/'backend/data/seed-weather.sqlite3.gz'
    if archive.exists():
        with gzip.open(archive, 'rb') as src, DATABASE.open('wb') as out:
            shutil.copyfileobj(src, out)


def save_cache():
    if not DATABASE.exists():
        return
    backup = CACHE/'weather-backup.sqlite3'
    with sqlite3.connect(DATABASE) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)
    temporary = COMPRESSED.with_suffix('.gz.tmp')
    with backup.open('rb') as src, gzip.open(temporary, 'wb', compresslevel=5) as out:
        shutil.copyfileobj(src, out)
    temporary.replace(COMPRESSED)


def validate(payload):
    now = datetime.now(timezone.utc)
    age = now-parse_timestamp(payload['issueTime'])
    if age > timedelta(hours=3) or age < timedelta(minutes=-5):
        raise ValueError(f'Refusing to publish stale forecast origin: {payload["issueTime"]}')
    quality = payload['dataQuality']
    for metric, minimum in [('temperatureCoverage', .45), ('humidityCoverage', .45), ('rainfallCoverage', .5)]:
        if quality[metric] < minimum:
            raise ValueError(f'Insufficient historical coverage: {metric}={quality[metric]}')
    if len(payload['stations']) < 8:
        raise ValueError('Too few forecast stations')
    for station in payload['stations']:
        if len(station['series']) != 24:
            raise ValueError('Every forecast must contain 24 hourly predictions')
        for point in station['series']:
            if not all(math.isfinite(point[k]) for k in ('predicted', 'lower', 'upper')):
                raise ValueError('Non-finite prediction')
            if not point['lower'] <= point['predicted'] <= point['upper']:
                raise ValueError('Invalid prediction interval')
    return quality


async def main():
    torch.set_num_threads(2)
    restore_cache()
    cache = WeatherCache(DATABASE)
    client = DataGovClient(cache)
    try:
        latest = await client.refresh_latest()
        logging.info('Latest observations: %s', latest['latest_observation'])
        history = await client.backfill(backfill_days=8)
        rows = cache.observations_since(datetime.now(timezone.utc)-timedelta(days=9))
        predictor = V4Predictor(ROOT/'backend/model/best_model.pt',
                                ROOT/'backend/model/v4_runtime_metadata.json')
        payload = predictor.predict(rows)
        quality = validate(payload)
        risk_count = attach_risk_results(payload)
        logging.info('Computed %s area, subzone and scenario risk outputs', risk_count)
        generated = parse_timestamp(payload['generatedAt'])
        next_run = generated.replace(minute=17, second=0, microsecond=0)
        if next_run <= generated:
            next_run += timedelta(hours=1)
        payload.update({
            'refreshSeconds': 3600,
            'nextRefreshAt': next_run.isoformat(),
            'publication': {
                'type': 'scheduled-model-inference',
                'model': 'V4-history-only',
                'note': 'V4 runs hourly from historical station observations and publishes the latest validated result.',
                'ingestionWarnings': len(latest['errors'])+len(history['errors']),
            },
        })
        target = ROOT/'dist/heatguard-data.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)
        logging.info('Published V4: origin=%s, window=%s to %s, quality=%s',
                     payload['issueTime'], payload['windowStart'], payload['windowEnd'], quality)
    finally:
        save_cache()


if __name__ == '__main__':
    asyncio.run(main())

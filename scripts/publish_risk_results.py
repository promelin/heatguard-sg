"""Export model outputs for displayed areas/times/scenarios, never model trees."""
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def attach_risk_results(payload):
    assets = ROOT/'dist/data/spatial'
    areas = json.loads((assets/'areas.geojson').read_text())['features']
    subzones = json.loads((assets/'subzones.geojson').read_text())['features']
    greens = {row['id']: row for row in json.loads((assets/'green-metrics.json').read_text())['areas']}
    script = (ROOT/'dist/app.js').read_text()
    block = script.split('const STATION_META = {', 1)[1].split('\n};', 1)[0]
    stations = {sid: (float(lon), float(lat)) for sid, lon, lat in re.findall(
        r'(S\d+):\s*\{\s*name:\s*"[^"]+",\s*lon:\s*([\d.]+),\s*lat:\s*([\d.]+)', block)}
    model_bundle = joblib.load(ROOT/'backend/model/risk_model.joblib')
    model = model_bundle['model']
    networks = [[(station['id'], station['series'][lead]['predicted'])
                 for station in payload['stations']] for lead in range(24)]
    if payload.get('current'):
        networks.append([(station['id'], station['heatIndex']) for station in payload['current']['stations']])

    def heat_at(centroid, network):
        numerator = denominator = 0.0
        for station, value in network:
            if station not in stations:
                continue
            lon, lat = stations[station]
            dx = (centroid[0]-lon)*math.cos(1.35*math.pi/180)
            dy = centroid[1]-lat
            weight = 1/(dx*dx+dy*dy+.000025)
            numerator += value*weight
            denominator += weight
        return numerator/denominator if denominator else 0.0

    records = []
    identities = []
    seen = set()
    by_area = {f['properties']['id']: f['properties'] for f in areas}
    for prefix, features in [('area', areas), ('subzone', subzones)]:
        for feature in features:
            p = feature['properties']
            seniors, residents = p.get('elderly_65plus_2026') or 0, p.get('population_2026') or 0
            if prefix == 'subzone':
                parent = by_area[p['planning_area_id']]
                area_km2 = max(.01, p.get('area_km2') or (parent.get('area_km2') or 0)*residents/max(1,parent.get('population_2026') or 0))
            else:
                area_km2 = p.get('area_km2') or 0
            density = seniors/area_km2 if area_km2 else 0
            share = p.get('elderly_65plus_share_pct') or 0
            old_share = (p.get('elderly_75plus_2026') or 0)/residents*100 if residents else 0
            green = min(100, max(0, greens.get(p['id'], {}).get('greenBaselineScore', 0)))
            for network in networks:
                base_heat = heat_at(p['centroid'], network)
                for shade in (range(0,31,5) if prefix == 'area' else [0]):
                    heat = max(0, base_heat-shade*(.01+.02*(1-green/100)))
                    for extra in (range(7) if prefix == 'area' else [0]):
                        identity = (f'{prefix}:{p["id"]}', f'{heat:.3f}:{extra}')
                        if identity in seen:
                            continue
                        seen.add(identity)
                        access = ((p.get('cooling_centres') or 0)+extra)/seniors*10000 if seniors else 0
                        records.append([heat, density, share, old_share, access])
                        identities.append(identity)
    probabilities = model.predict_proba(np.asarray(records, dtype=np.float64))
    scores = 50*probabilities[:, 1]+100*probabilities[:, 2]
    result = {}
    for (region, key), score in zip(identities, scores):
        result.setdefault(region, {})[key] = round(float(score), 4)
    payload['publishedRisk'] = {
        'kind': 'prediction-results-only',
        'thresholds': model_bundle['thresholds'],
        'generatedAt': payload['generatedAt'],
        'note': 'Private model outputs for the displayed hourly forecasts, observations and supported cooling scenarios. No trees or weights are supplied.',
        'scores': result,
    }
    return len(records)


if __name__ == '__main__':
    target = ROOT/'dist/heatguard-data.json'
    payload = json.loads(target.read_text())
    count = attach_risk_results(payload)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    temporary.replace(target)
    print(f'Exported {count} private-model risk predictions for {len(payload["publishedRisk"]["scores"])} areas/subzones.')

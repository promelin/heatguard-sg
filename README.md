# HeatGuard SG

[HeatGuard SG](https://promelin.github.io/heatguard-sg/) is DeltaNexus's open-source heat-planning dashboard for Singapore. It combines current station heat index, a planning-area Heat Risk Score, cooling-infrastructure scenarios and local micro-environment heat estimates in one map-led interface.

[![HeatGuard SG concept poster](docs/HeatGuard-SG-Concept-Poster.png)](docs/HeatGuard-SG-Concept-Poster.pdf)

**[Open the live dashboard](https://promelin.github.io/heatguard-sg/)** · **[Download the one-page concept poster](docs/HeatGuard-SG-Concept-Poster.pdf)**

## Contributors

**DeltaNexus:** Chen Xuanhong and Zhang Jiaheng, National University of Singapore, Chemistry, 2025 cohort.

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/fromzerotoinfinityplus">
        <img src="https://avatars.githubusercontent.com/u/338661854?v=4" width="88" alt="fromzerotoinfinityplus"><br>
        <sub><b>@fromzerotoinfinityplus</b></sub>
      </a>
    </td>
  </tr>
</table>

Full project credits are recorded in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Implemented modules

- **Live Heat Index** — current heat-index conditions across the weather-station network, with observation and refresh times.
- **Area Heat Risk** — a policy-calibrated random-forest score that combines forecast heat, official 2026 age structure and mapped eldercare access across 55 planning areas.
- **Cooling Simulation** — selectable parks, cycling routes and park connectors, area-specific green baselines, adjustable cooling points and corridor shade.
- **Nature-positive Layout Generator** — a Conditional Spatial VAE that accepts a user-drawn boundary on a zoomable Singapore training-data map and generates 10–1,000 aligned building, blue-green, mobility and community-facility layout candidates for an unseen site. The map includes planning-area navigation for focus only, up to 40× zoom and lazy-loaded exact building footprints; the model remains conditioned on the user's arbitrary drawn boundary.
- **Local Heat Estimate** — a separate spatial model that combines the station-network background with nearby vegetation, HDB building form, coastline exposure and geographic position at 250 m, 500 m or 1 km. The overview covers 55 planning areas and reveals 332 selectable subzones when zoomed in.
- **Community Data** — structured, privacy-minimised collection for green-pocket proposals, build confirmation, monthly condition checks and on-site comfort feedback.

The public site also includes planning-area and subzone demographics, close-zoom HDB building detail, a ranked priority queue, model evidence and the completed DAISI B1 delivery flow.

## Repository layout

```text
backend/          FastAPI service, weather ingestion and model runtime
backend/model/    Forecast, risk and spatial-layout inference weights and context raster
dist/             GitHub Pages website, spatial layers and published forecast
docs/             Concept poster and project visuals
reports/          Held-out validation evidence
scripts/          Forecast publication, model validation and data-build tools
```

## Run locally

Use Python 3.10 or newer. The project was developed in the local `cueq` environment.

```bash
conda activate cueq
python -m pip install -r backend/requirements.txt
cp backend/.env.example .env
python run_live.py
```

Open `http://localhost:8000`. The service exposes `/api/health`, `/api/heatguard-data.json`, `/api/local-heat-index`, `/api/spatial-layout/status`, `/api/spatial-layout/jobs` and `/api/community-submissions`, and serves the website from the same origin. Community records, photos and the fitted local-heat model are private backend runtime data and are excluded from Git.

For layout generation, select a planning area to focus the map or click an area boundary, inspect the real buildings and blue-green context, then draw any site boundary. Choose target dwelling density and footprint coverage and request 10–1,000 candidates. The backend converts the boundary and mapped surroundings into the model's 14 condition channels, samples with an expanded latent range, checks hard exclusions and returns a coverage-aware, spatially diverse subset as crisp native 256 × 256 previews.

## Train the local environment model

The local model learns a station's heat-index difference from the same-hour peer-station background. Nearby parks, HDB footprints and height, building density, coastline proximity, latitude, longitude and CBD distance provide the spatial inputs. Validation holds out complete weather stations rather than randomly splitting repeated hours.

```bash
conda activate cueq
python -m pip install -r backend/requirements.txt
python scripts/train_local_heat_model.py
```

This creates the private, Git-ignored `backend/model/local_heat_model.joblib`, updates the public validation evidence in `reports/local-heat-model-validation.*`, and publishes results for 55 planning areas and 332 subzones to `dist/data/local-heat-results.json`. The released validation uses 2,210 hourly observations across 11 stations and reports a leave-one-station-out MAE of 1.011 °C and RMSE of 1.482 °C.

Example backend request:

```bash
curl -X POST http://localhost:8000/api/local-heat-index \
  -H 'Content-Type: application/json' \
  -d '{"longitude":103.8519,"latitude":1.2903,"radius_m":500,"background_heat_index_c":34}'
```

The background heat condition is required for an absolute heat-index estimate; if omitted, the live backend uses the latest station-network mean. This is spatial downscaling, not a next-day weather forecast. Feature profiles far from the training stations are conservatively shrunk toward the background, and every response includes a research-use notice.

For a website-only preview:

```bash
python -m http.server 8001 --directory dist
```

Open `http://localhost:8001`. HTTP serving enables the lazy-loaded HDB and green-infrastructure layers.

## Reproduce inference

The repository includes both released model artefacts:

- `backend/model/best_model.pt` — V4 history-only full Temporal Fusion Transformer.
- `backend/model/risk_model.joblib` — policy-calibrated planning-area random forest.
- `backend/model/v4_runtime_metadata.json` — feature order and training normalisation used by the V4 runtime.
- `backend/model/spatial_cvae_inference.pt` — compact FP16 Conditional Spatial VAE inference weights.
- `backend/model/spatial_context.tif` — compressed 10 m Singapore context used to condition arbitrary drawn sites.

The new local-environment model artefact is intentionally not released in the public repository. Its reproducible training code, validation report and published prediction results are included.

Install CPU PyTorch and the inference dependencies, then publish a fresh result:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
python scripts/publish_forecast.py
```

`DATA_GOV_SG_API_KEY` is optional and belongs in a local `.env` file or GitHub Actions secret. No credentials are stored in this repository. The scheduled workflow recomputes forecasts from current public observations and commits only the updated result file.

## Data sources

- data.gov.sg / NEA: air temperature, relative humidity and rainfall
- URA: planning-area and subzone boundaries
- SingStat and HDB: 2026 age structure and building context
- MOH: eldercare-service locations
- NParks: [Parks and Nature Reserves](https://data.gov.sg/datasets/d_77d7ec97be83d44f61b85454f844382f/view) and Park Connector Network
- LTA: cycling network

Detailed spatial provenance is recorded in `dist/data/spatial/metadata.json`. HeatGuard SG is designed for planning review with human oversight; public health actions should continue to follow official Singapore guidance.

## License

Released under the [MIT License](LICENSE). Source datasets remain subject to their respective providers' terms.

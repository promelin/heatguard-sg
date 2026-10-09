# HeatGuard SG Conditional Spatial VAE

This module learns the spatial grammar of existing Singapore HDB neighbourhoods and generates multiple nature-positive layout candidates for a conditioned site.

## Representation

Each example is a 2.56 km × 2.56 km tile at 10 m resolution (`256 × 256`). The model receives 14 condition channels:

1. site boundary;
2. editable area;
3. retained buildings;
4. retained water;
5. retained nature;
6. hard exclusion area;
7. surrounding buildings;
8. surrounding roads;
9. surrounding walking and cycling network;
10. surrounding green cover;
11. surrounding water;
12. MP2025 land-use environment;
13. target dwelling density;
14. target HDB footprint coverage.

It generates 13 aligned channels:

1. HDB footprint;
2. HDB height;
3. park green;
4. forest and nature;
5. community green;
6. water and blue-green space;
7. roads;
8. walking network;
9. cycling network;
10. schools;
11. eldercare;
12. community facilities;
13. public open space.

The model is a conditional U-Net VAE. A condition encoder supplies spatial skip features, while a posterior encoder learns a stochastic latent representation from the condition and observed layout. The optimised decoder injects the latent vector at every spatial scale and regularises pairwise sample diversity, so one site condition can yield many distinct plans. Hard exclusions are supplied as conditions and building overlap is explicitly penalised during training.

## Build training tiles

Run this locally in the `cueq` environment after preparing the canonical data:

```bash
conda run -n cueq python -m spatial_cvae.make_dataset \
  --processed-root ../data/processed/sg_layout/v1 \
  --output spatial_cvae/data/training_tiles_v1 \
  --num-samples 1000
```

The output uses memory-mapped `uint8` arrays to keep training I/O predictable.

## Vanda workflow

All computation on Vanda must be submitted through PBS. The login node is only used for file transfer and job submission.

```bash
qsub spatial_cvae/pbs/env_check.pbs
qsub spatial_cvae/pbs/gpu_smoke.pbs
qsub spatial_cvae/pbs/train_sweep.pbs
qsub spatial_cvae/pbs/train_diversity_sweep.pbs
```

The GPU jobs use the maintained `spark_jax` environment on Vanda (PyTorch 2.11 / CUDA 12.8), which supports the A40 nodes. The sweeps compare model width, latent dimension, KL weight, learning rate and diversity regularisation. Runs are split by planning area rather than randomly by tile, reducing spatial leakage.

## Generate candidates

After selecting a checkpoint:

```bash
HEATGUARD_CVAE_CHECKPOINT=/path/to/best.pt \
HEATGUARD_CVAE_SAMPLE_INDEX=0 \
qsub spatial_cvae/pbs/generate_1000.pbs
```

The generator writes 100 or 1,000 candidates as a memory-mapped `uint8` array and removes generated building pixels inside hard-exclusion areas. A separate decision scorer can subsequently rank the candidates for biodiversity, connectivity, thermal comfort, housing capacity and accessibility.

## Arbitrary-site web inference

The website does not choose from 55 preset planning areas. A user zooms the real Singapore training-data map and draws a new polygon of any supported size. The backend projects that boundary to SVY21, reads the surrounding 10 m building, road, walking/cycling, vegetation, water, exclusion and land-use context, and rasterises the new site into the 14 model condition channels. Sites up to 2.56 km across retain the training resolution; larger sites use an adaptive metres-per-pixel extent.

Build the compact context artefacts locally in `cueq`:

```bash
conda run -n cueq python -m spatial_cvae.build_context_raster \
  --processed-root ../data/processed/sg_layout/v1 \
  --output backend/model/spatial_context.tif \
  --metadata backend/model/spatial_context.json

conda run -n cueq python -m spatial_cvae.export_context_preview \
  --context backend/model/spatial_context.tif \
  --output dist/data/spatial/planner-context.png \
  --metadata dist/data/spatial/planner-context.json

conda run -n cueq python -m spatial_cvae.export_planner_vectors \
  --data-root ../data/processed/sg_layout/v1 \
  --output dist/data/spatial
```

The last command creates planning-area-indexed building files so the browser can load exact footprints only for the area currently in view. Planning-area selection is a navigation aid; it never replaces the user-drawn arbitrary boundary used by the model.

The browser map is a visualisation of the same inference context, not a generic basemap. It can therefore reveal real buildings, vegetation, water and mobility networks progressively as the user zooms before drawing a site.

## Verified Vanda run

The 9 October 2026 run trained on 1,000 tiles covering 28 planning areas and selected `inject_b32_z192_d3`. Its minimum planning-area validation loss was `0.73938`. A 1,000-plan Tampines East generation test produced 1,000 unique binarised layouts, mean cross-candidate pixel standard deviation `0.08683`, and zero building overlap with hard-exclusion pixels. The local checkpoint and machine-readable reports are stored under `spatial_cvae/results/vanda_20261009/` (ignored by Git because of checkpoint size).

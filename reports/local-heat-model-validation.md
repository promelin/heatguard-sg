# Local environment heat-index model validation

This model spatially downscales a supplied Singapore-wide background heat index. It does not predict tomorrow's weather.

- Selected model: `ridge`
- Training rows: 2,210
- Weather stations: 11
- Search radius: 500 m
- Leave-one-station-out MAE: 1.011 °C
- Leave-one-station-out RMSE: 1.482 °C
- Leave-one-station-out R²: 0.845

## Inputs

- Background heat index from the station network
- Nearby park-cover proxy
- HDB building footprint, density, floor-area ratio and height proxy
- Coast distance and coastal exposure proxy
- Latitude, longitude and distance to the CBD

## Validation design

Every fold holds out one complete weather station. This prevents repeated hourly observations from the same location appearing in both training and validation.
For feature profiles far from any training station, the local adjustment is conservatively shrunk toward the network background instead of extrapolating without bound.
Published map results evaluate the centroids of 55 planning areas and 332 official subzones at 250 m, 500 m and 1 km analysis radii.

## Limitations

The available labels cover a small station network and a short held-out replay window. Park coverage is a centroid-and-distance proxy; building coverage currently represents mapped HDB footprints; water exposure represents coastline proximity, not all inland water bodies. The result is a research estimate, not an observation, causal cooling effect, weather forecast or official alert.

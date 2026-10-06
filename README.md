# HeatGuard SG — public website

DeltaNexus's DAISI B1 pre-release dashboard: observed station heat index, planning-area risk, cooling-infrastructure scenarios and V4 hourly forecasts for the next 24 hours.

This repository contains the website, public spatial data and published prediction results only. Model weights and inference code remain in a separate private repository.

Private scheduled inference reads data.gov.sg weather observations, constructs 168 hours of history and runs the V4 model. It also computes area and subzone risk estimates and the supported cooling scenarios privately. The resulting JSON is published here, then GitHub Pages redeploys automatically. Page loads do not trigger a separate model run; visitors explore the latest computed forecast. The browser checks for updates every five minutes.

Forecasts are scheduled hourly but may be delayed by upstream APIs, execution quotas or the scheduler. The site displays generation/observation times and warns when a forecast is stale. A failed computation leaves the last successful prediction intact.

The risk classifier follows a planning policy, not measured health outcomes. Cooling scenarios are planning assumptions, not validated causal cooling effects. HDB building population values are estimates. This is not an official Singapore heat alert.

Data sources: data.gov.sg / NEA weather; SingStat / HDB demographics; URA planning boundaries and supplied building footprints; NParks parks, reserves and Park Connector Network; LTA cycling network. See spatial metadata and the dashboard's evidence view for provenance.

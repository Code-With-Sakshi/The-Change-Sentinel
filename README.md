# The Change Sentinel

**Watch a district from orbit and produce case files, not curiosities.**

## Overview

The Change Sentinel is a prototype for satellite-based land-change screening, focused on identifying potential quarry-like surface changes using before-and-after satellite imagery.

The project explores how freely available satellite data can help highlight areas that may require further investigation. It provides a Streamlit interface for viewing satellite imagery and screening potential changes.

The current implementation focuses on a specific use case rather than attempting to detect every type of land-use change.

## Problem Statement

Unauthorised quarrying and other land-surface changes can be visible in satellite imagery before they are reported on the ground. However, manually comparing images is time-consuming, and simple change detection can produce false alarms from agriculture, river sediment, water, shadows, and other natural or seasonal changes.

The goal of The Change Sentinel is to explore a lightweight, accessible approach to screening satellite imagery for potential quarry-related changes and supporting further investigation.

## What We Implemented

- **Satellite image comparison:** Uses before-and-after imagery for the Bhedi Kharka study area.
- **Quarry-focused screening:** Focuses on potential quarry-like surface changes rather than attempting to detect every change category.
- **Streamlit interface:** Provides a web interface for running and viewing the screening workflow.
- **Fixed screening threshold:** Uses a configured threshold of `0.60` rather than exposing a user-adjustable threshold slider.
- **Reference examples:** Includes KMZ reference files for a known quarry and non-quarry examples representing agriculture, river sediment, roads, and water.
- **Lightweight project structure:** Separates the Streamlit application, model logic, dependencies, and sample data.

## Technology Stack

- Python
- Streamlit
- Raster satellite imagery in GeoTIFF format
- KMZ reference data
- The Python libraries listed in `requirements.txt`

## Study Area and Data

The current prototype uses satellite imagery associated with the Bhedi Kharka study area, including before-and-after scenes dated January 24, 2020, and December 22, 2021.

The repository includes sample satellite rasters and KMZ reference examples. These provide inputs for the current screening workflow; they do not, by themselves, establish that the system has been validated across an entire district.

## What Is Not Yet Implemented or Validated

The original problem statement calls for a pipeline that filters misleading changes, identifies actionable unauthorised activity, and produces evidence-backed case files. The current prototype does not yet demonstrate the complete end-to-end system.

The following capabilities remain incomplete or unverified:

- **Continuous district monitoring:** Automated, recurring scans of an entire district are not yet demonstrated.
- **Robust cloud and shadow filtering:** Reliable removal of clouds and shadows has not yet been established as a complete part of the workflow.
- **Seasonal normalisation:** The system has not yet demonstrated robust separation of quarry activity from seasonal agriculture, monsoon effects, or other changing land conditions.
- **Verified detection accuracy:** Comprehensive evaluation against independently verified quarry and non-quarry locations has not yet been established.
- **Complete case files:** A fully automated evidence package containing all required fields is not yet demonstrated.
- **Area and jurisdiction verification:** Reliable area estimation and automatic assignment to administrative boundaries have not yet been established as complete capabilities.
- **Officer-oriented prioritisation:** A validated ranking system that consistently places the most important cases first has not yet been demonstrated.
- **Unauthorised activity verification:** Satellite-based screening alone cannot establish whether an activity is illegal. Further verification using relevant records and field investigation would be required.

## Future Work

The next stage of development would focus on:

1. Adding robust cloud and shadow masking and appropriate seasonal comparisons.
2. Evaluating detections against labelled, independently verified examples.
3. Estimating the area of each candidate change.
4. Integrating open administrative boundaries to identify the relevant jurisdiction.
5. Generating structured case files with before-and-after imagery, dates, confidence information, area estimates, and supporting evidence.
6. Developing and evaluating a prioritisation method for reviewing candidate detections.
7. Extending the workflow to repeated monitoring of a defined district while keeping compute and data requirements manageable.

## Responsible Use

The Change Sentinel is a preliminary screening prototype. Its outputs should be treated as potential changes requiring review, not as confirmed illegal quarrying or proof of a violation. Any enforcement decision should rely on appropriate corroborating evidence and authorised investigation.

## Project Status

**Prototype — quarry-focused satellite change screening.**

The project demonstrates an initial workflow for exploring potential quarry-related surface changes. Completing and validating the full monitoring and evidence-backed case-file pipeline described in the problem statement remains future work.

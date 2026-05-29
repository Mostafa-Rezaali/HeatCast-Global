# Project Summary

## Overview

Extreme heat and rapidly developing drought are among the most damaging hazards in the United
States, yet their week-3 predictability remains limited because local land-atmosphere feedbacks
interact with remote teleconnection patterns and global circulation anomalies. This fellowship will
develop a probabilistic, physics-guided graph neural forecasting framework for 15-day prediction of
CONUS daily maximum temperature anomalies and associated flash-drought risk. The project builds from
my MeshFlowNet/conditional-flow-matching model, a GraphCast-style icosahedral mesh encoder-
processor-decoder trained on PRISM, ERA5, teleconnection indices, topography, seasonal radiation,
soil-moisture, and global circulation fields. The proposed host organization is University of
Florida, Department of Geography. The proposed scientific mentors are David Keellings, University of
Florida.

## Intellectual Merit

The project asks: how much week-3 predictability of U.S. heat extremes is contained in global
circulation, ocean-atmosphere teleconnections, soil-moisture memory, and local persistence, and can
conditional flow models extract this signal with calibrated uncertainty? Three aims will: (1) extend
a deterministic 15-day residual MeshFlowNet into an ensemble conditional-flow model; (2) diagnose
regional and teleconnection-dependent skill using 5-fold leave-years-out hindcasts for 1981-2023;
and (3) benchmark against persistence, climatology, ridge/teleconnection baselines, and operational
S2S reference skill. Preliminary hindcasts over 5,332 MJJAS samples show stitched temporal anomaly
correlation of 0.1016 versus 0.0735 for persistence, motivating a focused investigation of where and
why learned graph dynamics add value.

## Broader Impacts

Improved week-3 heat-risk guidance can support public-health planning, water resources management,
energy operations, and agricultural preparedness. The fellowship will produce open, reproducible
workflows; publish model diagnostics and derived hindcast products with metadata; and translate the
research into teaching modules on trustworthy AI for climate extremes. I will mentor undergraduate
and REU students in climate-data science, with emphasis on transparent evaluation, uncertainty, and
equitable access to AI tools for environmental risk assessment.


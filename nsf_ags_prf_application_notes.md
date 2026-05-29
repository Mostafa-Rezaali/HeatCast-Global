# NSF AGS-PRF Application Prep Notes

Source context: Mostafa_Rezaali_CV.pdf, local GraphCast/MeshFlowNet repository, NSF 22-639 AGS-PRF solicitation.

## Applicant

Name: Mostafa Rezaali

Location/status from CV: Gainesville, Florida; lawful permanent resident of the United States.

Current affiliation: Ph.D. candidate in Climate Science, Geography, University of Florida.

Education:
- Ph.D. candidate, Climate Science/Geography, University of Florida, GPA 3.94/4.0, Aug 2022-present.
- Graduate Certificate in Atmospheric Sciences, University of Florida.
- M.Sc., Civil and Environmental Engineering, Qom University of Technology, GPA 4.0/4.0, Sep 2016-Sep 2018.
- B.Sc., Civil and Environmental Engineering, IAUKHSH, Sep 2011-Feb 2016.

Research identity:
- AI for extreme weather, especially heat waves and flash drought.
- Deep learning for spatiotemporal climate and atmospheric-pollution forecasting.
- Strong prior publication record in environmental modeling, hydrology, air quality, and climate/health risk.

Selected metrics from CV:
- H-index: 10.
- Total citations: 402 as of December 2025.
- Lead-authored publications include Urban Climate 2024 ozone forecasting, Journal of Hydrology 2021 probabilistic water demand forecasting, and multiple environmental modeling papers.

Mentoring/teaching:
- Invited lecturer, "Deep Learning Applications in Climate Science," University of Florida, April 2025.
- Invited lecturer, climate change impacts on future plant distributions, University of Florida, Spring 2025.
- NSF LEAP REU mentor, May-August 2025.
- M.Sc. student advising experience.

## NSF AGS-PRF Fit

Most relevant AGS areas:
- Climate and Large-Scale Dynamics (CLD).
- Physical and Dynamic Meteorology (PDM).

Eligibility facts to verify before submission:
- U.S. lawful permanent resident at submission.
- Current graduate student or Ph.D. within the solicitation timing window.
- Fellowship must begin within 6 months of award notification if selected.
- Only one AGS-PRF proposal may be submitted per individual per year.

## Candidate Project Framing

Working title options:
- Postdoctoral Fellowship: AGS-PRF: Teleconnection-Aware Graph Flow Models for Week-3 Extreme Heat Prediction
- Postdoctoral Fellowship: AGS-PRF: Learning Subseasonal Extreme Heat Predictability with Conditional Flow Models
- Postdoctoral Fellowship: AGS-PRF: Physics-Guided Graph Neural Forecasting of 15-Day U.S. Heat Extremes

One-sentence project concept:
Develop and evaluate a physics-guided, teleconnection-aware graph neural forecasting system that uses conditional flow modeling and global atmospheric context to improve 15-day prediction of U.S. extreme heat anomalies.

Core science question:
How much predictable signal for week-3 U.S. extreme heat is contained in global circulation, teleconnection, land-surface, and local persistence fields, and can graph-based conditional flow models extract that signal with calibrated uncertainty?

Technical approach from local repository:
- Model family: MeshFlowNet, a GraphCast-style icosahedral mesh encoder-processor-decoder GNN.
- Modes supported: deterministic GraphCast-style residual prediction and probabilistic GenCast/conditional-flow-matching mode.
- Forecast target: PRISM daily maximum 2-meter temperature anomaly at lead time +15 days over CONUS.
- Forecast domain: CONUS, 0.04-degree PRISM grid, 621 x 1405.
- Period: MJJAS, 1981-2023.
- Cross-validation: 5-fold leave-k-years-out split covering 43 years.
- Local inputs: current and lagged PRISM T2m, geopotential, soil moisture, sea-level pressure, 2m temperature, 850-hPa humidity/temperature/winds, 300-hPa geopotential, topography, latitude, longitude, day-of-year sine/cosine, top-of-atmosphere insolation, land mask.
- Vector conditioning: 5 teleconnection indices.
- Global context: 59 ERA5/coarse variables including SST, OLR, winds, geopotential, temperature, humidity, sea-level pressure, surface pressure, TCWV, and multilevel atmospheric fields.
- Output strategy: direct 15-day prediction with a persistence residual head.
- Evaluation metrics: TAC, R2, anomaly R2, MAE/CRPS, persistence and climatology baselines, bootstrap significance.

Current model results from review packet:
- Samples: 5,332.
- Years: 1981-2023.
- Stitched model TAC: 0.1016.
- Persistence TAC: 0.0735.
- TAC improvement over persistence: +0.0281.
- Per-fold validation R2: about 0.58-0.60.
- Per-fold best validation TAC: 0.0504-0.1450.

## NSF Required Content Reminders

Project Summary:
- Maximum 1 page.
- Must include overview, separate Intellectual Merit statement, and separate Broader Impacts statement.
- Must identify proposed scientific mentor(s).
- Must identify proposed host organization(s).

Project Description:
- Maximum 10 single-spaced pages including figures/tables.
- Must include detailed research plan.
- Must include clearly delineated Intellectual Merit and Broader Impacts sections.
- Must justify host institution and mentor choice.
- Must relate proposed work to host research/education efforts and facilities.
- Must describe long-term career goals and how the fellowship advances them.
- Must include a separate section header exactly labeled "Broader Impacts" on its own line.

Supplementary documents:
- Data Management Plan, maximum 2 pages.
- Host institution letter(s), one per host institution, signed by mentor and department chair/equivalent.
- Mentor letter must certify the proposal has been read and approved, adequate facilities/support will be provided, and include mentoring plan.
- Collaborators and Other Affiliations information is required for the mentor(s).

Budget:
- Research.gov should pre-populate AGS-PRF stipend and fellowship allowance.
- Budget justification must include spending plan for fellowship allowance.

## Unknowns Needed From User Or Portal

- Proposed host institution.
- Proposed scientific mentor(s), with NSF ID or email.
- Requested fellowship start date.
- Final proposal title.
- Whether any similar proposals/applications are current or pending and must be disclosed.
- Whether the project will remain at University of Florida or move to a new host institution; if current institution, justification must address breadth/new connections.
- Uploaded files status: project summary, project description, references cited, biosketch, current and pending support, COA, DMP, host letter, budget justification, facilities/equipment/other resources.

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "nsf_ags_prf_uploads" / "final_format"
UPLOAD = ROOT / "nsf_ags_prf_uploads"
OUT.mkdir(parents=True, exist_ok=True)

HOST = "University of Florida, Department of Geography"
SPONSOR = "David Keellings, University of Florida"
TITLE = (
    "Postdoctoral Fellowship: AGS-PRF: Probabilistic Prediction of Heat Waves "
    "and Flash Droughts Using Physics-Informed Conditional Flow Matching on an "
    "Icosahedral Mesh"
)


PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\usepackage{enumitem}
\usepackage{array}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}

\setstretch{1.03}
\pagestyle{empty}
\setlength{\parindent}{15pt}
\setlength{\parskip}{4pt}
\setlist{nosep,leftmargin=0.24in}
\titlespacing*{\section}{0pt}{0.35em}{0.1em}
\titlespacing*{\subsection}{0pt}{0.25em}{0.08em}
\titleformat{\section}{\normalfont\bfseries\large}{}{0pt}{}
\titleformat{\subsection}{\normalfont\bfseries}{}{0pt}{}

\begin{document}
"""

ENDING = "\n\\end{document}\n"


def clean_tex(text: str) -> str:
    return (
        text.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def doc_title(title: str) -> str:
    return rf"""\begin{{center}}
{{\Large\bfseries {title}}}\\[0.05in]
{{\normalsize Mostafa Rezaali}}\\
{{\normalsize NSF AGS-PRF Application}}
\end{{center}}
\vspace{{-0.08in}}
"""


def compile_doc(stem: str, title: str, body: str) -> Path:
    tex_path = OUT / f"{stem}.tex"
    tex_path.write_text(PREAMBLE + doc_title(title) + body.strip() + ENDING, encoding="utf-8")
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    subprocess.run(cmd, cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return tex_path.with_suffix(".pdf")


PROJECT_SUMMARY = rf"""
\section*{{Overview}}
Extreme heat and rapidly developing drought are among the most damaging climate hazards in the United States, yet their week-3 predictability remains limited because local land-atmosphere feedbacks interact with global circulation anomalies, teleconnection state, and soil-moisture memory. This fellowship will develop a probabilistic, physics-informed GeoAI framework for 15-day prediction of CONUS daily maximum temperature anomalies and flash-drought-relevant heat risk. The project builds from my conditional-flow-matching (CFM) model, a GraphCast-style icosahedral mesh network trained with PRISM, ERA5, teleconnection indices, topography, seasonal radiation, soil moisture, and global atmospheric context fields. The proposed host institution is {HOST}. The proposed sponsoring scientist is {SPONSOR}.

\section*{{Intellectual Merit}}
The project asks how much week-3 predictability of U.S. heat extremes is contained in global circulation, land-surface memory, local persistence, and teleconnection state, and whether CFM can extract that signal with calibrated uncertainty. Three aims will: (1) extend deterministic MeshFlowNet into a probabilistic CFM ensemble; (2) diagnose regional and regime-dependent skill using leave-years-out hindcasts for 1981--2023; and (3) evaluate heat-wave and flash-drought-relevant exceedance probabilities against persistence, climatology, and teleconnection baselines. Preliminary hindcasts over 5,332 MJJAS samples show stitched temporal anomaly correlation of 0.1016 versus 0.0735 for persistence, motivating a focused study of where graph-based AI adds physically meaningful subseasonal forecast value.

\section*{{Broader Impacts}}
Improved week-3 heat-risk guidance can support public-health planning, water resources, energy operations, and agricultural preparedness. The fellowship will produce reproducible workflows, hindcast diagnostics, teaching modules on trustworthy AI for climate extremes, and mentoring opportunities for students entering climate-data science from geography, environmental engineering, and data science. The broader goal is transparent, uncertainty-aware climate AI that helps communities interpret and prepare for extreme heat and drought risk while making model limitations clear.
"""


PROJECT_DESCRIPTION = rf"""
\section*{{Project Title}}
\textbf{{{TITLE}}}

\section*{{Motivation and Central Hypothesis}}
Heat waves and flash droughts are compound hydroclimate hazards with large consequences for public health, energy demand, water resources, ecosystems, and agriculture. Their impacts often emerge on decision horizons of two to four weeks, when emergency managers, utilities, farmers, and public-health agencies still have time to act, but when deterministic forecast skill is often weak. Predictability at these leads is not purely local. It reflects slowly evolving ocean-atmosphere patterns, stationary waves, soil-moisture memory, radiative seasonality, synoptic persistence, and land-surface feedbacks. My central hypothesis is that a graph-based conditional flow model, trained directly on high-resolution U.S. temperature anomalies while conditioned on global circulation and teleconnection states, can reveal and exploit a measurable but spatially heterogeneous week-3 signal for extreme heat and flash-drought precursors.

\section*{{Preliminary Work and Model Foundation}}
I have implemented MeshFlowNet, a GraphCast-style icosahedral mesh encoder-processor-decoder for direct 15-day CONUS daily maximum temperature prediction. The model uses current and lagged local PRISM temperature anomalies, atmospheric and land-surface fields, topography, latitude, longitude, seasonal insolation, land mask, five teleconnection indices, and a 59-channel global ERA5/coarse atmospheric context stack. It supports deterministic residual prediction and probabilistic conditional-flow-matching modes. The current deterministic five-fold leave-years-out hindcast covers MJJAS 1981--2023, with each year predicted by a model that did not train on that year. Across 5,332 samples, stitched model temporal anomaly correlation is 0.1016 compared with 0.0735 for persistence, a +0.0281 improvement.

\begin{{center}}
\includegraphics[width=0.96\textwidth]{{figures/figure_preliminary_prediction.png}}\\[-0.04in]
\footnotesize\textbf{{Figure 1.}} Example 15-day hindcast from the preliminary MeshFlowNet system, showing verifying anomaly, direct model prediction, and prediction-minus-truth error over CONUS.
\end{{center}}

These preliminary results are not presented as an operational system. They show that the model extracts nontrivial anomaly signal at a difficult lead time and justify a systematic postdoctoral investigation of predictability, uncertainty, and physical interpretation. The fellowship will convert this foundation into a calibrated probabilistic framework and a documented scientific benchmark for week-3 heat and flash-drought-relevant risk.

\section*{{Aim 1: Build a Calibrated Conditional-Flow Forecast System}}
The first aim will convert the current deterministic residual model into a probabilistic forecast system. The deterministic branch predicts $y(t+15)$ as persistence plus a learned residual in one forward pass. The conditional-flow branch will learn a velocity field from initialization-time state to the verifying anomaly field, conditioned on teleconnection indices and global atmosphere/ocean fields. Ensembles will be generated by sampling the flow and by training multiple seeds per cross-validation fold. Calibration will be assessed with continuous ranked probability score, reliability diagrams for exceedance of local 90th and 95th percentiles, rank histograms, Brier score, and spatially stratified spread-skill relationships.

This aim will test whether conditional flow matching can deliver useful uncertainty rather than only sharper deterministic maps. A successful result will be a reproducible probabilistic benchmark for week-3 U.S. heat prediction. If the CFM ensembles are underdispersed, I will test temperature scaling, ensemble perturbations, and loss weighting for extreme cases. If the ensembles are sharp but poorly calibrated, I will diagnose whether the problem is regional, seasonal, or associated with specific circulation regimes.

\section*{{Aim 2: Diagnose Sources of Subseasonal Predictability}}
The second aim will use controlled ablations and interpretable diagnostics to determine which features carry week-3 skill. I will compare models trained with: local persistence only; local meteorology plus static and seasonal terms; teleconnection indices; global ERA5 fields; and the full conditioning stack. The analysis will produce per-pixel temporal anomaly correlation, skill over persistence, regional summaries for NOAA climate regions, block-bootstrap significance tests by year, and event composites for strong teleconnection regimes.

The scientific target is not only a higher metric, but a physical account of where AI forecast skill is consistent with known pathways such as Pacific-North American wave trains, moisture transport, soil-moisture feedback, subtropical ridge variability, and persistent high-pressure anomalies. These diagnostics will help distinguish robust predictive signal from overfitting or spatially smooth climatological reconstruction.

\section*{{Aim 3: Connect Heat Prediction to Flash-Drought-Relevant Risk}}
The third aim will connect temperature-anomaly prediction to flash-drought precursors by evaluating rapid intensification of heat and atmospheric demand in conjunction with antecedent soil-moisture state. The model already ingests soil moisture, humidity, winds, sea-level pressure, radiation, and large-scale circulation fields. I will evaluate whether the learned representations anticipate co-occurring hot, dry, high-demand regimes over agriculturally important regions. Products will be reported as calibrated probabilistic exceedance maps, regional time series, and case studies rather than deterministic binary declarations.

\section*{{Data and Model Architecture}}
The hindcast system will use PRISM daily maximum temperature over CONUS at 0.04-degree resolution, ERA5 atmospheric variables and global context fields, teleconnection indices, topography, latitude, longitude, land mask, top-of-atmosphere insolation, and seasonal encodings. The current local input stack contains current and lagged PRISM temperature fields; geopotential, soil moisture, sea-level pressure, 2-m temperature, 850-hPa humidity, 850-hPa temperature and winds, 300-hPa geopotential; topography; latitude and longitude; day-of-year sine and cosine; insolation; and land mask. The global context stack contains SST, OLR, winds, geopotential, temperature, humidity, surface pressure, sea-level pressure, and total column water vapor across multiple levels.

\begin{{center}}
\includegraphics[width=0.86\textwidth]{{figures/figure_study_area_conus.png}}\\[-0.04in]
\footnotesize\textbf{{Figure 2.}} Proposed study domain over the contiguous United States for model training, validation, regional diagnostics, and heat-risk evaluation.
\end{{center}}

MeshFlowNet maps the CONUS grid to an icosahedral mesh, performs message passing with graph interaction networks and conditioning, and decodes back to the land grid. This representation preserves global-to-regional context while avoiding purely convolutional locality. The probabilistic branch will use continuous-time embeddings and conditional flow matching; the deterministic branch will remain as a transparent baseline and an efficient ablation.

\section*{{Evaluation and Validation}}
All claims will be based on leave-years-out cross-validation over 1981--2023. Metrics will include temporal anomaly correlation, spatial anomaly $R^2$, MAE, MSE skill relative to persistence, CRPS, reliability of extreme exceedance probabilities, and block-bootstrap confidence intervals. Baselines will include climatology, persistence, ridge regression on teleconnection/global summary features, and published operational S2S reference skill where metric and domain are comparable. I will report negative or regionally limited results explicitly because understanding where AI models fail is essential for trustworthy climate-risk use.

\section*{{Detailed Research Activity Plan}}
Months 1--3 will focus on data audit, cross-validation lock-in, and reproducible baseline runs. Months 4--6 will finalize deterministic baselines and regional skill maps. Months 7--12 will train probabilistic CFM ensembles and complete CRPS, reliability, and spread-skill evaluation. Months 13--16 will complete ablations and teleconnection-regime diagnostics. Months 17--20 will connect heat prediction to flash-drought-relevant risk products. Months 21--24 will finalize manuscript submissions, public documentation, derived product release, and conference dissemination.

\section*{{Justification for Host Institution and Sponsoring Scientist}}
The proposed host institution is {HOST}. The host setting is scientifically appropriate because it provides deep expertise in climatology, climate extremes, hydroclimatology, geospatial analysis, and environmental prediction. The proposed sponsoring scientist is {SPONSOR}, whose expertise in heat waves, climate extremes, and applied climatology directly aligns with the proposed work. This environment will provide disciplinary grounding for interpreting AI forecast skill, evaluating physical consistency, and translating probabilistic predictions into climate-risk knowledge.

The fellowship will support a transition from doctoral model development to an independent research program in probabilistic subseasonal predictability, uncertainty quantification, and open forecast diagnostics. The sponsoring scientist will support regular research meetings, manuscript planning, proposal-development feedback, conference preparation, mentoring opportunities, and career-development milestones.

\section*{{Facilities and Resources}}
The project will use research computing resources, Python/PyTorch software environments, geospatial and climate-data analysis tools, and publicly available PRISM/ERA5-derived data workflows. The local codebase already runs on GPU-equipped computing systems and includes scripts for training, hindcast export, bootstrap statistics, baseline comparison, temporal anomaly correlation maps, sample maps, and CRPS/reliability analysis. These resources are sufficient for the proposed two-year work plan.

\section*{{Intellectual Merit}}
The intellectual merit is the integration of subseasonal predictability theory, graph neural weather models, and probabilistic flow matching. The project will advance knowledge by quantifying when global and land-surface information improves 15-day U.S. heat prediction beyond persistence; producing spatially explicit diagnostics of learned teleconnection skill; and testing whether conditional-flow models can provide calibrated uncertainty for regional extreme-temperature risk. The outcome will be a physically evaluated AI framework rather than a black-box score table.

\section*{{Broader Impacts}}
The broader impacts center on usable, transparent climate-risk intelligence and training. First, the project will produce reproducible code, derived hindcast diagnostics, and documentation suitable for students and practitioners who need to understand AI forecast limitations. Second, I will develop a short teaching module on graph neural weather prediction and uncertainty for graduate climate modeling or environmental data-science courses. Third, I will mentor undergraduate and REU students in open climate-data workflows, with attention to students entering atmospheric science from geography, environmental engineering, and data science. Fourth, the forecast diagnostics will be framed around heat-health, water, energy, and agriculture use cases where calibrated uncertainty can be more valuable than a single deterministic forecast.

\section*{{Expected Products and Success Criteria}}
Success will be assessed by scientific and broader-impact outputs: a documented cross-validated hindcast benchmark with uncertainty metrics; at least two manuscripts, one on model and predictability and one on physical-regime interpretation or flash-drought-relevant risk; public release of code, metadata, and nonrestricted derived products; student mentoring and teaching materials; and a clear professional transition from doctoral model development to an independent research program in AI-enabled atmospheric predictability.

\newpage
\section*{{Detailed Methodological Plan: Data Assembly and Quality Control}}
The first technical phase will establish a locked, auditable data pipeline. PRISM daily maximum temperature fields will be transformed into anomalies using a consistent climatological baseline and land mask. ERA5 and related atmospheric fields will be regridded or sampled consistently with the model architecture, with metadata documenting variable names, pressure levels, units, time stamps, and preprocessing choices. Teleconnection indices will be standardized using training-period statistics within each cross-validation fold to avoid leakage. Static fields, seasonal encodings, and top-of-atmosphere radiation will be versioned with the model configuration.

Quality control will include checks for missing days, temporal misalignment, duplicated files, unit inconsistency, grid mismatch, and accidental use of validation-year information during training. I will produce summary tables for sample counts by year and fold, and maps showing valid-grid coverage and climatological variance. These checks are necessary because small data-leakage or alignment errors can produce misleading gains in subseasonal AI forecasting.

\begin{{center}}\scriptsize
\textbf{{Table 1. Core predictor groups and quality-control checks.}}\\[0.03in]
\begin{{tabularx}}{{\textwidth}}{{p{{0.18\textwidth}}p{{0.34\textwidth}}p{{0.17\textwidth}}X}}
\toprule
\textbf{{Group}} & \textbf{{Variables / fields}} & \textbf{{Role}} & \textbf{{Quality-control checks}}\\
\midrule
PRISM local state & Current and lagged daily maximum temperature anomalies & Persistence memory and local anomaly structure & Missing days, land-mask consistency, anomaly baseline, no validation-year leakage\\
Atmospheric state & Geopotential, sea-level pressure, 2-m temperature, 850-hPa humidity, 850-hPa temperature/winds, 300-hPa geopotential & Synoptic and regional forcing & Unit checks, time alignment, pressure-level consistency, grid registration\\
Land-surface memory & Soil moisture, topography, land mask, latitude, longitude & Heat amplification and spatial heterogeneity & Static-field versioning, mask agreement, terrain-grid consistency\\
Seasonal forcing & Day-of-year sine/cosine, top-of-atmosphere insolation & Annual cycle and radiative context & Leap-day handling, fold-specific normalization, temporal indexing\\
Remote context & SST, OLR, winds, humidity, temperature, pressure, TCWV, teleconnection indices & Global circulation and teleconnection predictors & Training-fold standardization, field completeness, index-source audit\\
\bottomrule
\end{{tabularx}}
\end{{center}}

\newpage
\section*{{Detailed Methodological Plan: Baselines and Ablations}}
The second technical phase will build baselines that are scientifically interpretable. Persistence and climatology will define minimum forecast value. Ridge or regularized linear models using teleconnection and global-summary predictors will test whether nonlinear graph structure is necessary. Local-only and global-context ablations will separate skill from local memory versus remote atmospheric drivers. Static-field and seasonality ablations will test whether the model is relying on physically meaningful anomaly evolution rather than geographic memorization.

Each ablation will be evaluated with the same cross-validation years, same verification masks, and same statistical tests. Results will be summarized as regional skill tables, per-pixel maps, and event-conditioned diagnostics. This structure will make it possible to identify whether CFM improves forecast uncertainty, deterministic anomaly reconstruction, or both.

\begin{{center}}\scriptsize
\textbf{{Table 2. Selected training and validation metrics from \texttt{{cvfold2\_pub\_weekly7}}.}}\\[0.03in]
\begin{{tabular}}{{rrrrrrrrr}}
\toprule
\textbf{{Epoch}} & \textbf{{Train}} & \textbf{{RMSE}} & \textbf{{R2}} & \textbf{{TAC}} & \textbf{{Wk7 TAC}} & \textbf{{Wk7 pers.}} & \textbf{{Spatial R2}} & \textbf{{MAE}}\\
\midrule
1 & 0.398 & 0.894 & 0.183 & 0.042 & 0.112 & 0.107 & 0.208 & 0.676\\
2 & 0.283 & 0.806 & 0.337 & 0.041 & 0.111 & 0.107 & 0.330 & 0.611\\
3 & 0.234 & 0.728 & 0.459 & 0.042 & 0.103 & 0.107 & 0.444 & 0.553\\
4 & 0.223 & 0.700 & 0.499 & 0.044 & 0.094 & 0.107 & 0.494 & 0.530\\
5 & 0.218 & 0.691 & 0.513 & 0.044 & 0.087 & 0.107 & 0.513 & 0.521\\
6 & 0.219 & 0.684 & 0.522 & 0.046 & 0.084 & 0.107 & 0.523 & 0.515\\
7 & 0.207 & 0.678 & 0.531 & 0.052 & 0.089 & 0.107 & 0.531 & 0.511\\
8 & 0.207 & 0.674 & 0.536 & 0.057 & 0.091 & 0.107 & 0.535 & 0.508\\
9 & 0.207 & 0.672 & 0.539 & 0.061 & 0.094 & 0.107 & 0.537 & 0.507\\
10 & 0.182 & 0.670 & 0.541 & 0.067 & 0.102 & 0.107 & 0.539 & 0.506\\
11 & 0.189 & 0.670 & 0.541 & 0.070 & 0.103 & 0.107 & 0.539 & 0.506\\
12 & 0.198 & 0.671 & 0.540 & 0.072 & 0.103 & 0.107 & 0.537 & 0.507\\
13 & 0.186 & 0.673 & 0.538 & 0.076 & 0.106 & 0.107 & 0.535 & 0.507\\
14 & 0.180 & 0.676 & 0.533 & 0.077 & 0.106 & 0.107 & 0.531 & 0.509\\
15 & 0.168 & 0.679 & 0.529 & 0.076 & 0.107 & 0.107 & 0.528 & 0.511\\
16 & 0.177 & 0.683 & 0.523 & 0.075 & 0.103 & 0.107 & 0.522 & 0.515\\
\bottomrule
\end{{tabular}}\\[0.03in]
\footnotesize Values show rapid reduction in RMSE and MAE, improved validation R2, increasing full-period TAC, and weekly-7 TAC compared with the persistence reference.
\end{{center}}

\newpage
\section*{{Detailed Methodological Plan: Conditional Flow Matching}}
The CFM component will represent forecast uncertainty by learning a conditional transformation from a simple distribution to verifying temperature-anomaly fields. The conditioning variables will include current state, lagged temperature, land-surface memory, global circulation, and teleconnection state. I will evaluate multiple sampling strategies and training seeds to separate aleatoric forecast uncertainty from model-training variability.

The probabilistic output will be evaluated using distributional metrics rather than only ensemble mean performance. CRPS, reliability, spread-skill consistency, rank histograms, and exceedance-probability calibration will determine whether the forecast probabilities are useful. If the model produces visually plausible but statistically miscalibrated ensembles, calibration diagnostics will guide retraining, loss weighting, or post-processing.

\begin{{center}}\scriptsize
\textbf{{Table 3. Model components and expected scientific contribution.}}\\[0.03in]
\begin{{tabularx}}{{\textwidth}}{{p{{0.22\textwidth}}p{{0.32\textwidth}}p{{0.18\textwidth}}X}}
\toprule
\textbf{{Component}} & \textbf{{Implementation}} & \textbf{{Output}} & \textbf{{Scientific purpose}}\\
\midrule
Grid-to-mesh encoder & Maps PRISM/ERA5 fields onto an icosahedral mesh & Latent graph state & Allows regional predictions to ingest nonlocal atmospheric context\\
Graph processor & Repeated message passing with conditioning & Updated latent dynamics & Tests whether teleconnection information improves week-3 heat skill\\
Mesh-to-grid decoder & Converts graph state back to CONUS grid & 15-day anomaly field & Produces spatially explicit heat-risk diagnostics\\
Residual head & Persistence plus learned correction & Deterministic benchmark & Separates learned anomaly evolution from simple persistence\\
CFM head & Conditional velocity field and ensemble sampling & Predictive distribution & Evaluates calibrated uncertainty and exceedance probabilities\\
\bottomrule
\end{{tabularx}}
\end{{center}}

\begin{{center}}\scriptsize
\textbf{{Table 4. Verification metrics and interpretation.}}\\[0.03in]
\begin{{tabularx}}{{\textwidth}}{{p{{0.22\textwidth}}p{{0.18\textwidth}}X}}
\toprule
\textbf{{Metric}} & \textbf{{Target}} & \textbf{{Interpretation}}\\
\midrule
Temporal anomaly correlation & Deterministic skill & Tracks daily anomaly evolution at each location\\
Spatial anomaly R2 & Map fidelity & Measures predicted spatial pattern agreement\\
CRPS & Probabilistic skill & Scores full predictive distributions\\
Reliability / rank histograms & Calibration & Tests whether probabilities can be interpreted honestly\\
MSE skill vs. persistence & Practical value & Tests added value beyond a low-cost baseline\\
\bottomrule
\end{{tabularx}}
\end{{center}}

\newpage
\section*{{Detailed Methodological Plan: Physical Interpretation}}
A central risk in AI weather and climate prediction is that high-dimensional models can appear skillful without revealing why. To address this, the project will connect skill diagnostics to physically interpretable regimes. Composite analysis will examine years and periods with strong teleconnection signals, persistent ridging, dry soil-moisture states, and large regional heat anomalies. Regional skill will be compared across humid, arid, coastal, and continental climates, and across periods with different background circulation.

Interpretation will focus on whether skill improvements align with plausible atmospheric pathways. For example, if the model improves over persistence during certain Pacific-North American or subtropical ridge regimes, the analysis will examine whether global-context inputs are essential in those cases. If skill is limited to regions dominated by persistence, the project will state that limitation clearly.

\begin{{center}}\scriptsize
\textbf{{Table 5. Physical interpretation matrix for regime-dependent skill.}}\\[0.03in]
\begin{{tabularx}}{{\textwidth}}{{p{{0.20\textwidth}}p{{0.27\textwidth}}p{{0.24\textwidth}}X}}
\toprule
\textbf{{Diagnostic}} & \textbf{{Comparison}} & \textbf{{Expected signal}} & \textbf{{Use in interpretation}}\\
\midrule
Teleconnection composites & Strong vs. weak index states & Skill changes under remote forcing & Tests whether global context is used physically\\
Soil-moisture stratification & Dry vs. wet antecedent states & Larger heat amplification under dry soils & Links skill to land-atmosphere feedback\\
Regional grouping & West, Plains, Southeast, Northeast & Spatially heterogeneous skill & Identifies where prediction is meaningful\\
Event amplitude bins & Moderate vs. extreme anomalies & Reliability changes in tails & Tests handling of high-impact events\\
Input ablations & Full vs. local-only/global-only & Skill loss when pathways are removed & Supports interpretation of predictors\\
\bottomrule
\end{{tabularx}}
\end{{center}}

\begin{{center}}\scriptsize
\textbf{{Table 6. Risk-management logic for model outcomes.}}\\[0.03in]
\begin{{tabularx}}{{\textwidth}}{{p{{0.25\textwidth}}p{{0.32\textwidth}}X}}
\toprule
\textbf{{Outcome}} & \textbf{{Response}} & \textbf{{Scientific value}}\\
\midrule
CFM improves calibration & Release calibrated hindcast diagnostics & Demonstrates probabilistic graph-forecast utility\\
Only deterministic skill improves & Compare residual and ensemble behavior & Identifies uncertainty-modeling limitations\\
Skill is regime-limited & Publish regime-dependent predictability & Clarifies when week-3 prediction is plausible\\
Baselines match AI model & Report limits and failure modes & Prevents overclaiming and creates a benchmark\\
\bottomrule
\end{{tabularx}}
\end{{center}}

\newpage
\section*{{Detailed Work Plan: Broader Impacts and Training}}
The broader-impact work will be integrated with the research rather than treated as a separate activity. Reproducible code and derived diagnostics will be organized so students can follow the full workflow from data preparation to model evaluation. Teaching material will use small examples to explain graph neural weather prediction, forecast uncertainty, reliability, and the difference between experimental research products and operational guidance.

Mentoring activities will emphasize transparent evaluation, ethical use of AI, and climate-risk communication. Students will be encouraged to ask not only whether a model is accurate, but for whom, where, when, and with what uncertainty. This framing supports responsible use of AI in environmental prediction and helps broaden participation for students whose training begins in geography, environmental engineering, or data science.

\begin{{center}}\scriptsize
\textbf{{Table 7. Two-year work plan, products, and review checkpoints.}}\\[0.03in]
\begin{{tabularx}}{{\textwidth}}{{p{{0.13\textwidth}}p{{0.33\textwidth}}p{{0.25\textwidth}}X}}
\toprule
\textbf{{Period}} & \textbf{{Research activity}} & \textbf{{Product}} & \textbf{{Checkpoint}}\\
\midrule
Months 1--3 & Data audit, fold lock-in, baseline reproduction & Preprocessing and baseline report & Confirm no leakage and consistent samples\\
Months 4--6 & Deterministic model and skill maps & Regional TAC/R2 maps and tables & Identify regions with added skill\\
Months 7--12 & CFM ensembles and calibration & CRPS, reliability, rank histograms & Decide calibration strategy\\
Months 13--16 & Ablations and regime diagnostics & Teleconnection/soil-moisture composites & Link skill to mechanisms\\
Months 17--20 & Flash-drought heat-risk products & Exceedance maps and cases & Evaluate use-relevant uncertainty\\
Months 21--24 & Manuscripts, release, teaching module & Papers, repository, archive & Complete dissemination milestones\\
\bottomrule
\end{{tabularx}}
\end{{center}}

\begin{{center}}\scriptsize
\textbf{{Table 8. Broader-impact deliverables.}}\\[0.03in]
\begin{{tabularx}}{{\textwidth}}{{p{{0.24\textwidth}}p{{0.35\textwidth}}X}}
\toprule
\textbf{{Deliverable}} & \textbf{{Audience}} & \textbf{{Planned content}}\\
\midrule
Teaching module & Graduate and advanced undergraduate students & Graph weather models, uncertainty, reliability, and limitations\\
Reproducible workflow & Climate-data science learners and researchers & Data-processing scripts, model configs, diagnostics, metadata\\
Mentoring activities & REU and student researchers & Model evaluation, transparent reporting, climate-risk communication\\
Public derived products & Scientific community & Hindcast diagnostics, tables, and figure-generation workflows\\
\bottomrule
\end{{tabularx}}
\end{{center}}

"""


REFERENCES = r"""
\noindent Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., Merose, A., Hoyer, S., Holland, G., Vinyals, O., Stott, J., Pritzel, A., Mohamed, S., and Battaglia, P. (2023). Learning skillful medium-range global weather forecasting. \textit{Science}, 382(6677), 1416--1421.

\noindent Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T. R., El-Kadi, A., Masters, D., Ewalds, T., Stott, J., Mohamed, S., Battaglia, P., Lam, R., and Willson, M. (2024). Probabilistic weather forecasting with machine learning. \textit{Nature}, 637, 84--90.

\noindent Pegion, K., Kirtman, B. P., Becker, E., Collins, D. C., LaJoie, E., Burgman, R., Bell, R., DelSole, T., Min, D., Zhu, Y., Li, W., Sinsky, E., Guan, H., Gottschalck, J., Metzger, E. J., Barton, N. P., Achuthavarier, D., Marshak, J., Koster, R., Lin, H., Gagnon, N., Bell, M., Tippett, M. K., Robertson, A. W., Sun, S., Benjamin, S. G., Green, B. W., Bleck, R., and Kim, H. (2019). The Subseasonal Experiment SubX: A multimodel subseasonal prediction experiment. \textit{Bulletin of the American Meteorological Society}, 100(10), 2043--2060.

\noindent Vitart, F., Ardilouze, C., Bonet, A., Brookshaw, A., Chen, M., Codorean, C., Deque, M., Ferranti, L., Fucile, E., Fuentes, M., Hendon, H., Hodgson, J., Kang, H. S., Kumar, A., Lin, H., Liu, G., Liu, X., Malguzzi, P., Mallas, I., Manoussakis, M., Mastrangelo, D., MacLachlan, C., McLean, P., Minami, A., Mladek, R., Nakazawa, T., Najm, S., Nie, Y., Rixen, M., Robertson, A. W., Ruti, P., Sun, C., Takaya, Y., Tolstykh, M., Venuti, F., Waliser, D., Woolnough, S., Wu, T., Won, D. J., Xiao, H., Zaripov, R., and Zhang, L. (2017). The Sub-seasonal to Seasonal Prediction Project Database. \textit{Bulletin of the American Meteorological Society}, 98(1), 163--173.

\noindent White, C. J., Carlsen, H., Robertson, A. W., Klein, R. J. T., Lazo, J. K., Kumar, A., Vitart, F., Coughlan de Perez, E., Ray, A. J., Murray, V., Bharwani, S., MacLeod, D., James, R., Fleming, L., Morse, A. P., Eggen, B., Graham, R., Kjellstrom, E., Becker, E., Pegion, K. V., Holbrook, N. J., McEvoy, D., Depledge, M., Perkins-Kirkpatrick, S., Brown, T. J., Street, R., Jones, L., Remenyi, T. A., Hodgson-Johnston, I., Buontempo, C., Lamb, R., Meinke, H., Arheimer, B., and Zebiak, S. E. (2017). Potential applications of subseasonal-to-seasonal predictions. \textit{Meteorological Applications}, 24(3), 315--325.

\noindent Otkin, J. A., Anderson, M. C., Hain, C., Svoboda, M., Johnson, D., Mueller, R., Tadesse, T., Wardlow, B., and Brown, J. (2016). Assessing the evolution of soil moisture and vegetation conditions during the 2012 United States flash drought. \textit{Agricultural and Forest Meteorology}, 218--219, 230--242.

\noindent Rasp, S., Dueben, P. D., Scher, S., Weyn, J. A., Mouatadid, S., and Thuerey, N. (2020). WeatherBench: A benchmark data set for data-driven weather forecasting. \textit{Journal of Advances in Modeling Earth Systems}, 12, e2020MS002203.

\noindent Rezaali, M., Jahangir, M. S., Fouladi-Fard, R., and Keellings, D. (2024). An ensemble deep learning approach to spatiotemporal tropospheric ozone forecasting: A case study of Tehran, Iran. \textit{Urban Climate}, 55, 101950.

\noindent Rezaali, M., Quilty, J., and Karimi, A. (2021). Probabilistic urban water demand forecasting using wavelet-based machine learning models. \textit{Journal of Hydrology}, 600, 126358.
"""


BUDGET = r"""
\section*{Overview}
This AGS-PRF fellowship budget follows the solicitation-specified fellowship structure. The budget contains stipend support for the Fellow and a fellowship allowance. No voluntary committed cost sharing is included. The requested funds are mapped to the standard AGS-PRF structure and are justified by the research, dissemination, data-management, and professional-development activities described in the Project Description.

\section*{Year 1 Stipend}
\textbf{Stipend: \$70,000.} The stipend will support full-time fellowship effort by Mostafa Rezaali on research, broader impacts, mentoring, dissemination, and professional development. Activities include developing the conditional-flow forecast model, preparing the cross-validated hindcast archive, computing deterministic and probabilistic verification metrics, documenting workflows, and preparing the first manuscript. The stipend is not requested as salary for institutional personnel; it supports the Fellow's full-time training and research effort.

\section*{Year 1 Fellowship Allowance}
\textbf{Fellowship allowance: \$30,000.} Planned uses include health insurance and benefit-related costs; high-performance computing, cloud storage, and secure data-storage expenses; workstation or peripheral support needed for model development and analysis; conference and workshop travel to present results and receive feedback; open-access publication and data-archive fees; and modest software, training, and dissemination costs directly related to the fellowship. These costs are necessary because probabilistic graph-based climate forecasting requires repeated model training, ensemble evaluation, large derived data products, and transparent public release of results.

\section*{Year 2 Stipend}
\textbf{Stipend: \$72,000.} The second-year stipend will support full-time fellowship effort including probabilistic ensemble training, uncertainty-calibration analyses, physical-regime diagnostics, flash-drought-relevant risk evaluation, manuscript preparation, open-science release, student mentoring, and career-development activities. The second year emphasizes completion, interpretation, dissemination, and transition from model development to independent research leadership.

\section*{Year 2 Fellowship Allowance}
\textbf{Fellowship allowance: \$30,000.} Planned uses mirror Year 1 and will support continued health/benefit costs, computing and storage, conference participation, publication and data dissemination, project-specific supplies, software, and professional development. Anticipated travel includes at least one national conference such as AGU, AMS, or a comparable climate/AI meeting, plus targeted workshops or short training opportunities in subseasonal prediction, uncertainty quantification, or responsible AI.

\section*{Computing, Storage, and Software Rationale}
The project requires GPU-enabled training, repeated cross-validation, ensemble generation, reliability analysis, and storage of gridded hindcast products. Allowance funds may support computing allocations, cloud or institutional storage, backup media, data-transfer charges, and workstation peripherals needed to maintain efficient and reproducible workflows. Software costs, if any, will be limited to project-specific analysis, visualization, archive preparation, or accessibility needs not already provided by the host institution.

\section*{Travel, Publication, and Dissemination Rationale}
Travel funds will allow the Fellow to present results, receive critique from the atmospheric-science community, build professional networks, and identify pathways for translating probabilistic heat-risk diagnostics into useful climate services. Publication and archive funds will support open-access manuscripts, persistent data/code releases, and long-term preservation of reproducible research products.

\section*{Budget Control and Compliance}
All spending will be tied directly to fellowship research, training, or dissemination objectives. The Fellow will track allowance spending, maintain documentation for project-related expenses, and follow NSF, Research.gov, and host-institution rules. No funds are requested for unrelated personnel, construction, entertainment, or voluntary committed cost sharing.

\newpage
\section*{Detailed Computing and Storage Justification}
The model-development plan requires repeated training and evaluation of deterministic and probabilistic graph models across leave-years-out validation folds. The fellowship allowance may support GPU-enabled computing, scratch storage, archival storage, data-transfer costs, and backup resources needed to preserve reproducible products. These costs are tied directly to the project because CFM ensembles require multiple model samples, verification files, and derived diagnostics for reliability, CRPS, and regional skill analysis.

Storage needs will include processed PRISM and ERA5-derived predictors, fold-specific model inputs, trained model checkpoints, generated ensemble hindcasts, skill maps, summary tables, and manuscript figures. Funds may also support secure storage and backup media to reduce the risk of data loss during the fellowship.

\newpage
\section*{Detailed Travel and Dissemination Justification}
Travel funds will support presentation of project results at atmospheric-science, climate, geospatial, or AI-for-science meetings. These venues are important for receiving technical feedback, comparing model diagnostics with related subseasonal prediction work, building collaborations, and developing the Fellow's independent professional network. Allowable costs may include registration, transportation, lodging, meals, and related travel expenses consistent with institutional and NSF rules.

Dissemination costs may include open-access publication fees, data-archive costs, persistent DOI creation, accessibility preparation, and repository documentation. These costs are justified because the project explicitly commits to transparent release of nonrestricted derived products, code, and teaching materials.

\newpage
\section*{Detailed Training and Professional Development Justification}
The fellowship allowance may support short courses, workshops, or professional-development activities in subseasonal prediction, uncertainty quantification, scientific software, responsible AI, or climate-risk communication. These activities will strengthen the Fellow's ability to lead an independent research program and to translate AI methods into atmospheric-science questions.

Professional-development spending will be modest and project-relevant. Priority will be given to activities that improve research quality, reproducibility, mentoring capacity, or dissemination. The requested budget therefore supports both the scientific execution of the project and the AGS-PRF goal of developing an independent early-career scientist.
"""


FACILITIES = rf"""
\section*{{Host Institution}}
The host institution is {HOST}. The Department of Geography provides an appropriate environment for a fellowship centered on climate extremes, hydroclimatology, geospatial analysis, environmental prediction, and AI-enabled climate-risk modeling. The department provides access to scholarly mentoring, seminars, research collaboration, and professional-development opportunities relevant to the proposed work.

\section*{{Sponsoring Scientist and Intellectual Environment}}
The sponsoring scientist is {SPONSOR}. The sponsor's expertise in heat waves, climate extremes, and applied climatology directly supports the proposed research. The intellectual environment will help ensure that model development is evaluated against physical understanding of atmospheric variability, land-atmosphere coupling, and regional climate risk.

\section*{{Computational Resources}}
The project will use Python/PyTorch software environments, geospatial and climate-data processing tools, and GPU-enabled computing resources available through the host research environment and institutional computing pathways. The existing codebase includes scripts for model training, cross-validation, hindcast export, bootstrap statistics, baseline comparison, temporal anomaly correlation maps, sample maps, and probabilistic verification. These resources are adequate for training deterministic and probabilistic graph models, storing derived hindcast products, and generating reproducible diagnostics.

\section*{{Data Resources}}
The project will use publicly available and/or institutionally accessible climate datasets, including PRISM daily maximum temperature fields, ERA5 atmospheric variables, teleconnection indices, topography, land masks, seasonal radiation, and land-surface variables. Raw third-party datasets will be obtained through their official providers. Derived products needed to reproduce figures and tables will be documented and preserved according to the Data Management and Sharing Plan.

\section*{{Software and Analysis Environment}}
The analysis environment includes Python, scientific computing libraries, geospatial processing tools, NetCDF/GRIB workflows, visualization packages, and version-control practices. The project will emphasize reproducible workflow design, clear metadata, and portable scripts for model training and verification. No specialized laboratory equipment is required beyond computing, storage, and standard research software.

\section*{{Professional Development and Dissemination Resources}}
The host environment will support manuscript preparation, conference presentations, student mentoring, seminars, and development of teaching material related to trustworthy AI for climate extremes. These resources will help the Fellow build an independent research identity in probabilistic subseasonal prediction and climate-risk analytics.
"""


DMP = r"""
\section*{Products of Research}
The project will produce source code, trained-model configuration files, evaluation scripts, derived hindcast diagnostics, summary tables, selected nonrestricted figures, metadata, teaching materials, and manuscripts. The project will use third-party climate datasets such as PRISM and ERA5. Those source datasets will not be redistributed if license or provider terms require users to obtain them from the original sources.

\section*{Data and Code Formats}
Code will be released in plain-text Python and shell scripts with environment files where possible. Derived data products will be released in standard scientific formats such as NetCDF, NPZ, CSV, or GeoTIFF as appropriate, with README files describing variables, units, grids, time coverage, cross-validation splits, and known limitations. Figures and tables used in manuscripts will be preserved with scripts or notebooks sufficient to reproduce them from derived products.

\section*{Access and Sharing}
Nonrestricted derived products and code will be shared through a public repository and archived with a persistent DOI through a repository such as Zenodo or an institutional archive. Large intermediate files and third-party raw data will be documented with scripts and instructions so authorized users can reproduce them from original providers. Repository documentation will distinguish between raw third-party inputs, derived hindcast products, trained-model outputs, and manuscript-ready diagnostics.

\section*{Preservation}
Final code, metadata, manuscripts, teaching materials, and derived data needed to reproduce major figures and tables will be preserved for at least five years after the end of the award. Versioned releases will be used for manuscript-associated artifacts so that published results can be traced to a specific code and data snapshot.

\section*{Privacy, Security, and Ethical Considerations}
The project uses environmental and climate data, not human-subjects data. No personally identifiable information will be collected. Model limitations, uncertainty, and appropriate-use caveats will be documented to reduce the risk of overinterpreting experimental forecast products. Experimental forecasts will be presented as research outputs and not as official operational guidance.

\section*{Roles and Responsibilities}
The Fellow will maintain the repository, prepare metadata, document workflows, and archive final products. The sponsoring scientist will advise on scientific quality control, reproducible workflow design, and appropriate dissemination. Data-management tasks will be reviewed at manuscript and release milestones to ensure that code, metadata, and derived products remain consistent.

\newpage
\section*{Workflow Documentation and Reuse}
The project will include documentation for environment setup, data preprocessing, model training, evaluation, and figure generation. Where third-party raw data cannot be redistributed, the repository will provide clear instructions for obtaining the source data and rebuilding derived products. Configuration files will record variables, pressure levels, spatial grids, lead times, training years, validation years, and model settings.

\section*{Quality Control for Shared Products}
Before release, derived products will be checked for missing values, unit consistency, grid metadata, time-coordinate accuracy, and agreement with manuscript figures. Public products will include caveats explaining that the forecasts are research hindcasts rather than operational warnings. This documentation will help future users understand the scope, limits, and appropriate interpretation of the project outputs.
"""


BIOSKETCH = r"""
\section*{Biographical Sketch Common Form}
\textbf{Name:} Mostafa Rezaali\\
\textbf{Position Title:} Graduate Research Assistant and Ph.D. Candidate\\
\textbf{Organization:} University of Florida\\
\textbf{Source basis:} Prepared from SciENcv record 2685636 visible content and local CV publication details. The SciENcv page lists the University of Florida Ph.D. preparation entry, University of Florida appointments, and an empty related-products table; products below are therefore populated from the CV for completeness.

\section*{A. Professional Preparation}
\begin{tabularx}{\textwidth}{p{0.30\textwidth}p{0.21\textwidth}p{0.23\textwidth}X}
\toprule
\textbf{Organization} & \textbf{Location} & \textbf{Degree / Training} & \textbf{Field / Date}\\
\midrule
University of Florida & Gainesville, Florida & Doctor of Philosophy & Department of Geography, Aug 2026\\
University of Florida & Gainesville, Florida & Graduate Certificate & Atmospheric Sciences\\
Qom University of Technology & Qom, Iran & M.Sc. & Civil and Environmental Engineering, 2018\\
IAUKHSH & Isfahan, Iran & B.Sc. & Civil and Environmental Engineering, 2016\\
\bottomrule
\end{tabularx}

\section*{B. Appointments and Positions}
\begin{tabularx}{\textwidth}{p{0.16\textwidth}p{0.31\textwidth}p{0.34\textwidth}X}
\toprule
\textbf{Dates} & \textbf{Title} & \textbf{Organization / Department} & \textbf{Location}\\
\midrule
2022--2026 & GRA & University of Florida & Gainesville, Florida\\
2022--Present & Graduate Research Assistant (PhD Candidate) & University of Florida, Department of Geography & Gainesville, Florida\\
2025 & NSF LEAP REU Mentor & University of Florida & Gainesville, Florida\\
2025 & Invited Lecturer, Deep Learning Applications in Climate Science & University of Florida & Gainesville, Florida\\
2025 & Invited Lecturer, Climate Change Impacts on Future Plant Distributions & University of Florida & Gainesville, Florida\\
2021--Present & M.Sc. Student Advisor & Alborz University of Medical Sciences & Iran\\
\bottomrule
\end{tabularx}

\section*{C. Products}
\textbf{Lead-authored products related to the proposed project, sorted by date.}
\begin{enumerate}
\item Rezaali, M., Fouladi-Fard, R., O'Shaughnessy, P., Naddafi, K., and Karimi, A. (2025). Assessment of AERMOD and ADMS for NOx dispersion modeling with a combination of line and point sources. \textit{Stochastic Environmental Research and Risk Assessment}.
\item Rezaali, M., Jahangir, M. S., Fouladi-Fard, R., and Keellings, D. (2024). An ensemble deep learning approach to spatiotemporal tropospheric ozone forecasting: A case study of Tehran, Iran. \textit{Urban Climate}, 55, 101950.
\item Rezaali, M., Fouladi-Fard, R., and Karimi, A. (2023). Performance of TANN, NARX, and GMDHT models for urban water demand forecasting. \textit{Avicenna Journal of Environmental Health Engineering}, 10(2), 85--97.
\item Rezaali, M., Quilty, J., and Karimi, A. (2021). Probabilistic urban water demand forecasting using wavelet-based machine learning models. \textit{Journal of Hydrology}, 600, 126358.
\item Rezaali, M., Fouladi-Fard, R., Mojarad, H., Sorooshian, A., Mahdinia, M., et al. (2021). A wavelet-based random forest approach for indoor BTEX spatiotemporal modeling and health risk assessment. \textit{Environmental Science and Pollution Research}, 28, 22522--22535.
\end{enumerate}

\textbf{Other significant products.}
\begin{enumerate}
\item Narayanan, A., Rezaali, M., Bunting, E. L., and Keellings, D. (2025). It's getting hot in here: Spatial impact of humidity on heat wave severity in the U.S. \textit{Science of The Total Environment}, 963, 178397.
\item Farajollahi, M., Fahiminia, M., Fouladi-Fard, R., Rezaali, M., and Sorooshian, A. (2024). Human and ecological health-risk research related to environmental exposure.
\item Rezaali, M., and Fouladi-Fard, R. (2021). Aerosolized SARS-CoV-2 exposure assessment: dispersion modeling with AERMOD. \textit{Journal of Environmental Health Science and Engineering}, 19, 285--293.
\item Rezaali, M., Fouladi-Fard, R., and Karimi, A. (2020). Identification of temporal and spatial patterns of river water quality parameters using NLPCA and multivariate statistical techniques. \textit{International Journal of Environmental Science and Technology}, 17, 2977--2994.
\item Rezaali, M., and Karimi, A. (2019). Decentralized wastewater treatment plants site selection of Qom Province using environmental and spatial criteria.
\end{enumerate}

\section*{D. Synergistic Activities}
\begin{enumerate}
\item Mentored emerging climate-data scientists through NSF LEAP REU activities and student research advising.
\item Delivered invited instruction on deep learning applications in climate science and climate-change impacts.
\item Developed interdisciplinary AI workflows for climate, hydrology, air quality, and environmental health.
\item Provided peer-review service for journals including \textit{Scientific Reports}, \textit{Journal of Hydrology}, and \textit{Springer Nature Applied Sciences}.
\item Built reproducible Python, MATLAB, R, and geospatial workflows for large climate and environmental datasets.
\end{enumerate}

\section*{Eligibility and Certification Context}
Mostafa Rezaali is a Ph.D. candidate applying as an early-career researcher transitioning from doctoral training to postdoctoral research and does not hold a tenure-track or equivalent permanent research position. The Ph.D. preparation entry in SciENcv lists University of Florida, Department of Geography, with expected receipt in August 2026. The final submission should confirm the exact degree timeline and use the certified SciENcv export if Research.gov requires SciENcv certification.
"""


MENTORING = rf"""
\section*{{Mentoring Framework}}
Although the AGS-PRF applicant is the postdoctoral fellow rather than a hired postdoctoral researcher, the fellowship includes a structured mentoring relationship with the sponsoring scientist. The proposed sponsoring scientist is {SPONSOR}, and the proposed host institution is {HOST}. The mentoring framework will support scientific independence, technical growth, publication planning, professional networking, teaching development, and preparation for a faculty or research-scientist career.

\section*{{Research Mentoring}}
The Fellow and sponsoring scientist will meet regularly to review research progress, model diagnostics, physical interpretation, manuscript development, conference abstracts, and reproducible workflow milestones. Mentoring will emphasize rigorous climate-science interpretation of AI results, transparent uncertainty communication, and responsible framing of experimental forecast products.

\section*{{Professional Development}}
Professional development will include manuscript planning, proposal-writing feedback, conference presentation preparation, networking with atmospheric-science and geospatial communities, and opportunities to mentor students in climate-data science. The Fellow will also develop teaching material on trustworthy AI for climate extremes and will seek feedback on communicating uncertainty to interdisciplinary audiences.

\section*{{Assessment}}
Progress will be assessed through quarterly milestones: baseline and CFM model completion, verification diagnostics, physical-regime analysis, manuscript drafts, public release preparation, and career-development activities. The mentoring relationship will be adjusted as needed to ensure the Fellow develops an independent research identity.
"""


DOCS = [
    ("project_summary", "Project Summary", PROJECT_SUMMARY),
    ("project_description", "Project Description", PROJECT_DESCRIPTION),
    ("references_cited", "References Cited", REFERENCES),
    ("budget_justification", "Budget Justification", BUDGET),
    ("facilities_equipment_other_resources", "Facilities, Equipment and Other Resources", FACILITIES),
    ("data_management_plan", "Data Management and Sharing Plan", DMP),
    ("biographical_sketch_mostafa_rezaali", "Biographical Sketch", BIOSKETCH),
    ("mentoring_plan", "Mentoring Plan", MENTORING),
]


def main():
    outputs = []
    for stem, title, body in DOCS:
        pdf = compile_doc(stem, title, body)
        outputs.append(pdf)

    for pdf in outputs:
        target = UPLOAD / pdf.name
        target.write_bytes(pdf.read_bytes())

    print("Generated final-format PDFs:")
    for pdf in outputs:
        print(pdf)


if __name__ == "__main__":
    main()

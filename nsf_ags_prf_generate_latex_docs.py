from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "nsf_ags_prf_uploads" / "latex_main"
OUT.mkdir(parents=True, exist_ok=True)


PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}

% ======= PACKAGES =======
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{setspace}
\setstretch{1.5}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{lastpage}
\usepackage{hyperref}
\usepackage{mathpazo}
\usepackage{graphicx}
\usepackage{subcaption}
\linespread{1.5}

% ======= HEADER & FOOTER =======
\pagestyle{fancy}
\fancyhf{}

% Header (faded + smaller)
\fancyhead[L]{\footnotesize\textcolor{gray!90}{\textit{%HEADER%}}}
\fancyhead[R]{\footnotesize\textcolor{gray!90}{\textit{Mostafa Rezaali}}}

% Footer (faded)
\fancyfoot[L]{\footnotesize\textcolor{gray!90}{\textit{NSF AGS-PRF -- University of Florida}}}
\fancyfoot[R]{\footnotesize\textcolor{gray!90}{Page \thepage}}

\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

% ======= PARAGRAPH STYLE =======
\setlength{\parindent}{15pt}
\setlength{\parskip}{6pt}
\titlespacing*{\section}{0pt}{0.45em}{0.15em}
\titlespacing*{\subsection}{0pt}{0.35em}{0.1em}

% ======= DOCUMENT BEGINS =======
\begin{document}

% ======= TITLE BLOCK =======
\begin{center}
    \vspace*{-1.5cm}
    {\Large\bfseries Postdoctoral Fellowship Application} \\[0.2cm]
    {\small\textit{NSF Atmospheric and Geospace Sciences Postdoctoral Research Fellowship}} \\[0.05cm]
    {\small\textit{Department of Geography, University of Florida}} \\[0.1cm]
    \rule{0.6\textwidth}{0.4pt}
\end{center}
\vspace{-0.5cm}

\noindent
Mostafa Rezaali \\
Ph.D. Candidate, Climate Science (Geography) \\
University of Florida \\
mostafarezaali@gmail.com \quad $\mid$ \quad (352) 709-5350 \\[0.3cm]
{\large\bfseries %TITLE%}\\[-0.15cm]
\rule{\textwidth}{0.4pt}
"""


ENDING = "\n\\end{document}\n"


DOCS = {
    "project_summary": {
        "header": "Project Summary",
        "title": "Project Summary",
        "body": r"""
\small
\setstretch{1.08}
\vspace{-0.15cm}
\section*{Overview}
Extreme heat and rapidly developing drought are among the most damaging U.S. climate hazards, yet week-3 predictability remains limited because local land-atmosphere feedbacks interact with global circulation anomalies, teleconnections, and soil-moisture memory. This fellowship will develop a probabilistic, physics-informed GeoAI framework for 15-day prediction of CONUS daily maximum temperature anomalies and flash-drought-relevant heat risk. The project builds from my conditional-flow-matching (CFM) model, a GraphCast-style icosahedral mesh network trained with PRISM, ERA5, teleconnection indices, topography, seasonal radiation, soil moisture, and global atmospheric context fields.

\section*{Intellectual Merit}
The project asks how much week-3 predictability is contained in global circulation, land-surface memory, local persistence, and teleconnection state, and whether CFM can extract that signal with calibrated uncertainty. Three aims will: (1) extend deterministic MeshFlowNet into a probabilistic CFM ensemble; (2) diagnose regional and regime-dependent skill using leave-years-out hindcasts for 1981--2023; and (3) evaluate heat-wave and flash-drought-relevant exceedance probabilities against persistence, climatology, and teleconnection baselines. Preliminary hindcasts over 5,332 MJJAS samples show stitched temporal anomaly correlation of 0.1016 versus 0.0735 for persistence, motivating a focused study of where graph-based AI adds physically meaningful subseasonal forecast value.

\section*{Broader Impacts}
Improved week-3 heat-risk guidance can support public-health planning, water resources, energy operations, and agricultural preparedness. The fellowship will produce reproducible workflows, hindcast diagnostics, teaching modules on trustworthy AI for climate extremes, and mentoring opportunities for students entering climate-data science from geography, environmental engineering, and data science.
""",
    },
    "project_description": {
        "header": "Project Description",
        "title": "Project Description",
        "body": r"""
\section*{Project Title}
\textbf{Postdoctoral Fellowship: AGS-PRF: Probabilistic Prediction of Heat Waves and Flash Droughts Using Physics-Informed Conditional Flow Matching on an Icosahedral Mesh}

\section*{Motivation and Central Hypothesis}
Heat waves and flash droughts are compound hydroclimate hazards with large consequences for public health, energy demand, water resources, ecosystems, and agriculture. Their impacts often emerge on decision horizons of two to four weeks, when emergency managers, utilities, farmers, and public-health agencies still have time to act, but when deterministic forecast skill is often weak. Predictability at these leads is not purely local. It reflects slowly evolving ocean-atmosphere patterns, stationary waves, soil-moisture memory, radiative seasonality, synoptic persistence, and land-surface feedbacks.

My central hypothesis is that a graph-based conditional flow model, trained directly on high-resolution U.S. temperature anomalies while conditioned on global circulation and teleconnection states, can reveal and exploit a measurable but spatially heterogeneous week-3 signal for extreme heat and flash-drought precursors. The proposed work treats AI not as a replacement for physical reasoning, but as a tool for testing where atmosphere-land-ocean memory produces useful probabilistic predictability.

\section*{Preliminary Work}
I have implemented MeshFlowNet, a GraphCast-style icosahedral mesh encoder-processor-decoder for direct 15-day CONUS daily maximum temperature prediction. The model uses current and lagged local PRISM temperature anomalies, atmospheric and land-surface fields, topography, latitude, longitude, seasonal insolation, land mask, five teleconnection indices, and a 59-channel global ERA5/coarse atmospheric context stack. It supports deterministic residual prediction and probabilistic conditional-flow-matching modes.

The current deterministic 5-fold leave-years-out hindcast covers MJJAS 1981--2023, with each year predicted by a model that did not train on that year. Across 5,332 samples, stitched model temporal anomaly correlation (TAC) is 0.1016 compared with 0.0735 for persistence, a +0.0281 improvement. Per-fold validation $R^2$ is approximately 0.58--0.60, and sample maps show skillful reconstruction of several high-amplitude heat anomalies. These results are preliminary, but they show that the model extracts nontrivial anomaly signal at a difficult lead time and justifies a systematic postdoctoral investigation of predictability, uncertainty, and physical interpretation.

\section*{Aim 1: Build a Calibrated Conditional-Flow Forecast System}
The first aim will convert the current deterministic residual model into a probabilistic forecast system. The deterministic branch predicts $y(t+15)$ as persistence plus a learned residual in one forward pass. The conditional-flow branch will learn a velocity field from initialization-time state to the verifying anomaly field, conditioned on teleconnection indices and global atmosphere/ocean fields. Ensembles will be generated by sampling the flow and by training multiple seeds per cross-validation fold.

Calibration will be assessed with continuous ranked probability score (CRPS), reliability diagrams for exceedance of local 90th and 95th percentiles, rank histograms, Brier score, and spatially stratified spread-skill relationships. This aim will test whether conditional flow matching can deliver useful uncertainty rather than only sharper deterministic maps. A successful result will be a reproducible probabilistic benchmark for week-3 U.S. heat prediction.

\section*{Aim 2: Diagnose Sources of Subseasonal Predictability}
The second aim will use controlled ablations and interpretable diagnostics to determine which features carry week-3 skill. I will compare models trained with: local persistence only; local meteorology plus static and seasonal terms; teleconnection indices; global ERA5 fields; and the full conditioning stack. The analysis will produce per-pixel TAC, skill over persistence, regional skill summaries for NOAA climate regions, block-bootstrap significance tests by year, and event composites for strong teleconnection regimes.

The scientific target is not only a higher metric, but a physical account of where AI forecast skill is consistent with known pathways such as Pacific-North American wave trains, moisture transport, soil-moisture feedback, subtropical ridge variability, and persistent high-pressure anomalies. These diagnostics will help distinguish robust predictive signal from overfitting or spatially smooth climatological reconstruction.

\section*{Aim 3: Connect Heat Prediction to Flash-Drought-Relevant Risk}
The third aim will connect temperature-anomaly prediction to flash-drought precursors by evaluating rapid intensification of heat and atmospheric demand in conjunction with antecedent soil-moisture state. The model already ingests soil moisture, humidity, winds, sea-level pressure, radiation, and large-scale circulation fields. I will evaluate whether the learned representations anticipate co-occurring hot, dry, high-demand regimes over agriculturally important regions.

Products will be reported as calibrated probabilistic exceedance maps, regional time series, and case studies rather than deterministic binary declarations. This framing keeps the science focused on predictability and uncertainty while making the outcomes relevant to water, agriculture, energy, and health decision contexts.

\section*{Data and Model Architecture}
The hindcast system will use PRISM daily maximum temperature over CONUS at 0.04-degree resolution, ERA5 atmospheric variables and global context fields, teleconnection indices, topography, latitude, longitude, land mask, top-of-atmosphere insolation, and seasonal encodings. The current local input stack contains current and lagged PRISM temperature fields; geopotential, soil moisture, sea-level pressure, 2-m temperature, 850-hPa humidity, 850-hPa temperature and winds, 300-hPa geopotential; topography; latitude and longitude; day-of-year sine and cosine; insolation; and land mask. The global context stack contains SST, OLR, winds, geopotential, temperature, humidity, surface pressure, sea-level pressure, and total column water vapor across multiple levels.

MeshFlowNet maps the CONUS grid to an icosahedral mesh, performs message passing with graph interaction networks and conditioning, and decodes back to the land grid. This representation preserves global-to-regional context while avoiding purely convolutional locality. The probabilistic branch will use continuous-time embeddings and conditional flow matching; the deterministic branch will remain as a transparent baseline and an efficient ablation.

\section*{Evaluation and Validation}
All claims will be based on leave-years-out cross-validation over 1981--2023. Metrics will include TAC, spatial anomaly $R^2$, MAE, MSE skill relative to persistence, CRPS, reliability of extreme exceedance probabilities, and block-bootstrap confidence intervals. Baselines will include climatology, persistence, ridge regression on teleconnection/global summary features, and published operational S2S reference skill where metric and domain are comparable. I will report negative or regionally limited results explicitly because understanding where AI models fail is essential for trustworthy climate-risk use.

\section*{Intellectual Merit}
The intellectual merit is the integration of subseasonal predictability theory, graph neural weather models, and probabilistic flow matching. The project will advance knowledge by quantifying when global and land-surface information improves 15-day U.S. heat prediction beyond persistence; producing spatially explicit diagnostics of learned teleconnection skill; and testing whether conditional-flow models can provide calibrated uncertainty for regional extreme-temperature risk. The outcome will be a physically evaluated AI framework rather than a black-box score table.

\section*{Broader Impacts}
The broader impacts center on usable, transparent climate-risk intelligence and training. First, the project will produce reproducible code, derived hindcast diagnostics, and documentation suitable for students and practitioners who need to understand AI forecast limitations. Second, I will develop a short teaching module on graph neural weather prediction and uncertainty for graduate climate modeling or environmental data-science courses. Third, I will mentor undergraduate and REU students in open climate-data workflows, with attention to students entering atmospheric science from geography, environmental engineering, and data science. Fourth, the forecast diagnostics will be framed around heat-health, water, energy, and agriculture use cases where calibrated uncertainty can be more valuable than a single deterministic forecast.

\section*{Host Institution, Mentoring, and Career Development}
The proposed host context is the Department of Geography at the University of Florida, which provides a strong intellectual environment for climate science, hydroclimatology, climate extremes, geospatial analysis, and AI-enabled environmental prediction. The fellowship will support a transition from doctoral model development to an independent research program in probabilistic subseasonal predictability, uncertainty quantification, and open forecast diagnostics.

Professional development will include regular research meetings, manuscript and proposal-development milestones, conference presentations, student-mentoring opportunities, training in responsible and human-centered GeoAI, and explicit preparation for independent faculty or research-scientist positions. The mentoring plan will emphasize scientific rigor, technical independence, communication to interdisciplinary audiences, and development of a visible research identity in AI-enabled atmospheric predictability.

\section*{Facilities and Resources}
The project will use research computing resources, Python/PyTorch software environments, geospatial and climate-data analysis tools, and publicly available PRISM/ERA5-derived data workflows. The local codebase already runs on GPU-equipped computing systems and includes scripts for training, hindcast export, bootstrap statistics, baseline comparison, TAC maps, sample maps, and CRPS/reliability analysis. The separate Facilities, Equipment and Other Resources upload states: See the Project Description, as requested by the AGS-PRF solicitation.

\section*{Timeline and Milestones}
Months 1--6: finalize data audit, deterministic baseline, ridge/climatology/persistence baselines, and per-pixel skill maps. Months 7--12: train probabilistic conditional-flow ensembles, compute CRPS and reliability diagnostics, and prepare the first manuscript on week-3 heat predictability. Months 13--18: complete teleconnection-regime ablations, block-bootstrap significance tests, and flash-drought-relevant risk diagnostics. Months 19--24: release reproducible workflows and derived hindcast products, submit a second manuscript, present results at AGU/AMS or comparable meetings, and complete career-development milestones with mentors.

\section*{Assessment of Success}
Success will be assessed by scientific and broader-impact outputs: (1) a documented cross-validated hindcast benchmark with uncertainty metrics; (2) at least two manuscripts, one on model and predictability and one on physical-regime interpretation or flash-drought-relevant risk; (3) public release of code, metadata, and nonrestricted derived products; (4) student mentoring and teaching materials; and (5) a clear professional transition from doctoral model development to an independent research program in AI-enabled atmospheric predictability.
""",
    },
    "references_cited": {
        "header": "References Cited",
        "title": "References Cited",
        "body": r"""
\noindent Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., Merose, A., Hoyer, S., Holland, G., Vinyals, O., Stott, J., Pritzel, A., Mohamed, S., and Battaglia, P. (2023). Learning skillful medium-range global weather forecasting. \textit{Science}, 382(6677), 1416--1421.

\noindent Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T. R., El-Kadi, A., Masters, D., Ewalds, T., Stott, J., Mohamed, S., Battaglia, P., Lam, R., and Willson, M. (2024). Probabilistic weather forecasting with machine learning. \textit{Nature}, 637, 84--90.

\noindent Pegion, K., Kirtman, B. P., Becker, E., Collins, D. C., LaJoie, E., Burgman, R., Bell, R., DelSole, T., Min, D., Zhu, Y., Li, W., Sinsky, E., Guan, H., Gottschalck, J., Metzger, E. J., Barton, N. P., Achuthavarier, D., Marshak, J., Koster, R., Lin, H., Gagnon, N., Bell, M., Tippett, M. K., Robertson, A. W., Sun, S., Benjamin, S. G., Green, B. W., Bleck, R., and Kim, H. (2019). The Subseasonal Experiment (SubX): A multimodel subseasonal prediction experiment. \textit{Bulletin of the American Meteorological Society}, 100(10), 2043--2060.

\noindent Vitart, F., Ardilouze, C., Bonet, A., Brookshaw, A., Chen, M., Codorean, C., Deque, M., Ferranti, L., Fucile, E., Fuentes, M., Hendon, H., Hodgson, J., Kang, H. S., Kumar, A., Lin, H., Liu, G., Liu, X., Malguzzi, P., Mallas, I., Manoussakis, M., Mastrangelo, D., MacLachlan, C., McLean, P., Minami, A., Mladek, R., Nakazawa, T., Najm, S., Nie, Y., Rixen, M., Robertson, A. W., Ruti, P., Sun, C., Takaya, Y., Tolstykh, M., Venuti, F., Waliser, D., Woolnough, S., Wu, T., Won, D. J., Xiao, H., Zaripov, R., and Zhang, L. (2017). The Sub-seasonal to Seasonal (S2S) Prediction Project Database. \textit{Bulletin of the American Meteorological Society}, 98(1), 163--173.

\noindent White, C. J., Carlsen, H., Robertson, A. W., Klein, R. J. T., Lazo, J. K., Kumar, A., Vitart, F., Coughlan de Perez, E., Ray, A. J., Murray, V., Bharwani, S., MacLeod, D., James, R., Fleming, L., Morse, A. P., Eggen, B., Graham, R., Kjellstrom, E., Becker, E., Pegion, K. V., Holbrook, N. J., McEvoy, D., Depledge, M., Perkins-Kirkpatrick, S., Brown, T. J., Street, R., Jones, L., Remenyi, T. A., Hodgson-Johnston, I., Buontempo, C., Lamb, R., Meinke, H., Arheimer, B., and Zebiak, S. E. (2017). Potential applications of subseasonal-to-seasonal (S2S) predictions. \textit{Meteorological Applications}, 24(3), 315--325.

\noindent Otkin, J. A., Anderson, M. C., Hain, C., Svoboda, M., Johnson, D., Mueller, R., Tadesse, T., Wardlow, B., and Brown, J. (2016). Assessing the evolution of soil moisture and vegetation conditions during the 2012 United States flash drought. \textit{Agricultural and Forest Meteorology}, 218--219, 230--242.

\noindent Rasp, S., Dueben, P. D., Scher, S., Weyn, J. A., Mouatadid, S., and Thuerey, N. (2020). WeatherBench: A benchmark data set for data-driven weather forecasting. \textit{Journal of Advances in Modeling Earth Systems}, 12, e2020MS002203.

\noindent Rezaali, M., Jahangir, M. S., Fouladi-Fard, R., and Keellings, D. (2024). An ensemble deep learning approach to spatiotemporal tropospheric ozone forecasting: A case study of Tehran, Iran. \textit{Urban Climate}, 55, 101950.

\noindent Rezaali, M., Quilty, J., and Karimi, A. (2021). Probabilistic urban water demand forecasting using wavelet-based machine learning models. \textit{Journal of Hydrology}, 600, 126358.
""",
    },
    "budget_justification": {
        "header": "Budget Justification",
        "title": "Budget Justification",
        "body": r"""
\section*{Overview}
This AGS-PRF fellowship budget follows the solicitation-specified fellowship structure. The budget contains stipend support for the Fellow and a fellowship allowance. No voluntary committed cost sharing is included.

\section*{Year 1}
\textbf{Stipend: \$70,000.} The stipend will support full-time fellowship effort by Mostafa Rezaali on the research, broader-impact, mentoring, dissemination, and professional-development activities described in the Project Description. Activities include developing the conditional-flow forecast model, preparing the cross-validated hindcast archive, computing deterministic and probabilistic verification metrics, and preparing the first manuscript.

\textbf{Fellowship allowance: \$30,000.} Planned uses include health insurance and benefit-related costs; high-performance computing, cloud storage, and secure data-storage expenses; workstation or peripheral support needed for model development and analysis; conference and workshop travel to present results and receive feedback from the atmospheric-science and GeoAI communities; open-access publication and data-archive fees; and modest software, training, and dissemination costs directly related to the fellowship.

\section*{Year 2}
\textbf{Stipend: \$72,000.} The stipend will support full-time fellowship effort during the second year, including probabilistic ensemble training, uncertainty-calibration analyses, physical-regime diagnostics, manuscripts, open-science release, student mentoring, and career-development activities.

\textbf{Fellowship allowance: \$30,000.} Planned uses mirror Year 1 and will support continued health/benefit costs, computing and storage, conference participation, publication and data dissemination, and project-specific supplies or software. The allowance will be administered responsibly by the Fellow in accordance with NSF and AGS-PRF requirements.

\section*{Budget Rationale}
The requested resources are necessary because probabilistic graph-based climate forecasting requires substantial computing, data storage, reproducible workflow development, and dissemination. The fellowship allowance will enable training and evaluation of ensembles, preservation of derived diagnostics, presentation of results to the atmospheric-science community, and open release of reusable scientific products. All requested items are directly aligned with the research and broader-impact activities of the fellowship.
""",
    },
    "facilities_equipment_other_resources": {
        "header": "Facilities, Equipment and Other Resources",
        "title": "Facilities, Equipment and Other Resources",
        "body": r"""
\section*{Statement}
See the Project Description.

\section*{Context}
The AGS-PRF solicitation instructs applicants to enter ``See the Project Description'' in this section. The relevant facilities, computational resources, software environment, data resources, mentoring structure, and professional-development context are therefore described in the Project Description rather than repeated here.
""",
    },
    "data_management_plan": {
        "header": "Data Management Plan",
        "title": "Data Management Plan",
        "body": r"""
\section*{Products of Research}
The project will produce source code, trained-model configuration files, evaluation scripts, derived hindcast diagnostics, summary tables, selected nonrestricted figures, metadata, teaching materials, and manuscripts. The project will use third-party climate datasets such as PRISM and ERA5. Those source datasets will not be redistributed if license or provider terms require users to obtain them from the original sources.

\section*{Data and Code Formats}
Code will be released in plain-text Python and shell scripts with environment files where possible. Derived data products will be released in standard scientific formats such as NetCDF, NPZ, CSV, or GeoTIFF as appropriate, with README files describing variables, units, grids, time coverage, cross-validation splits, and known limitations. Figures and tables used in manuscripts will be preserved with scripts or notebooks sufficient to reproduce them from the derived products.

\section*{Access and Sharing}
Nonrestricted derived products and code will be shared through a public repository and archived with a persistent DOI through a repository such as Zenodo or an institutional archive. Large intermediate files and third-party raw data will be documented with scripts and instructions so that authorized users can reproduce them from original providers. Repository documentation will distinguish between raw third-party inputs, derived hindcast products, trained-model outputs, and manuscript-ready diagnostics.

\section*{Preservation}
Final code, metadata, manuscripts, teaching materials, and derived data needed to reproduce major figures and tables will be preserved for at least five years after the end of the award. Versioned releases will be used for manuscript-associated artifacts so that published results can be traced to a specific code and data snapshot.

\section*{Privacy, Security, and Ethical Considerations}
The project uses environmental and climate data, not human-subjects data. No personally identifiable information will be collected. Model limitations, uncertainty, and appropriate-use caveats will be documented to reduce the risk of overinterpreting experimental forecast products. Experimental forecasts will be presented as research outputs and not as official operational guidance.

\section*{Roles and Responsibilities}
The Fellow will be responsible for maintaining the repository, preparing metadata, documenting workflows, and archiving final products. Mentors will advise on scientific quality control, reproducible workflow design, and appropriate dissemination.
""",
    },
    "phd_abstract": {
        "header": "Ph.D. Abstract",
        "title": "Ph.D. Abstract",
        "body": r"""
My doctoral research in Climate Science and Geography at the University of Florida focuses on artificial intelligence for extreme weather, especially heat waves and flash drought. The work develops and evaluates deep learning methods for spatiotemporal climate prediction, with attention to large-scale atmospheric drivers, local land-surface feedbacks, and forecast verification.

A major component is MeshFlowNet, a GraphCast-style graph neural network on an icosahedral mesh for direct 15-day prediction of CONUS daily maximum temperature anomalies. The model combines local PRISM temperature history, ERA5 atmospheric and land-surface variables, teleconnection indices, global circulation fields, topography, seasonal radiation, and land-mask information. My dissertation research uses cross-validated hindcasts, temporal anomaly correlation, uncertainty metrics, and baseline comparisons to determine where learned forecast models add value over climatology and persistence.

The broader goal is to improve physically interpretable and uncertainty-aware climate-risk prediction for heat, water, energy, agriculture, and public-health planning. This doctoral work provides the technical and scientific foundation for the proposed NSF AGS-PRF project on physics-informed conditional flow matching for probabilistic prediction of heat waves and flash-drought-relevant risk.
""",
    },
}


def write_and_compile(stem: str, spec: dict) -> Path:
    tex = PREAMBLE.replace("%HEADER%", spec["header"]).replace("%TITLE%", spec["title"])
    tex += "\n" + spec["body"].strip() + ENDING
    tex_path = OUT / f"{stem}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    subprocess.run(cmd, cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    subprocess.run(cmd, cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return tex_path.with_suffix(".pdf")


def main():
    outputs = []
    for stem, spec in DOCS.items():
        outputs.append(write_and_compile(stem, spec))

    # Keep the project summary name expected by the portal workflow too.
    summary_copy = ROOT / "nsf_ags_prf_uploads" / "project_summary_latex_template.pdf"
    summary_copy.write_bytes((OUT / "project_summary.pdf").read_bytes())

    print("Generated LaTeX proposal PDFs:")
    for path in outputs:
        print(path)
    print(summary_copy)


if __name__ == "__main__":
    main()

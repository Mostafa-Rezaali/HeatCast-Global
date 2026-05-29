from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "nsf_ags_prf_uploads" / "personnel_latex"
OUT.mkdir(parents=True, exist_ok=True)

COMMON_PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}

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
\titlespacing*{\section}{0pt}{0.55em}{0.2em}
\titlespacing*{\subsection}{0pt}{0.45em}{0.15em}

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
\href{mailto:mostafarezaali@gmail.com}{mostafarezaali@gmail.com} \quad $\mid$ \quad (352) 709-5350 \\[0.3cm]
{\large\bfseries %TITLE%}\\[-0.15cm]
\rule{\textwidth}{0.4pt}
"""

ENDING = r"\end{document}" + "\n"


DOCS = {
    "biographical_sketch_mostafa_rezaali": {
        "header": "Biographical Sketch",
        "title": "Biographical Sketch",
        "body": r"""
\section*{Identifying Information}
Mostafa Rezaali is a Ph.D. Candidate in Climate Science (Geography) at the University of Florida and a lawful permanent resident of the United States. His research develops artificial-intelligence methods for climate extremes and environmental prediction, with emphasis on heat waves, flash droughts, spatiotemporal climate forecasting, air-quality modeling, and uncertainty-aware decision support. His NSF AGS-PRF work is centered on conditional flow matching (CFM) and graph-based atmospheric representations for probabilistic prediction of subseasonal U.S. heat extremes.

Rezaali's training spans climate science, geography, civil and environmental engineering, atmospheric sciences, machine learning, and geospatial data science. This interdisciplinary preparation positions him to translate recent advances in graph neural networks, generative modeling, and probabilistic forecast verification into tools for atmospheric science, climate-risk assessment, and public-interest forecasting.

\section*{Professional Preparation}
\begin{tabular}{p{0.30\textwidth}p{0.27\textwidth}p{0.18\textwidth}p{0.15\textwidth}}
\textbf{Institution} & \textbf{Location} & \textbf{Degree/Area} & \textbf{Dates}\\
University of Florida & Gainesville, FL & Ph.D., Climate Science/Geography & 2022--present\\
University of Florida & Gainesville, FL & Graduate Certificate, Atmospheric Sciences & 2022--present\\
Qom University of Technology & Qom, Iran & M.Sc., Civil and Environmental Engineering & 2016--2018\\
IAUKHSH & Isfahan, Iran & B.Sc., Civil and Environmental Engineering & 2011--2016\\
\end{tabular}

\section*{Appointments and Positions}
\begin{itemize}
\item Graduate Research Assistant, Department of Geography, University of Florida, 2022--present.
\item NSF LEAP REU Mentor, 2025.
\item Invited Lecturer, Deep Learning Applications in Climate Science, University of Florida, 2025.
\item Invited Lecturer, Climate Change Impacts on Future Plant Distributions, University of Florida, 2025.
\item M.Sc. Student Advisor, Alborz University of Medical Sciences, 2021--present.
\item Research and modeling contributor to projects involving machine learning, climate extremes, water-demand forecasting, air-quality prediction, and environmental-health risk.
\end{itemize}

\section*{Research Expertise and Technical Preparation}
Rezaali's research portfolio emphasizes data-driven modeling of high-dimensional environmental systems. He has worked with large gridded climate and atmospheric datasets, including daily temperature fields, atmospheric circulation variables, surface and land-state predictors, and environmental-monitoring data. His computational experience includes Python, PyTorch-style deep learning workflows, R, MATLAB, ArcGIS/ArcPy, Linux environments, \LaTeX, NetCDF/GRIB processing, spatial statistics, model evaluation, and visualization.

For the proposed fellowship, this technical preparation is directly relevant to the development of probabilistic graph models on an icosahedral mesh. The planned CFM framework builds on Rezaali's prior experience with ensemble deep learning, residual prediction, cross-validation over historical climate records, and forecast verification against persistence and climatology baselines. His prior climate and environmental applications provide a strong foundation for evaluating whether modern generative learning systems can extract physically meaningful, calibrated predictability from atmospheric precursor fields.

\section*{Products Most Closely Related to the Proposed Project}
\begin{enumerate}
\item Rezaali, M., Jahangir, M. S., Fouladi-Fard, R., and Keellings, D. (2024). An ensemble deep learning approach to spatiotemporal tropospheric ozone forecasting: A case study of Tehran, Iran. \textit{Urban Climate}, 55, 101950.
\item Narayanan, A., Rezaali, M., Bunting, E. L., and Keellings, D. (2025). It's getting hot in here: Spatial impact of humidity on heat wave severity in the US. \textit{Science of The Total Environment}, 963, 178397.
\item Rezaali, M., Quilty, J., and Karimi, A. (2021). Probabilistic urban water demand forecasting using wavelet-based machine learning models. \textit{Journal of Hydrology}, 600, 126358.
\item Rezaali, M., Fouladi-Fard, R., Mojarad, H., Sorooshian, A., Mahdinia, M., et al. (2021). A wavelet-based random forest approach for indoor BTEX spatiotemporal modeling and health risk assessment. \textit{Environmental Science and Pollution Research}, 28, 22522--22535.
\item Rezaali, M., Fouladi-Fard, R., and Karimi, A. (2023). Performance of TANN, NARX, and GMDHT models for urban water demand forecasting: A case study in a residential complex in Qom, Iran. \textit{Avicenna Journal of Environmental Health Engineering}, 10(2), 85--97.
\end{enumerate}

\section*{Other Significant Products}
\begin{enumerate}
\item Rezaali, M., Fouladi-Fard, R., O'Shaughnessy, P., Naddafi, K., and Karimi, A. (2025). Assessment of AERMOD and ADMS for NOx dispersion modeling with a combination of line and point sources. \textit{Stochastic Environmental Research and Risk Assessment}, 1--15.
\item Rezaali, M., Fouladi-Fard, R., and Karimi, A. (2020). Identification of temporal and spatial patterns of river water quality parameters using NLPCA and multivariate statistical techniques. \textit{International Journal of Environmental Science and Technology}, 17, 2977--2994.
\item Rezaali, M., and Fouladi-Fard, R. (2021). Aerosolized SARS-CoV-2 exposure assessment: dispersion modeling with AERMOD. \textit{Journal of Environmental Health Science and Engineering}, 19, 285--293.
\end{enumerate}

\section*{Graduate Training, Mentoring, and Educational Contributions}
Rezaali has integrated research with mentoring and teaching. As an NSF LEAP REU mentor, he helped undergraduate researchers engage with climate and environmental data science, including problem framing, data preparation, model development, interpretation, and communication of results. His guest lectures at the University of Florida introduced students to deep learning applications in climate science and to modeling climate-change impacts on plant distributions, linking technical methods to applied climate questions.

His mentoring philosophy emphasizes practical reproducibility, transparent uncertainty communication, and the responsible use of AI in environmental decision-making. These activities support the broader educational goals of the AGS-PRF by preparing him to lead interdisciplinary research groups that connect atmospheric science, geospatial data science, and public-interest climate services.

\section*{Synergistic and Service Highlights}
Reviewer service includes \textit{Scientific Reports}, \textit{Journal of Hydrology}, and \textit{Springer Nature Applied Sciences}. Teaching and mentoring activities include graduate guest lectures in climate-data science and mentoring of undergraduate and master's students on neural-network applications to climate and environmental hazards. Rezaali's broader professional contributions are unified by a commitment to making AI-based climate prediction more interpretable, reproducible, and useful for communities exposed to extreme heat, drought, and environmental-health risks.
""",
    },
    "current_pending_support_mostafa_rezaali": {
        "header": "Current and Pending Support",
        "title": "Current and Pending (Other) Support",
        "body": r"""
\section*{Current Support}
\textbf{University of Florida Graduate Research Assistantship.} Mostafa Rezaali is currently supported as a graduate research assistant in the Department of Geography, University of Florida. This support is associated with doctoral research and training in climate science, artificial intelligence for climate extremes, hydroclimatology, and spatiotemporal environmental prediction. The assistantship supports activities such as data preparation, model development, literature synthesis, forecast verification, manuscript preparation, and participation in the intellectual life of the department.

\noindent\textbf{Role:} Graduate Research Assistant and Ph.D. candidate.\\
\textbf{Sponsoring organization:} University of Florida.\\
\textbf{Approximate effort:} Graduate research appointment during doctoral training.\\
\textbf{Relationship to AGS-PRF proposal:} The assistantship supports doctoral-stage research and training. It is expected to end before or upon transition to the NSF AGS-PRF fellowship if the fellowship is awarded.

\section*{Pending Support}
\textbf{NSF AGS-PRF Proposal: Probabilistic Prediction of Heat Waves and Flash Droughts Using Physics-Informed Conditional Flow Matching on an Icosahedral Mesh.} This pending proposal requests support for an independent postdoctoral fellowship project focused on subseasonal prediction of heat waves and flash droughts using a GraphCast-style icosahedral mesh model and conditional flow matching. The project will test whether global atmospheric context, teleconnection indices, land-surface fields, and local persistence can be combined in a probabilistic graph framework to improve calibrated forecasts of U.S. heat extremes at extended lead times.

\noindent\textbf{Role:} Proposed Postdoctoral Fellow.\\
\textbf{Proposed host context:} Department of Geography, University of Florida.\\
\textbf{Research focus:} Probabilistic, physics-informed GeoAI for atmospheric and climate extremes.\\
\textbf{Requested support:} Fellowship stipend and fellowship allowance consistent with the AGS-PRF solicitation and Research.gov budget entry.\\
\textbf{Project period:} Two years, with the requested start date entered in Research.gov.

\section*{Other Current or Pending Support}
No other current or pending sponsored research support, in-kind support, or overlapping proposal support is known from the materials available for this application draft. This statement should be reviewed and updated before final submission if any additional proposals, appointments, in-kind resources, consulting commitments, foreign or domestic support, laboratory resources, equipment access, travel support, consulting arrangements, or institutional commitments must be disclosed under NSF requirements.

\section*{In-Kind Contributions and Facilities}
The proposed fellowship relies on normal host-institution research infrastructure, computing access, scholarly mentoring, and professional-development support rather than a separately committed in-kind award. Any formal allocation of computing, data storage, personnel time, travel support, laboratory access, software licenses, or institutional funds beyond standard host support should be added here before final submission if such commitments are made.

\section*{Overlap Statement}
The pending NSF AGS-PRF proposal is intended to support an independent postdoctoral research program in probabilistic subseasonal heat-extreme prediction. It does not duplicate the graduate assistantship support, which is associated with doctoral training and current research activities at the University of Florida. The fellowship project is forward-looking, postdoctoral in scope, and centered on building a CFM-based forecasting framework and professional-development pathway. The graduate assistantship is a current training and research appointment that should not overlap in effort or budget with the fellowship period if the award is made.

\section*{Certification Note}
This draft is prepared from the information available locally for application assembly. NSF may require Current and Pending (Other) Support to be generated and certified through SciENcv. Before final submission, the applicant should review this content against the official NSF disclosure requirements and confirm that all support, in-kind resources, and pending proposals are accurately represented.
""",
    },
    "collaborators_other_affiliations_mostafa_rezaali": {
        "header": "Collaborators and Other Affiliations",
        "title": "Collaborators and Other Affiliations",
        "body": r"""
\section*{Graduate, Postdoctoral, and Thesis Advisors}
\begin{itemize}
\item David Keellings, University of Florida.
\item Ali Karimi, Qom University of Technology.
\end{itemize}

\section*{Principal Collaborators and Co-Authors}
The following individuals are known collaborators or co-authors from the application materials and CV:
\begin{itemize}
\item A. Narayanan, E. L. Bunting, and David Keellings.
\item R. Fouladi-Fard, A. Karimi, M. S. Jahangir, J. Quilty, H. Mojarad, A. Sorooshian, M. Mahdinia, P. O'Shaughnessy, K. Naddafi.
\item M. Farajollahi, M. Fahiminia, N. R. Rahimi, R. Aali, Shahryari, N. Moghadam Yekta, B. Mohammadnezhad, A. Rasouli.
\end{itemize}

\section*{Institutional Affiliations}
\begin{itemize}
\item University of Florida, Gainesville, Florida, United States.
\item Qom University of Technology, Qom, Iran.
\item IAUKHSH, Isfahan, Iran.
\end{itemize}

\section*{Disclosure Note}
This PDF is a readable companion summary prepared from the CV and application materials available locally. NSF normally requires the Collaborators and Other Affiliations information to be prepared using the NSF COA spreadsheet template and uploaded as the Research.gov single-copy COA document. Before final submission, the official COA template should be reviewed to ensure all categories, dates, co-authors within the required lookback window, collaborators, and organizational affiliations are complete.
""",
    },
    "synergistic_activities_mostafa_rezaali": {
        "header": "Synergistic Activities",
        "title": "Synergistic Activities",
        "body": r"""
\section*{Synergistic Activities}
\begin{enumerate}
\item \textbf{Mentoring emerging climate-data scientists.} Served as an NSF LEAP REU mentor in 2025 and co-advised undergraduate and master's-level research using neural-network models for climate and environmental hazards. This work included helping students translate broad climate-risk questions into data workflows, model experiments, uncertainty summaries, and interpretable results.
\item \textbf{Teaching deep learning for climate science.} Delivered invited lectures at the University of Florida on deep learning applications in climate science, including CNN, LSTM, and hybrid ConvLSTM methods for spatiotemporal climate data. These lectures connected core machine-learning concepts to climate impacts, geospatial analysis, and responsible use of AI for environmental prediction.
\item \textbf{Translating AI methods across climate, hydrology, and air quality.} Developed and published machine-learning workflows for tropospheric ozone forecasting, urban water-demand forecasting, air-dispersion modeling, and environmental-health risk assessment. This cross-domain work supports the AGS-PRF goal of broadening atmospheric-science methods through interdisciplinary data science.
\item \textbf{Professional peer review and disciplinary service.} Reviewed manuscripts for journals including \textit{Scientific Reports}, \textit{Journal of Hydrology}, and \textit{Springer Nature Applied Sciences}. This service contributes to rigorous interdisciplinary scholarship at the intersection of climate science, hydrology, geospatial analysis, and environmental modeling.
\item \textbf{Open and reproducible climate-risk workflows.} Built Python, MATLAB, R, and geospatial workflows for large environmental datasets, including NetCDF/GRIB processing, model training, hindcast evaluation, and diagnostic visualization. The proposed fellowship will extend this commitment through transparent evaluation of probabilistic forecasts and reusable workflows for heat-extreme prediction.
\end{enumerate}
""",
    },
}


def write_doc(stem: str, spec: dict) -> Path:
    tex = COMMON_PREAMBLE.replace("%HEADER%", spec["header"]).replace("%TITLE%", spec["title"])
    tex += spec["body"].strip() + "\n\n" + ENDING
    path = OUT / f"{stem}.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def compile_tex(path: Path) -> Path:
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", path.name]
    subprocess.run(cmd, cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    subprocess.run(cmd, cwd=OUT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return path.with_suffix(".pdf")


def main():
    pdfs = []
    for stem, spec in DOCS.items():
        tex_path = write_doc(stem, spec)
        pdfs.append(compile_tex(tex_path))
    print("Generated personnel PDFs:")
    for pdf in pdfs:
        print(pdf)


if __name__ == "__main__":
    main()

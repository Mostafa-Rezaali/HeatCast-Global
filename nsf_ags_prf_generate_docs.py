#!/usr/bin/env python3
"""Generate draft NSF AGS-PRF proposal PDFs from local application notes."""

from pathlib import Path
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    ListFlowable,
    ListItem,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "nsf_ags_prf_uploads"
OUT.mkdir(exist_ok=True)

TITLE = (
    "Postdoctoral Fellowship: AGS-PRF: Probabilistic Prediction of Heat Waves "
    "and Flash Droughts Using Physics-Informed Conditional Flow Matching on an "
    "Icosahedral Mesh"
)

HOST = "University of Florida, Department of Geography"
MENTORS = "David Keellings, University of Florida"


def styles():
    base = getSampleStyleSheet()
    base["Title"].fontName = "Helvetica-Bold"
    base["Title"].fontSize = 13
    base["Title"].leading = 16
    base["Title"].alignment = TA_CENTER
    base["Heading1"].fontName = "Helvetica-Bold"
    base["Heading1"].fontSize = 11
    base["Heading1"].leading = 13
    base["Heading2"].fontName = "Helvetica-Bold"
    base["Heading2"].fontSize = 10
    base["Heading2"].leading = 12
    base["BodyText"].fontName = "Helvetica"
    base["BodyText"].fontSize = 10
    base["BodyText"].leading = 12
    base.add(
        ParagraphStyle(
            name="Small",
            parent=base["BodyText"],
            fontSize=8.8,
            leading=10.4,
        )
    )
    return base


S = styles()


def clean(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def para(text, style="BodyText"):
    return Paragraph(clean(" ".join(text.strip().split())), S[style])


def heading(text, level=1):
    return Paragraph(clean(text), S["Heading1" if level == 1 else "Heading2"])


def bullet_list(items, style="BodyText"):
    return ListFlowable(
        [ListItem(para(item, style), leftIndent=10) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
        bulletFontName="Helvetica",
        bulletFontSize=7,
    )


def save_source(name, blocks):
    path = OUT / f"{name}.md"
    with path.open("w", encoding="utf-8") as f:
        for block in blocks:
            if isinstance(block, tuple):
                kind, value = block
                if kind == "title":
                    f.write(f"# {value}\n\n")
                elif kind == "heading":
                    f.write(f"## {value}\n\n")
                elif kind == "bullet":
                    for item in value:
                        f.write(f"- {item}\n")
                    f.write("\n")
            else:
                f.write(textwrap.fill(block, 100) + "\n\n")
    return path


def make_pdf(name, blocks, top_title=None):
    pdf_path = OUT / f"{name}.pdf"
    md_path = save_source(name, blocks)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    story = []
    if top_title:
        story.extend([Paragraph(clean(top_title), S["Title"]), Spacer(1, 0.12 * inch)])
    for block in blocks:
        if isinstance(block, tuple):
            kind, value = block
            if kind == "title":
                story.extend([Paragraph(clean(value), S["Title"]), Spacer(1, 0.12 * inch)])
            elif kind == "heading":
                story.extend([heading(value), Spacer(1, 0.04 * inch)])
            elif kind == "heading2":
                story.extend([heading(value, 2), Spacer(1, 0.03 * inch)])
            elif kind == "bullet":
                story.extend([bullet_list(value), Spacer(1, 0.05 * inch)])
            elif kind == "pagebreak":
                story.append(PageBreak())
        else:
            story.extend([para(block), Spacer(1, 0.06 * inch)])
    doc.build(story)
    return pdf_path, md_path


project_summary = [
    ("title", "Project Summary"),
    ("heading", "Overview"),
    (
        "Extreme heat and rapidly developing drought are among the most damaging hazards "
        "in the United States, yet their week-3 predictability remains limited because local land-atmosphere "
        "feedbacks interact with remote teleconnection patterns and global circulation anomalies. This "
        "fellowship will develop a probabilistic, physics-guided graph neural forecasting framework for "
        "15-day prediction of CONUS daily maximum temperature anomalies and associated flash-drought risk. "
        "The project builds from my MeshFlowNet/conditional-flow-matching model, a GraphCast-style "
        "icosahedral mesh encoder-processor-decoder trained on PRISM, ERA5, teleconnection indices, "
        "topography, seasonal radiation, soil-moisture, and global circulation fields. The proposed host "
        f"organization is {HOST}. The proposed scientific mentors are {MENTORS}."
    ),
    ("heading", "Intellectual Merit"),
    (
        "The project asks: how much week-3 predictability of U.S. heat extremes is "
        "contained in global circulation, ocean-atmosphere teleconnections, soil-moisture memory, and local "
        "persistence, and can conditional flow models extract this signal with calibrated uncertainty? "
        "Three aims will: (1) extend a deterministic 15-day residual MeshFlowNet into an ensemble "
        "conditional-flow model; (2) diagnose regional and teleconnection-dependent skill using 5-fold "
        "leave-years-out hindcasts for 1981-2023; and (3) benchmark against persistence, climatology, "
        "ridge/teleconnection baselines, and operational S2S reference skill. Preliminary hindcasts over "
        "5,332 MJJAS samples show stitched temporal anomaly correlation of 0.1016 versus 0.0735 for "
        "persistence, motivating a focused investigation of where and why learned graph dynamics add value."
    ),
    ("heading", "Broader Impacts"),
    (
        "Improved week-3 heat-risk guidance can support public-health planning, water "
        "resources management, energy operations, and agricultural preparedness. The fellowship will produce "
        "open, reproducible workflows; publish model diagnostics and derived hindcast products with metadata; "
        "and translate the research into teaching modules on trustworthy AI for climate extremes. I will "
        "mentor undergraduate and REU students in climate-data science, with emphasis on transparent evaluation, "
        "uncertainty, and equitable access to AI tools for environmental risk assessment."
    ),
]


project_description = [
    ("title", TITLE),
    (
        "Motivation and central hypothesis. Heat waves and flash droughts are compound hydroclimate hazards "
        "with strong societal consequences. Their impacts often emerge on decision horizons of two to four "
        "weeks, when emergency managers, utilities, farmers, and public-health agencies still have time to "
        "act but dynamical forecast skill is often weak. Predictability at these leads is not purely local: "
        "it reflects slowly evolving ocean-atmosphere patterns, stationary waves, soil-moisture memory, "
        "radiative seasonality, synoptic persistence, and land-surface feedbacks. My central hypothesis is "
        "that a graph-based conditional flow model, trained directly on high-resolution U.S. temperature "
        "anomalies while conditioned on global circulation and teleconnection states, can reveal and exploit "
        "a measurable but spatially heterogeneous week-3 signal for extreme heat and flash-drought precursors."
    ),
    (
        "Preliminary work. I have implemented MeshFlowNet, a GraphCast-style icosahedral mesh "
        "encoder-processor-decoder for direct 15-day CONUS daily maximum temperature prediction. The model "
        "uses current and lagged local PRISM temperature anomalies, atmospheric and land-surface fields, "
        "topography, latitude, longitude, seasonal insolation, land mask, five teleconnection indices, and "
        "a 59-channel global ERA5/coarse atmospheric context stack. It supports deterministic residual "
        "prediction and probabilistic conditional-flow-matching modes. The current deterministic 5-fold "
        "leave-years-out hindcast covers MJJAS 1981-2023, with each year predicted by a model that did not "
        "train on that year. Across 5,332 samples, stitched model TAC is 0.1016 compared with 0.0735 for "
        "persistence, a +0.0281 improvement. Per-fold validation R2 is about 0.58-0.60, and sample maps show "
        "skillful reconstruction of several high-amplitude heat anomalies. These results are preliminary "
        "rather than a final forecast system, but they show that the model extracts nontrivial anomaly signal "
        "at a difficult lead time."
    ),
    ("heading", "Aim 1: Build a calibrated conditional-flow forecast system for week-3 heat extremes"),
    (
        "The first aim will convert the current deterministic residual model into a probabilistic forecast "
        "system. The deterministic branch predicts y(t+15) as persistence plus a learned residual in one "
        "forward pass. The conditional-flow branch will learn a velocity field from initialization-time state "
        "to the verifying anomaly field, conditioned on teleconnection indices and global atmosphere/ocean "
        "fields. Ensembles will be generated by sampling the flow and by training multiple seeds per cross-"
        "validation fold. Calibration will be assessed with CRPS, reliability diagrams for exceedance of "
        "local 90th and 95th percentiles, rank histograms, and spatially stratified spread-skill relationships. "
        "This aim will test whether conditional flow matching can deliver useful uncertainty rather than only "
        "a sharper deterministic map."
    ),
    ("heading", "Aim 2: Diagnose sources of subseasonal predictability across regions and regimes"),
    (
        "The second aim will use controlled ablations and interpretable diagnostics to determine which "
        "features carry week-3 skill. I will compare models trained with: local persistence only; local "
        "meteorology plus static/seasonal terms; teleconnection indices; global ERA5 fields; and the full "
        "conditioning stack. The analysis will produce per-pixel TAC, skill over persistence, regional skill "
        "summaries for NOAA climate regions, block-bootstrap significance tests by year, and event composites "
        "for strong teleconnection regimes. The scientific target is not only a higher metric, but a physical "
        "account of where AI forecast skill is consistent with known pathways such as Pacific-North American "
        "wave trains, moisture transport, soil-moisture feedback, and subtropical ridge variability."
    ),
    ("heading", "Aim 3: Extend from heat prediction to flash-drought-relevant risk products"),
    (
        "The third aim will connect temperature-anomaly prediction to flash-drought precursors by evaluating "
        "rapid intensification of heat and atmospheric demand in conjunction with antecedent soil-moisture "
        "state. The model already ingests soil moisture, humidity, winds, sea-level pressure, radiation, and "
        "large-scale circulation fields. I will evaluate whether the learned representations anticipate "
        "co-occurring hot, dry, high-demand regimes over agriculturally important regions. Products will be "
        "reported as calibrated probabilistic exceedance maps, regional time series, and case studies rather "
        "than as deterministic binary declarations. This will keep the science focused on predictability and "
        "uncertainty while making the outcomes relevant to water and agricultural decision contexts."
    ),
    ("heading", "Research design and methods"),
    (
        "Data. The hindcast system will use PRISM daily maximum temperature over CONUS at 0.04-degree "
        "resolution, ERA5 atmospheric variables and global context fields, teleconnection indices, topography, "
        "latitude, longitude, land mask, top-of-atmosphere insolation, and seasonal encodings. The current "
        "local input stack contains 19 channels: current and two lagged PRISM temperature fields; geopotential, "
        "soil moisture, sea-level pressure, 2-m temperature, 850-hPa humidity, 850-hPa temperature and winds, "
        "300-hPa geopotential; topography; latitude and longitude; day-of-year sine and cosine; insolation; "
        "and land mask. The global context stack contains 59 channels including SST, OLR, winds, geopotential, "
        "temperature, humidity, surface pressure, sea-level pressure, and total column water vapor across "
        "multiple levels."
    ),
    (
        "Model architecture. MeshFlowNet maps the CONUS grid to an icosahedral mesh, performs message passing "
        "with graph interaction networks and per-round FiLM conditioning, and decodes back to the land grid. "
        "This representation preserves global-to-regional context while avoiding purely convolutional locality. "
        "The probabilistic branch will use continuous-time embeddings and conditional flow matching; the "
        "deterministic branch will remain as a transparent baseline and an efficient ablation."
    ),
    (
        "Evaluation. All claims will be based on leave-years-out cross-validation over 1981-2023. Metrics will "
        "include temporal anomaly correlation, spatial anomaly R2, MAE, MSE skill relative to persistence, CRPS, "
        "reliability of extreme exceedance probabilities, and block-bootstrap confidence intervals. Baselines "
        "will include climatology, persistence, ridge regression on teleconnection/global summary features, and "
        "published operational S2S reference skill where the metric and domain are comparable. I will report "
        "negative or regionally limited results explicitly because understanding where AI models fail is essential "
        "for trustworthy climate-risk use."
    ),
    ("heading", "Intellectual Merit"),
    (
        "The intellectual merit is the integration of three threads that are usually separated: subseasonal "
        "predictability theory, graph neural weather models, and probabilistic flow matching. The project will "
        "advance knowledge by quantifying when global and land-surface information improves 15-day U.S. heat "
        "prediction beyond persistence; by producing spatially explicit diagnostics of learned teleconnection "
        "skill; and by testing whether conditional-flow models can provide calibrated uncertainty for regional "
        "extreme-temperature risk. The outcome will be a physically evaluated AI framework rather than a black-box "
        "score table."
    ),
    ("heading", "Broader Impacts"),
    (
        "The broader impacts center on usable, transparent climate-risk intelligence and training. First, the "
        "project will produce reproducible code, derived hindcast diagnostics, and documentation suitable for "
        "students and practitioners who need to understand AI forecast limitations. Second, I will develop a "
        "short teaching module on graph neural weather prediction and uncertainty for graduate climate modeling "
        "or environmental data-science courses, building from my invited lectures at the University of Florida. "
        "Third, I will mentor undergraduate and REU students in open climate-data workflows, with attention to "
        "students entering atmospheric science from geography, environmental engineering, and data science. "
        "Fourth, the forecast diagnostics will be framed around heat-health, water, energy, and agriculture use "
        "cases where calibrated uncertainty can be more valuable than a single deterministic forecast."
    ),
    ("heading", "Host institution, mentors, and professional development"),
    (
        "The proposed host institution is the University of Florida Department of Geography. I have been at UF "
        "for more than 12 months, and remaining at UF is scientifically justified by the direct fit between the "
        "project and UF strengths in climatology, extreme events, artificial intelligence, geospatial analysis, "
        "and high-performance computing. The choice will support a new postdoctoral research direction because "
        "the proposed fellowship shifts my work from model development during doctoral training toward an "
        "independent program in probabilistic subseasonal predictability, uncertainty quantification, and open "
        "forecast diagnostics. David Keellings will provide host mentoring in heat-wave climatology, climate "
        "extremes, and professional development. The mentoring plan will include regular research "
        "meetings, manuscript and proposal-development milestones, conference presentations, student-mentoring "
        "opportunities, and explicit preparation for independent faculty or research-scientist positions."
    ),
    ("heading", "Facilities and resources"),
    (
        "The project will use UF research computing resources, Python/PyTorch software environments, geospatial "
        "and climate-data analysis tools, and publicly available PRISM/ERA5-derived data workflows. The local "
        "codebase already runs on GPU-equipped high-performance computing systems and includes scripts for "
        "training, hindcast export, per-year bootstrap statistics, baseline comparison, TAC maps, sample maps, "
        "and CRPS/reliability analysis. The Project Description documents all substantive facilities and "
        "resources; the separate Facilities, Equipment and Other Resources upload states: See the Project "
        "Description, as requested by the AGS-PRF solicitation."
    ),
    ("heading", "Timeline and milestones"),
    (
        "Months 1-6: finalize data audit, deterministic baseline, ridge/climatology/persistence baselines, and "
        "per-pixel skill maps. Months 7-12: train probabilistic conditional-flow ensembles, compute CRPS and "
        "reliability diagnostics, and prepare the first manuscript on week-3 heat predictability. Months 13-18: "
        "complete teleconnection-regime ablations, block-bootstrap significance tests, and flash-drought-relevant "
        "risk diagnostics. Months 19-24: release reproducible workflows and derived hindcast products, submit "
        "a second manuscript, present results at AGU/AMS or comparable meetings, and complete career-development "
        "milestones with both mentors."
    ),
    ("heading", "Assessment of success"),
    (
        "Success will be assessed by scientific and broader-impact outputs: (1) a documented cross-validated "
        "hindcast benchmark with uncertainty metrics; (2) at least two manuscripts, one on model/predictability "
        "and one on physical-regime interpretation or flash-drought-relevant risk; (3) public release of code, "
        "metadata, and nonrestricted derived products; (4) student mentoring and teaching materials; and (5) a "
        "clear professional transition from doctoral model development to an independent research program in "
        "AI-enabled atmospheric predictability."
    ),
]


references = [
    ("title", "References Cited"),
    "Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., Merose, A., Hoyer, S., Holland, G., Vinyals, O., Stott, J., Pritzel, A., Mohamed, S., and Battaglia, P. (2023). Learning skillful medium-range global weather forecasting. Science, 382(6677), 1416-1421. https://doi.org/10.1126/science.adi2336",
    "Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T. R., El-Kadi, A., Masters, D., Ewalds, T., Stott, J., Mohamed, S., Battaglia, P., Lam, R., and Willson, M. (2024). Probabilistic weather forecasting with machine learning. Nature, 637, 84-90. https://doi.org/10.1038/s41586-024-08252-9",
    "Pegion, K., Kirtman, B. P., Becker, E., Collins, D. C., LaJoie, E., Burgman, R., Bell, R., DelSole, T., Min, D., Zhu, Y., Li, W., Sinsky, E., Guan, H., Gottschalck, J., Metzger, E. J., Barton, N. P., Achuthavarier, D., Marshak, J., Koster, R., Lin, H., Gagnon, N., Bell, M., Tippett, M. K., Robertson, A. W., Sun, S., Benjamin, S. G., Green, B. W., Bleck, R., and Kim, H. (2019). The Subseasonal Experiment (SubX): A multimodel subseasonal prediction experiment. Bulletin of the American Meteorological Society, 100(10), 2043-2060. https://doi.org/10.1175/BAMS-D-18-0270.1",
    "Vitart, F., Ardilouze, C., Bonet, A., Brookshaw, A., Chen, M., Codorean, C., Deque, M., Ferranti, L., Fucile, E., Fuentes, M., Hendon, H., Hodgson, J., Kang, H. S., Kumar, A., Lin, H., Liu, G., Liu, X., Malguzzi, P., Mallas, I., Manoussakis, M., Mastrangelo, D., MacLachlan, C., McLean, P., Minami, A., Mladek, R., Nakazawa, T., Najm, S., Nie, Y., Rixen, M., Robertson, A. W., Ruti, P., Sun, C., Takaya, Y., Tolstykh, M., Venuti, F., Waliser, D., Woolnough, S., Wu, T., Won, D. J., Xiao, H., Zaripov, R., and Zhang, L. (2017). The Sub-seasonal to Seasonal (S2S) Prediction Project Database. Bulletin of the American Meteorological Society, 98(1), 163-173. https://doi.org/10.1175/BAMS-D-16-0017.1",
    "White, C. J., Carlsen, H., Robertson, A. W., Klein, R. J. T., Lazo, J. K., Kumar, A., Vitart, F., Coughlan de Perez, E., Ray, A. J., Murray, V., Bharwani, S., MacLeod, D., James, R., Fleming, L., Morse, A. P., Eggen, B., Graham, R., Kjellstrom, E., Becker, E., Pegion, K. V., Holbrook, N. J., McEvoy, D., Depledge, M., Perkins-Kirkpatrick, S., Brown, T. J., Street, R., Jones, L., Remenyi, T. A., Hodgson-Johnston, I., Buontempo, C., Lamb, R., Meinke, H., Arheimer, B., and Zebiak, S. E. (2017). Potential applications of subseasonal-to-seasonal (S2S) predictions. Meteorological Applications, 24(3), 315-325. https://doi.org/10.1002/met.1654",
    "Otkin, J. A., Anderson, M. C., Hain, C., Svoboda, M., Johnson, D., Mueller, R., Tadesse, T., Wardlow, B., and Brown, J. (2016). Assessing the evolution of soil moisture and vegetation conditions during the 2012 United States flash drought. Agricultural and Forest Meteorology, 218-219, 230-242. https://doi.org/10.1016/j.agrformet.2015.12.065",
    "Rasp, S., Dueben, P. D., Scher, S., Weyn, J. A., Mouatadid, S., and Thuerey, N. (2020). WeatherBench: A benchmark data set for data-driven weather forecasting. Journal of Advances in Modeling Earth Systems, 12, e2020MS002203. https://doi.org/10.1029/2020MS002203",
    "Rezaali, M., Jahangir, M. S., Fouladi-Fard, R., and Keellings, D. (2024). An ensemble deep learning approach to spatiotemporal tropospheric ozone forecasting: A case study of Tehran, Iran. Urban Climate, 55, 101950.",
    "Rezaali, M., Quilty, J., and Karimi, A. (2021). Probabilistic urban water demand forecasting using wavelet-based machine learning models. Journal of Hydrology, 600, 126358.",
]


budget = [
    ("title", "Budget Justification"),
    (
        "This AGS-PRF fellowship budget follows the solicitation-specified fellowship structure. The budget "
        "contains stipend support for the Fellow and a fellowship allowance. No voluntary committed cost "
        "sharing is included."
    ),
    ("heading", "Year 1"),
    (
        "Stipend: $70,000. The stipend will support full-time fellowship effort by Mostafa Rezaali on the "
        "research, broader-impact, and professional-development activities described in the Project Description."
    ),
    (
        "Fellowship allowance: $30,000. Planned uses include health insurance and benefit-related costs; "
        "high-performance computing, cloud storage, and secure data-storage expenses; workstation or peripheral "
        "support needed for model development and analysis; conference and workshop travel to present results "
        "and receive feedback from the atmospheric-science community; open-access publication and data-archive "
        "fees; and modest software, training, and dissemination costs directly related to the fellowship."
    ),
    ("heading", "Year 2"),
    (
        "Stipend: $72,000. The stipend will support full-time fellowship effort during the second year, including "
        "probabilistic ensemble training, uncertainty-calibration analyses, manuscripts, open-science release, "
        "student mentoring, and career-development activities."
    ),
    (
        "Fellowship allowance: $30,000. Planned uses mirror Year 1 and will support continued health/benefit "
        "costs, computing and storage, conference participation, publication and data dissemination, and project-"
        "specific supplies or software. The allowance will be administered responsibly by the Fellow in accordance "
        "with NSF and AGS-PRF requirements."
    ),
]


facilities = [
    ("title", "Facilities, Equipment and Other Resources"),
    "See the Project Description.",
]


dmp = [
    ("title", "Data Management Plan"),
    (
        "Products of research. The project will produce source code, trained-model configuration files, "
        "evaluation scripts, derived hindcast diagnostics, summary tables, selected nonrestricted figures, "
        "metadata, and manuscripts. The project will use third-party climate datasets such as PRISM and ERA5; "
        "those data will not be redistributed if license or provider terms require users to obtain them from "
        "the original sources."
    ),
    (
        "Data and code formats. Code will be released in plain-text Python and shell scripts with environment "
        "files where possible. Derived data products will be released in standard scientific formats such as "
        "NetCDF, NPZ, CSV, or GeoTIFF as appropriate, with README files describing variables, units, grids, "
        "time coverage, cross-validation splits, and known limitations."
    ),
    (
        "Access and sharing. Nonrestricted derived products and code will be shared through a public repository "
        "and archived with a persistent DOI through a repository such as Zenodo or an institutional archive. "
        "Large intermediate files and third-party raw data will be documented with scripts and instructions so "
        "that authorized users can reproduce them from original providers."
    ),
    (
        "Preservation. Final code, metadata, manuscripts, and derived data needed to reproduce major figures "
        "and tables will be preserved for at least five years after the end of the award. Versioned releases "
        "will be used for manuscript-associated artifacts."
    ),
    (
        "Privacy, security, and ethical considerations. The project uses environmental and climate data, not "
        "human-subjects data. No personally identifiable information will be collected. Model limitations, "
        "uncertainty, and appropriate-use caveats will be documented to reduce the risk of overinterpreting "
        "experimental forecast products."
    ),
    (
        "Roles and responsibilities. The Fellow will be responsible for maintaining the repository, preparing "
        "metadata, and archiving final products. Mentors will advise on scientific quality control, reproducible "
        "workflow design, and appropriate dissemination."
    ),
]


synergistic = [
    ("title", "Synergistic Activities - Mostafa Rezaali"),
    (
        "1. Climate-AI teaching: Delivered invited graduate-level instruction on deep learning applications in "
        "climate science, including CNN, LSTM, and hybrid convLSTM approaches for spatiotemporal climate data."
    ),
    (
        "2. Interdisciplinary mentoring: Mentored and advised students on neural-network applications to climate, "
        "dust, air-quality, and environmental-health problems, including NSF LEAP REU mentoring."
    ),
    (
        "3. Open and reproducible climate-data workflows: Developed Python/PyTorch workflows for NetCDF, GRIB, "
        "PRISM, ERA5, and geospatial datasets, with emphasis on transparent validation, cross-validation, and "
        "model diagnostics."
    ),
    (
        "4. Peer-review service: Served as a journal reviewer for Scientific Reports, Journal of Hydrology, "
        "Springer Nature Applied Sciences, and other journals."
    ),
    (
        "5. Science communication across fields: Built a publication record connecting climate science, "
        "environmental engineering, air-quality modeling, hydrology, public health, and machine learning, "
        "supporting transfer of methods across environmental domains."
    ),
]


phd_abstract = [
    ("title", "Ph.D. Abstract"),
    (
        "My doctoral research in Climate Science and Geography at the University of Florida focuses on artificial "
        "intelligence for extreme weather, especially heat waves and flash drought. The work develops and evaluates "
        "deep learning methods for spatiotemporal climate prediction, with attention to large-scale atmospheric "
        "drivers, local land-surface feedbacks, and forecast verification. A major component is MeshFlowNet, a "
        "GraphCast-style graph neural network on an icosahedral mesh for direct 15-day prediction of CONUS daily "
        "maximum temperature anomalies. The model combines local PRISM temperature history, ERA5 atmospheric and "
        "land-surface variables, teleconnection indices, global circulation fields, topography, seasonal radiation, "
        "and land-mask information. My dissertation research uses cross-validated hindcasts, temporal anomaly "
        "correlation, uncertainty metrics, and baseline comparisons to determine where learned forecast models add "
        "value over climatology and persistence. The broader goal is to improve physically interpretable and "
        "uncertainty-aware climate-risk prediction for heat, water, energy, agriculture, and public-health planning."
    ),
]


host_letter_template = [
    ("title", "Host Institution Letter Template"),
    (
        "This is a template only. Do not submit without review, institutional approval, and signatures from the "
        "scientific mentor and department chair or equivalent."
    ),
    (
        f"If the proposal submitted by Mostafa Rezaali entitled \"{TITLE}\" is selected for funding by NSF, "
        "it is our intent to host the Fellow at the University of Florida Department of Geography and provide "
        "the facilities, mentoring, and institutional support described in the Project Description and "
        "Facilities, Equipment and Other Resources section."
    ),
    (
        "This letter should certify that the proposal has been read and approved by the proposed scientific "
        "mentor(s), that adequate facilities and support will be provided for the Fellow, and that the mentoring "
        "plan describes the role the mentor(s) will play in professional development and the training/research "
        "opportunities available at the host institution. The AGS-PRF solicitation states that this should not "
        "be a letter of recommendation."
    ),
    "Scientific mentor signature: ____________________________",
    "Department chair or equivalent signature: ____________________________",
]


sciencv_notes = [
    ("title", "SciENcv Source Notes for Certified Personnel Documents"),
    (
        "NSF requires the Biographical Sketch and Current and Pending (Other) Support documents to be generated "
        "and certified in SciENcv. The notes below are source material only; they are not a substitute for the "
        "certified SciENcv PDFs."
    ),
    ("heading", "Biographical Sketch Source"),
    (
        "Professional preparation: Ph.D. candidate in Climate Science/Geography, University of Florida, Aug "
        "2022-present; Graduate Certificate in Atmospheric Sciences, University of Florida; M.Sc. Civil and "
        "Environmental Engineering, Qom University of Technology, 2018; B.Sc. Civil and Environmental Engineering, "
        "IAUKHSH, 2016."
    ),
    (
        "Appointments: Graduate Research Assistant, University of Florida, 2022-present; invited lecturer and "
        "student mentor roles as listed in CV."
    ),
    (
        "Products to consider: Rezaali et al. 2024 Urban Climate; Rezaali et al. 2021 Journal of Hydrology; "
        "Rezaali et al. 2025 Stochastic Environmental Research and Risk Assessment; Rezaali et al. 2021 "
        "Environmental Science and Pollution Research; Narayanan et al. 2025 Science of the Total Environment."
    ),
    ("heading", "Current and Pending Support Source"),
    (
        "Include this NSF AGS-PRF proposal as pending. Add all current appointments, assistantships, fellowships, "
        "applications, in-kind resources, and any current or planned proposals that must be disclosed. Confirm "
        "whether the NSF LEAP Momentum Fellowship is past or current. Confirm any other pending postdoctoral "
        "applications."
    ),
    ("heading", "COA Source"),
    (
        "Use the official NSF Collaborators and Other Affiliations template. Coauthors from the CV include, at "
        "minimum: R. Fouladi-Fard, P. O'Shaughnessy, K. Naddafi, A. Karimi, M. S. Jahangir, D. Keellings, "
        "J. Quilty, H. Mojarad, A. Sorooshian, M. Mahdinia, N. Moghadam Yekta, B. Mohammadnezhad, A. Rasouli, "
        "A. Narayanan, E. L. Bunting, M. Farajollahi, M. Fahiminia, N. R. Rahimi, R. Aali, and A. Shahryari. "
        "Verify affiliations and 48-month collaborator windows before submission."
    ),
]


def main():
    outputs = []
    for name, blocks in [
        ("project_summary", project_summary),
        ("project_description", project_description),
        ("references_cited", references),
        ("budget_justification", budget),
        ("facilities_equipment_other_resources", facilities),
        ("data_management_plan", dmp),
        ("synergistic_activities_mostafa_rezaali", synergistic),
        ("phd_abstract", phd_abstract),
        ("host_institution_letter_template_do_not_submit_unsigned", host_letter_template),
        ("sciencv_personnel_source_notes", sciencv_notes),
    ]:
        outputs.append(make_pdf(name, blocks))
    print("Generated files:")
    for pdf, md in outputs:
        print(f"- {pdf}")
        print(f"  source: {md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate model-configuration and input-variable LaTeX tables."""

from pathlib import Path
import ast


OUTPUT_DIR = Path("paper_figures/tables")
EXTENDED_GLOBAL_VARIABLES_TXT = Path("data_cache/extended_global_variables.txt")

BASE_GLOBAL_VARIABLES = [
    "sst", "olr", "geopotential_200", "u_wind_200",
    "total_column_water_vapour", "v_wind_200", "geopotential_500",
    "temperature_850", "temperature_2m_global",
]

LOCAL_INPUTS = [
    ("1", "T2max(t)", "PRISM", "0.25 deg", "surface"),
    ("2", "T2max(t-1)", "PRISM", "0.25 deg", "surface"),
    ("3", "T2max(t-2)", "PRISM", "0.25 deg", "surface"),
    ("4", "Geopotential", "training NetCDF", "0.25 deg", "local"),
    ("5", "Soil moisture", "training NetCDF", "0.25 deg", "local"),
    ("6", "Sea-level pressure", "training NetCDF", "0.25 deg", "local"),
    ("7", "2m temperature", "training NetCDF", "0.25 deg", "local"),
    ("8", "850-hPa specific humidity", "training NetCDF", "0.25 deg", "850 hPa"),
    ("9", "850-hPa temperature", "training NetCDF", "0.25 deg", "850 hPa"),
    ("10", "850-hPa u wind", "training NetCDF", "0.25 deg", "850 hPa"),
    ("11", "850-hPa v wind", "training NetCDF", "0.25 deg", "850 hPa"),
    ("12", "300-hPa geopotential", "training NetCDF", "0.25 deg", "300 hPa"),
    ("13", "Topography", "ETOPO2022", "0.25 deg", "surface"),
    ("14", "Latitude", "computed", "0.25 deg", "static"),
    ("15", "Longitude", "computed", "0.25 deg", "static"),
    ("16", "DOY sine", "computed", "0.25 deg", "seasonal"),
    ("17", "DOY cosine", "computed", "0.25 deg", "seasonal"),
    ("18", "TOA insolation", "computed", "0.25 deg", "daily"),
    ("19", "Land mask", "PRISM valid mask", "0.25 deg", "static"),
]


def read_extended_globals(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    marker = "NEW_GLOBAL_VARIABLES"
    if marker not in text:
        return []
    start = text.index(marker)
    lo = text.index("[", start)
    hi = text.index("]", lo) + 1
    return ast.literal_eval(text[lo:hi])


def tex_escape(s):
    return str(s).replace("_", "\\_")


def write_table1():
    rows = [
        ("Architecture", "MeshFlowNet, icosahedral mesh GNN"),
        ("Total parameters", "approximately 4.4M plus grid refiner"),
        ("Mesh refinement level", "7, regional 8608-node CONUS mesh"),
        ("Processor rounds", "8"),
        ("Latent dimension", "128"),
        ("Input channels, local", "19"),
        ("Input channels, global", "59 ERA5 variables x 4 lags = 236"),
        ("Global lags", "t, t-3, t-7, t-14"),
        ("Target", "PRISM T2max daily z-score"),
        ("Lead time", "15 days, direct single forward pass"),
        ("Grid resolution", "0.25 deg, 621 x 1405"),
        ("Cross-validation", "5-fold leave-k-years-out"),
        ("Optimizer", "AdamW, LR=5e-5, weight decay=0.01"),
        ("Dropout", "0.15"),
        ("Early stopping", "validation true weekly7 TAC patience 5 after epoch 12"),
        ("Training hardware", "8x NVIDIA B200 GPUs"),
    ]
    with open(OUTPUT_DIR / "table1_model_config.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{ll}\\hline\n")
        f.write("Parameter & Value \\\\\\hline\n")
        for k, v in rows:
            f.write(f"{tex_escape(k)} & {tex_escape(v)} \\\\\n")
        f.write("\\hline\\end{tabular}\n")


def write_table_s1():
    globals_all = BASE_GLOBAL_VARIABLES + read_extended_globals(EXTENDED_GLOBAL_VARIABLES_TXT)
    with open(OUTPUT_DIR / "tableS1_input_variables.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lllll}\\hline\n")
        f.write("\\# & Variable & Source & Resolution & Level \\\\\\hline\n")
        for row in LOCAL_INPUTS:
            f.write(" & ".join(tex_escape(x) for x in row) + " \\\\\n")
        start = len(LOCAL_INPUTS) + 1
        for i, name in enumerate(globals_all, start=start):
            f.write(f"{i} & {tex_escape(name)} & ERA5 & 1 deg & global lagged \\\\\n")
        f.write("\\hline\\end{tabular}\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_table1()
    write_table_s1()


if __name__ == "__main__":
    main()

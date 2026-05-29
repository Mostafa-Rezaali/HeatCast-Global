from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageEnhance


OUT_DIR = Path(r"C:\Users\mosta\OneDrive\Desktop\GraphCast\nsf_ags_prf_uploads\final_format\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "figure_study_area_conus.png"
COUNTRIES = OUT_DIR / "ne_50m_admin_0_countries.geojson"
STATES = OUT_DIR / "us_states.geojson"


def download(url: str, path: Path) -> None:
    if not path.exists():
        urllib.request.urlretrieve(url, path)


download(
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson",
    COUNTRIES,
)
download(
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
    STATES,
)


W, H = 1500, 820
LON_MIN, LON_MAX = -128.5, -63.5
LAT_MIN, LAT_MAX = 24.0, 50.8
ZOOM = 5


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    n = 2**z
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def lonlat_to_webmerc(lon: float, lat: float, z: int) -> tuple[float, float]:
    x, y = lonlat_to_tile(lon, lat, z)
    return x * 256, y * 256


top_left = lonlat_to_webmerc(LON_MIN, LAT_MAX, ZOOM)
bottom_right = lonlat_to_webmerc(LON_MAX, LAT_MIN, ZOOM)
tile_x0 = math.floor(top_left[0] / 256)
tile_y0 = math.floor(top_left[1] / 256)
tile_x1 = math.floor(bottom_right[0] / 256)
tile_y1 = math.floor(bottom_right[1] / 256)

canvas = Image.new("RGB", ((tile_x1 - tile_x0 + 1) * 256, (tile_y1 - tile_y0 + 1) * 256), "white")
for tx in range(tile_x0, tile_x1 + 1):
    for ty in range(tile_y0, tile_y1 + 1):
        tile_path = OUT_DIR / f"esri_relief_z{ZOOM}_{tx}_{ty}.jpg"
        if not tile_path.exists():
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{ZOOM}/{ty}/{tx}"
            try:
                urllib.request.urlretrieve(url, tile_path)
            except Exception:
                continue
        try:
            tile = Image.open(tile_path).convert("RGB")
            canvas.paste(tile, ((tx - tile_x0) * 256, (ty - tile_y0) * 256))
        except Exception:
            pass

crop_left = int(top_left[0] - tile_x0 * 256)
crop_top = int(top_left[1] - tile_y0 * 256)
crop_right = int(bottom_right[0] - tile_x0 * 256)
crop_bottom = int(bottom_right[1] - tile_y0 * 256)
base = canvas.crop((crop_left, crop_top, crop_right, crop_bottom)).resize((W, H), Image.Resampling.LANCZOS)
base = ImageEnhance.Color(base).enhance(0.78)
base = ImageEnhance.Contrast(base).enhance(1.08)
base = ImageEnhance.Brightness(base).enhance(1.05)

overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(overlay)


def proj(lon: float, lat: float) -> tuple[float, float]:
    x, y = lonlat_to_webmerc(lon, lat, ZOOM)
    px = (x - top_left[0]) / (bottom_right[0] - top_left[0]) * W
    py = (y - top_left[1]) / (bottom_right[1] - top_left[1]) * H
    return px, py


try:
    title_font = ImageFont.truetype("arialbd.ttf", 24)
    font = ImageFont.truetype("arial.ttf", 20)
    small = ImageFont.truetype("arial.ttf", 17)
except Exception:
    title_font = font = small = None

# Lat/lon grid.
for lon in [-120, -110, -100, -90, -80, -70]:
    x, _ = proj(lon, LAT_MIN)
    d.line([(x, 0), (x, H)], fill=(255, 255, 255, 110), width=1)
    d.line([(x, 0), (x, H)], fill=(30, 30, 30, 45), width=1)
    d.text((x - 28, H - 36), f"{abs(lon)}°W", fill=(35, 35, 35, 255), font=font)
for lat in [25, 30, 35, 40, 45]:
    _, y = proj(LON_MIN, lat)
    d.line([(0, y), (W, y)], fill=(255, 255, 255, 110), width=1)
    d.line([(0, y), (W, y)], fill=(30, 30, 30, 45), width=1)
    d.text((12, y - 12), f"{lat}°N", fill=(35, 35, 35, 255), font=font)

# State boundaries, low contrast.
states = json.loads(STATES.read_text(encoding="utf-8"))
for feat in states["features"]:
    name = feat["properties"].get("name", "")
    if name in {"Alaska", "Hawaii", "Puerto Rico"}:
        continue
    geom = feat["geometry"]
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for poly in polys:
        pts = [proj(lon, lat) for lon, lat in poly[0]]
        if len(pts) > 2:
            d.line(pts + [pts[0]], fill=(55, 55, 55, 70), width=1)

# US mainland/coastline outline only, in red.
countries = json.loads(COUNTRIES.read_text(encoding="utf-8"))
usa = next(f for f in countries["features"] if f["properties"].get("ADMIN") == "United States of America")
geom = usa["geometry"]
polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
for poly in polys:
    exterior = poly[0]
    pts = []
    for lon, lat in exterior:
        if LON_MIN - 2 <= lon <= LON_MAX + 2 and LAT_MIN - 2 <= lat <= LAT_MAX + 2:
            pts.append(proj(lon, lat))
        else:
            if len(pts) > 2:
                d.line(pts, fill=(255, 0, 0, 255), width=4)
            pts = []
    if len(pts) > 2:
        d.line(pts, fill=(255, 0, 0, 255), width=4)

# Frame, title, legend, labels.
d.rectangle([0, 0, W - 1, H - 1], outline=(40, 40, 40, 255), width=2)
title = "The Proposed Study Area of the Contiguous United States"
tw = d.textlength(title, font=title_font)
d.rectangle([W / 2 - tw / 2 - 12, 4, W / 2 + tw / 2 + 12, 37], fill=(245, 245, 245, 190))
d.text((W / 2 - tw / 2, 8), title, fill=(0, 0, 0, 255), font=title_font)

d.rectangle([W - 145, 30, W - 25, 70], fill=(255, 255, 255, 232), outline=(45, 45, 45, 230), width=1)
d.line([(W - 126, 51), (W - 86, 51)], fill=(255, 0, 0, 255), width=4)
d.text((W - 77, 41), "CONUS", fill=(0, 0, 0, 255), font=small)

d.text((W / 2 - 52, H - 24), "Longitude", fill=(35, 35, 35, 255), font=font)
lat_label = Image.new("RGBA", (130, 35), (255, 255, 255, 0))
ld = ImageDraw.Draw(lat_label)
ld.text((0, 0), "Latitude", fill=(35, 35, 35, 255), font=font)
lat_label = lat_label.rotate(90, expand=True)
overlay.alpha_composite(lat_label, (2, H // 2 - 68))

result = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
result.save(OUT, quality=95)
print(OUT)

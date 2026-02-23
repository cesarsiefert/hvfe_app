import streamlit as st
import geopandas as gpd
import zipfile, os, tempfile, json
import ee, geemap
import numpy as np
import matplotlib.pyplot as plt

# Inicializar Earth Engine
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# Configurações
names = [
    'surface water',
    'regularly flooded wetlands',
    'headwater regions',
    'fixed‐width buffer (low‐order streams)',
    'fixed‐width buffer (surface water and high‐order streams)',
    'geomorphic floodplains',
    'fixed‐width buffer (regularly flooded wetlands)'
]
palette = ['65cbd6', '20456e', '206e35', 'f4fc08', 'fc0808', 'd69a65', 'd665b8']

min_tiles = [
    "projects/ee-vsgriffey/assets/riparian_full_min_tile1_V2",
    "projects/ee-vsgriffey/assets/riparian_full_min_tile2_V2",
    "projects/ee-vsgriffey/assets/riparian_full_min_tile3_V2",
    "projects/ee-vsgriffey/assets/riparian_full_min_tile4_V2",
    "projects/ee-vsgriffey/assets/riparian_full_min_tile5_V2",
    "projects/ee-vsgriffey/assets/riparian_full_min_tile6_V2",
    "projects/ee-vsgriffey/assets/riparian_full_min_tile7_V2",
    "projects/ee-vsgriffey/assets/riparian_full_min_tile8_V2",
    "projects/ee-vsgriffey/assets/riparian_full_min_tile9_V2",
]
max_tiles = [
    "projects/ee-vsgriffey/assets/riparian_max_tile1_V2",
    "projects/ee-vsgriffey/assets/riparian_max_tile2_V2",
    "projects/ee-vsgriffey/assets/riparian_max_tile3_V2",
    "projects/ee-vsgriffey/assets/riparian_max_tile4_V2",
    "projects/ee-vsgriffey/assets/riparian_max_tile5_V2",
    "projects/ee-vsgriffey/assets/riparian_max_tile6_V2",
    "projects/ee-vsgriffey/assets/riparian_max_tile7_V2",
    "projects/ee-vsgriffey/assets/riparian_max_tile8_V2",
    "projects/ee-vsgriffey/assets/riparian_max_tile9_V2",
]

fwcVis = {"min": 1, "max": 7, "palette": palette}

def get_hvfe_image(scenario: str) -> ee.Image:
    tiles = min_tiles if scenario == "MIN" else max_tiles
    ic = ee.ImageCollection([ee.Image(p) for p in tiles]).mosaic()
    return ic.rename("class").toInt16()

def upload_and_read_shapefile_zip(uploaded_file) -> gpd.GeoDataFrame:
    tmpdir = tempfile.mkdtemp(prefix="shp_upload_")
    zip_path = os.path.join(tmpdir, "uploaded.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.read())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmpdir)
    shp_path = None
    for root, _, files_ in os.walk(tmpdir):
        for fn in files_:
            if fn.lower().endswith(".shp"):
                shp_path = os.path.join(root, fn)
                break
        if shp_path: break
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf.dissolve()

def gdf_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    geojson = json.loads(gdf.to_json())
    geom = geojson["features"][0]["geometry"]
    ee_geom = ee.Geometry(geom)
    return ee.FeatureCollection([ee.Feature(ee_geom)])

def compute_area_by_class(img: ee.Image, region: ee.Geometry, scale: int) -> dict:
    area_img = ee.Image.pixelArea().rename("area").addBands(img.rename("class"))
    reducer = ee.Reducer.sum().group(groupField=1, groupName="class")
    stats = area_img.reduceRegion(
        reducer=reducer,
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True
    )
    groups = ee.List(stats.get("groups"))
    groups_py = groups.getInfo() if groups is not None else []
    out = {}
    for g in groups_py:
        c = int(g["class"])
        a = float(g["sum"])
        out[c] = out.get(c, 0.0) + a
    return out

def plot_pie(area_by_class: dict, scenario: str):
    classes = sorted(area_by_class.keys())
    areas = np.array([area_by_class[c] for c in classes], dtype=float)
    total = areas.sum()
    labels = [f"Classe {c}: {names[c-1]}" for c in classes]
    colors = [f"#{palette[c-1]}" for c in classes]
    perc = (areas / total) * 100
    fig, ax = plt.subplots()
    ax.pie(perc, labels=[f"{lab}\n{p:.1f}%" for lab, p in zip(labels, perc)], colors=colors, startangle=90)
    ax.set_title(f"HVFE ({scenario}) — % de área por classe")
    st.pyplot(fig)

# ------------------- Streamlit UI -------------------
st.title("HVFE (GEE) — Clip + Statistics per Class")

scenario = st.selectbox("Cenário:", ["MIN", "MAX"])
scale = st.number_input("Scale (m):", value=30)

uploaded_file = st.file_uploader("Upload shapefile ZIP", type="zip")

if uploaded_file:
    gdf = upload_and_read_shapefile_zip(uploaded_file)
    ee_fc = gdf_to_ee_fc(gdf)
    ee_geom = ee_fc.geometry()
    img = get_hvfe_image(scenario)

    if st.button("Create a pie chart"):
        area_by_class = compute_area_by_class(img, ee_geom, scale)
        plot_pie(area_by_class, scenario)

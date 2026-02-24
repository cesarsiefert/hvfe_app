# Author: Cesar A.C. Siefert
# This code was developed by Cesar Siefert. AI tools (Microsoft 365 Copilot) were used to assist with code refinement, debugging, and structural suggestions. All final implementation decisions and validation were performed by the author.

# Streamlit dashboard: HVFE Clip + Map + Stats + Export to Google Drive

import os
import io
import json
import zipfile
import tempfile
from typing import Dict, Tuple, List, Optional

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium


# -----------------------------
# EE Initialization helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def init_ee():
    """
    Initialize Earth Engine with a Cloud project:
    1) Use GOOGLE_CLOUD_PROJECT env var if set
    2) Otherwise, use your explicit project id (edit below)
    Falls back to interactive Authenticate().
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "hvfe-ee-project")
    try:
        ee.Initialize(project=project_id)
        return project_id
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        return project_id


# -----------------------------
# Constants and class labels
# -----------------------------
SCALE_M = 30  # fixed 30 m resolution for stats and export

MIN_CLASS_LABELS = {
    1: "Surface water",
    2: "Regularly flooded wetlands",
    3: "Headwater regions",
    4: "Fixed-width buffer around low-order streams (stream orders 1–3)",
    5: "Fixed-width buffer around surface water and high-order streams (orders ≥4)",
}

MAX_CLASS_LABELS = {
    **MIN_CLASS_LABELS,
    6: "Geomorphic floodplains",
    7: "Wetland buffer corridors",
}

PALETTE_EE = ["65cbd6", "20456e", "206e35", "f4fc08", "fc0808", "d69a65", "d665b8"]
PALETTE_HEX = ["#" + c for c in PALETTE_EE]

FWC_VIS = {"min": 1, "max": 7, "palette": PALETTE_EE}


# -----------------------------
# Rasters (MIN/MAX)
# -----------------------------
def get_fwc_max() -> ee.Image:
    return ee.ImageCollection(
        [
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile1_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile2_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile3_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile4_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile5_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile6_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile7_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile8_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile9_V2"),
        ]
    ).mosaic()


def get_fwc_min() -> ee.Image:
    return ee.ImageCollection(
        [
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile1_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile2_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile3_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile4_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile5_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile6_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile7_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile8_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile9_V2"),
        ]
    ).mosaic()


# -----------------------------
# Shapefile upload & parsing
# -----------------------------
REQUIRED_EXTS = {".shp", ".shx", ".dbf", ".prj"}


def extract_zip_to_temp(uploaded_file) -> str:
    tmpdir = tempfile.mkdtemp(prefix="shpzip_")
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        z.extractall(tmpdir)
    return tmpdir


def validate_shapefile_dir(folder: str) -> Tuple[bool, str]:
    files = os.listdir(folder)
    exts = {os.path.splitext(f)[1].lower() for f in files}
    missing = REQUIRED_EXTS - exts
    if missing:
        return False, f"Missing required files: {', '.join(sorted(missing))}"
    return True, ""


def read_shapefile_to_geojson(folder: str) -> Dict:
    import shapefile as pyshp

    shp_path = None
    for f in os.listdir(folder):
        if f.lower().endswith(".shp"):
            shp_path = os.path.join(folder, f)
            break
    if shp_path is None:
        raise FileNotFoundError("No .shp file found in the ZIP.")

    r = pyshp.Reader(shp_path)
    fields = r.fields[1:]
    field_names = [f[0] for f in fields]

    features = []
    for rec, shp in zip(r.records(), r.shapes()):
        props = {name: rec[i] for i, name in enumerate(field_names)}
        geom = shp.__geo_interface__
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "features": features}


def geojson_to_ee_geometry(geojson_fc: Dict) -> ee.Geometry:
    feats = [ee.Feature(ee.Geometry(f["geometry"])) for f in geojson_fc.get("features", [])]
    if not feats:
        raise ValueError("No geometries found in uploaded file.")
    return ee.FeatureCollection(feats).geometry().dissolve()


# -----------------------------
# Visualization helpers
# -----------------------------
def add_ee_tile_layer(m: folium.Map, image: ee.Image, vis_params: Dict, name: str):
    map_id = ee.Image(image).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
    ).add_to(m)


def center_of_geometry(geom: ee.Geometry):
    c = geom.centroid(1).coordinates().getInfo()
    return float(c[1]), float(c[0])  # (lat, lon)


def add_legend_to_map(m: folium.Map, label_map: Dict[int, str], palette_hex: List[str]):
    items = ""
    for cls in sorted(label_map.keys()):
        color = palette_hex[cls - 1]
        label = label_map[cls]
        items += f"""
        <div style="display:flex;align-items:center;margin:4px 0;line-height:1.2;">
            <div style="width:14px;height:14px;background:{color};border:1px solid #999;border-radius:2px;margin-right:8px;flex-shrink:0;"></div>
            <span style="font-size:12px;color:#222;">{cls}. {label}</span>
        </div>
        """

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 28px;
        left: 28px;
        z-index: 9999;
        background: rgba(250,250,250,0.95);
        padding: 10px 12px;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.18);
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        max-width: 420px;
    ">
        <div style="font-weight:600;margin-bottom:6px;color:#222;">Legend</div>
        {items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# -----------------------------
# Statistics helpers
# -----------------------------
def compute_area_by_classes(img: ee.Image, geom: ee.Geometry, classes: List[int], scale: int = SCALE_M) -> Dict[int, float]:
    class_band = img.toInt().rename("class").unmask(0)

    mask = None
    for v in classes:
        cond = class_band.eq(int(v))
        mask = cond if mask is None else mask.Or(cond)
    class_masked = class_band.updateMask(mask)

    area_img = ee.Image.pixelArea().rename("area").reproject(crs=img.projection(), scale=scale)

    reduced = area_img.addBands(class_masked).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=geom,
        scale=scale,
        maxPixels=1e13,
        tileScale=8,
        bestEffort=False,
    )

    groups = reduced.get("groups")
    groups_list = groups.getInfo() if groups is not None else []

    area_by_class = {v: 0.0 for v in classes}
    for g in groups_list:
        cls_val = int(g["class"])
        if cls_val in area_by_class:
            area_by_class[cls_val] = float(g["sum"])

    return area_by_class


def build_share_table(area_by_class: Dict[int, float], label_map: Dict[int, str], decimals: int = 2):
    import pandas as pd

    classes = sorted(area_by_class.keys())
    values_m2 = [area_by_class[c] for c in classes]
    total_m2 = sum(values_m2)

    labels = [label_map.get(c, f"Class {c}") for c in classes]

    if total_m2 <= 0:
        df = pd.DataFrame({"HVFE Classes": labels, "Share of the Area (%)": [0.0] * len(classes)})
        return df, 0.0

    perc_raw = [(v / total_m2) * 100.0 for v in values_m2]
    perc_rounded = [round(p, decimals) for p in perc_raw]

    target = round(100.0, decimals)
    diff = round(target - round(sum(perc_rounded), decimals), decimals)
    if abs(diff) >= (10 ** (-decimals)):
        idx_max = max(range(len(perc_raw)), key=lambda i: perc_raw[i])
        perc_rounded[idx_max] = round(perc_rounded[idx_max] + diff, decimals)

    df = pd.DataFrame({"HVFE Classes": labels, "Share of the Area (%)": perc_rounded})
    return df, total_m2


# -----------------------------
# Export helper (Drive)
# -----------------------------
def start_drive_export(
    image: ee.Image,
    region: ee.Geometry,
    scale: int,
    description: str,
    file_prefix: str,
    max_pixels: float = 1e13,
) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        fileNamePrefix=file_prefix,
        region=region,
        scale=scale,
        maxPixels=max_pixels,
    )
    task.start()
    return task


def get_task_state(task: ee.batch.Task) -> Tuple[str, Dict]:
    status = task.status()
    state = status.get("state", "UNKNOWN")
    return state, status


def render_task_progress(state: str, status: Dict):
    
    # this bar is state-based (queue/running/completed/failed).
    bar = st.progress(0)

    if state == "READY":
        bar.progress(10)
        st.info("Task is queued (READY).")
    elif state == "RUNNING":
        bar.progress(60)
        st.info("Task is running, please wait. Please note that the process may take several minutes to complete.")
    elif state == "COMPLETED":
        bar.progress(100)
        st.success("Export completed! Check your Google Drive.")
    elif state == "FAILED":
        bar.progress(100)
        st.error(f"Export failed: {status.get('error_message', 'Unknown error')}")
    elif state == "CANCELLED":
        bar.progress(100)
        st.warning("Export cancelled.")
    else:
        bar.progress(5)
        st.warning(f"Task state: {state}")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="HVFE App - Clipping and Stats", layout="wide")
st.title("HVFE App – Clipping and Statistics")
st.caption(
    "Upload a zipped Shapefile (WGS84) or a GeoJSON, select HVFE MIN/MAX raster, visualize, "
    "extract class shares; and export the clipped raster to Google Drive."
)

# Initialize EE once
with st.spinner("Initializing Earth Engine..."):
    project_used = init_ee()

# Session state defaults
if "geom" not in st.session_state:
    st.session_state.geom = None
if "map_center" not in st.session_state:
    st.session_state.map_center = None
if "last_zip_token" not in st.session_state:
    st.session_state.last_zip_token = None
if "last_geojson_token" not in st.session_state:
    st.session_state.last_geojson_token = None
if "export_task" not in st.session_state:
    st.session_state.export_task = None


# -----------------------------
# TOP BAR: uploads + scenario
# -----------------------------
top_left, top_right = st.columns([2, 1])

with top_left:
    uploaded_zip = st.file_uploader(
        "Upload your Shapefile (.zip with .shp/.shx/.dbf/.prj) in WGS84 (EPSG:4326)",
        type=["zip"],
        key="zip_uploader",
    )
    uploaded_geojson = st.file_uploader(
        "Or upload a GeoJSON (WGS84/EPSG:4326)", type=["geojson", "json"], key="geojson_uploader"
    )

with top_right:
    st.subheader("Select the HVFE Delineation Scenario")
    scenario = st.radio("HVFE scenario", ["Minimum", "Maximum"], horizontal=True)

# Build selected image (CORRECT)
img = get_fwc_min() if scenario == "Minimum" else get_fwc_max()


# -----------------------------
# Handle uploads ONLY when they CHANGE
# -----------------------------
def set_new_geom(geom_obj: ee.Geometry):
    st.session_state.geom = geom_obj
    lat, lon = center_of_geometry(geom_obj)
    st.session_state.map_center = (lat, lon)


if uploaded_zip is not None:
    zip_token = (uploaded_zip.name, getattr(uploaded_zip, "size", None))
    if st.session_state.last_zip_token != zip_token:
        with st.spinner("Reading ZIP and extracting shapefile..."):
            folder = extract_zip_to_temp(uploaded_zip)
            ok, msg = validate_shapefile_dir(folder)
            if not ok:
                st.error(msg)
            else:
                try:
                    geojson_fc = read_shapefile_to_geojson(folder)
                    geom = geojson_to_ee_geometry(geojson_fc)
                    set_new_geom(geom)
                    st.session_state.last_zip_token = zip_token
                    st.success("Shapefile successfully loaded.")
                except Exception as e:
                    st.error(
                        f"Error reading shapefile: {e}\n\n"
                        "Tip: Ensure the shapefile is in WGS84 (EPSG:4326), or upload a GeoJSON."
                    )

elif uploaded_geojson is not None:
    geojson_token = (uploaded_geojson.name, getattr(uploaded_geojson, "size", None))
    if st.session_state.last_geojson_token != geojson_token:
        try:
            geojson_fc = json.load(uploaded_geojson)
            if geojson_fc.get("type") == "Feature":
                geojson_fc = {"type": "FeatureCollection", "features": [geojson_fc]}
            geom = geojson_to_ee_geometry(geojson_fc)
            set_new_geom(geom)
            st.session_state.last_geojson_token = geojson_token
            st.success("GeoJSON successfully loaded.")
        except Exception as e:
            st.error(f"Error reading GeoJSON: {e}")

geom = st.session_state.geom


# -----------------------------
# MAIN DASHBOARD: map (left) + panel (right)
# -----------------------------
left, right = st.columns([2, 1.1])

with left:
    st.subheader("Map")
    if geom is None:
        st.info("Upload a zipped Shapefile (WGS84) or a GeoJSON to enable map, stats, and export.")
    else:
        clipped_current = img.clip(geom)

        lat, lon = st.session_state.get("map_center", center_of_geometry(geom))
        m = folium.Map(location=[lat, lon], zoom_start=9, tiles="OpenStreetMap")

        add_ee_tile_layer(m, clipped_current, FWC_VIS, f"HVFE {scenario}")

        # AOI outline
        try:
            aoi_geojson = ee.FeatureCollection([ee.Feature(geom)]).getInfo()
            folium.GeoJson(
                aoi_geojson,
                name="AOI",
                style_function=lambda x: {"fillColor": "#00000000", "color": "#333333", "weight": 2},
            ).add_to(m)
        except Exception:
            folium.CircleMarker(location=[lat, lon], radius=6, color="#333", fill=True).add_to(m)

        folium.LayerControl().add_to(m)

        # Legend inside the map
        if scenario == "Minimum":
            add_legend_to_map(m, MIN_CLASS_LABELS, PALETTE_HEX)
        else:
            add_legend_to_map(m, MAX_CLASS_LABELS, PALETTE_HEX)

        st_folium(m, use_container_width=True, height=780, key="map_view")


with right:
    st.subheader("Stats")

    if geom is None:
        st.caption("Waiting for an AOI upload...")
    else:
        clipped_img = img.clip(geom)

        # Classes/labels
        if scenario == "Minimum":
            classes = [1, 2, 3, 4, 5]
            label_map = MIN_CLASS_LABELS
        else:
            classes = [1, 2, 3, 4, 5, 6, 7]
            label_map = MAX_CLASS_LABELS

        if st.button("📊 Compute class shares"):
            with st.spinner("Computing class statistics..."):
                try:
                    area_by_class = compute_area_by_classes(clipped_img, geom, classes, scale=SCALE_M)
                    df, total_m2 = build_share_table(area_by_class, label_map, decimals=2)

                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.markdown(f"**Overall HVFE area:** {total_m2/1e6:.3f} km²")
                except Exception as e:
                    st.error(f"Error computing stats: {e}")

        st.divider()

        # --- Export (Google Drive) ---
        st.subheader("Export clipped HVFE raster (Google Drive)")

        st.caption("⚠️ The GeoTIFF will be saved to your Google Drive account.")

        # customizing export name without changing the rest of the app logic
        export_name = st.text_input("Export name (Drive)", value="hvfe_clip")

        if st.button("⬆️ Export to Google Drive"):
            with st.spinner("Submitting export task to Earth Engine..."):
                try:
                    # 1) Rectangular bounding box of AOI (not shown on map)
                    bbox = geom.bounds(1)

                    # 2) Clip scenario raster by bounding box for export
                    
                    img_to_export = img.clip(bbox).toByte()

                    task = start_drive_export(
                        image=img_to_export,
                        region=bbox,
                        scale=SCALE_M,
                        description=export_name,
                        file_prefix=export_name,
                    )

                    st.session_state.export_task = task
                    st.success("Export task submitted! You can monitor the status below by clicking on the 'Refresh export status' button. Please note that the process may take several minutes to complete.")
                except Exception as e:
                    st.error(f"Error submitting export task: {e}")

        # --- Progress monitoring ---
        task = st.session_state.export_task
        if task is not None:
            state, status = get_task_state(task)
            render_task_progress(state, status)

            # Show a couple helpful details
            #if "creation_timestamp_ms" in status:
            #   st.caption(f"Task created (ms): {status['creation_timestamp_ms']}")
            #if "id" in status:
            #    st.caption(f"Task id: {status['id']}")

            if st.button("🔄 Refresh export status"):
                st.rerun()

import os
import io
import zipfile
import tempfile
from datetime import datetime

import streamlit as st
import ee
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import mapping
import folium
from streamlit_folium import st_folium
import plotly.express as px
import requests


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="HVFE App - Clipping and Stats",
    layout="wide",
)

st.title("HVFE App – Clip & Stats (GEE)")
st.caption("Upload a zipped shapefile, select MIN or MAX raster, clip, visualize, chart stats, and download the clipped raster (no Drive export).")


# ----------------------------
# GEE Initialization
# ----------------------------
def init_ee():
    """Initialize Google Earth Engine with priority:
    1) Service account using env vars
    2) Existing initialization
    3) Interactive user authentication (local)
    """
    try:
        if ee.data._initialized:
            return
    except Exception:
        pass

    service_account = os.getenv("EE_SERVICE_ACCOUNT")
    service_key_json = os.getenv("EE_PRIVATE_KEY_JSON")  # full JSON string

    try:
        if service_account and service_key_json:
            credentials = ee.ServiceAccountCredentials(
                service_account=service_account,
                key_data=service_key_json,
            )
            ee.Initialize(credentials)
        else:
            # Try default initialize first (works if EE is already authenticated on host)
            ee.Initialize()
    except Exception:
        # Fallback to user authentication (local dev)
        ee.Authenticate()
        ee.Initialize()

init_ee()


# ----------------------------
# Constants: classes, palette, visualization
# ----------------------------
names = [
    'Surface Water',
    'Regularly Flooded wetlands',
    'Headwater regions',
    'Fixed‐width buffer (low‐order streams)',
    'Fixed‐width buffer (high‐order streams)',
    'Geomorphic floodplains',
    'Fixed‐width buffer (regularly flooded wetlands)'
]

palette = ['#65cbd6', '#20456e', '#206e35', '#f4fc08', '#fc0808', '#d69a65', '#d665b8']

fwc_vis = dict(min=1, max=7, palette=palette)


# ----------------------------
# Build the HVFE images (MAX and MIN mosaics)
# ----------------------------
def get_fwc_image(choice: str) -> ee.Image:
    """Return the HVFE image mosaic for 'MIN' or 'MAX'."""
    if choice == "MAX":
        fwc_max = ee.ImageCollection([
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile1_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile2_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile3_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile4_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile5_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile6_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile7_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile8_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_max_tile9_V2")
        ]).mosaic()
        return fwc_max
    else:
        fwc_min = ee.ImageCollection([
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile1_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile2_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile3_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile4_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile5_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile6_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile7_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile8_V2"),
            ee.Image("projects/ee-vsgriffey/assets/riparian_full_min_tile9_V2")
        ]).mosaic()
        return fwc_min


# ----------------------------
# Helpers: shapefile upload -> ee.Geometry
# ----------------------------
def load_shapefile_from_zip(uploaded_zip_file) -> tuple[ee.Geometry, gpd.GeoDataFrame]:
    """Read a zipped shapefile into a single ee.Geometry (union of features)."""
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip_file) as zf:
        zf.extractall(tmpdir)

    shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".shp")]
    if not shp_files:
        raise ValueError("No .shp file found in the uploaded ZIP.")

    gdf = gpd.read_file(shp_files[0])
    if gdf.empty or gdf.geometry.is_empty.all():
        raise ValueError("The shapefile has no valid geometries.")

    # Reproject to WGS84 for Earth Engine
    gdf = gdf.to_crs(epsg=4326)

    # Union all geometries to a single geometry (clip region)
    union_geom = unary_union(gdf.geometry)
    if union_geom.is_empty:
        raise ValueError("Geometry (union) is empty after processing.")

    ee_geom = ee.Geometry(mapping(union_geom), geodesic=False)
    return ee_geom, gdf


# ----------------------------
# Folium helpers (EE tile layer + legend)
# ----------------------------
def add_ee_layer(m, ee_image, vis_params, name):
    """Add an Earth Engine image layer to a folium map."""
    map_id_dict = ee.Image(ee_image).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        opacity=0.85
    ).add_to(m)


def add_legend(m, title, labels, colors):
    """Add a simple HTML legend to folium map."""
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; z-index: 9999; 
        background-color: rgba(255,255,255,0.9);
        padding: 10px 12px; border: 1px solid #ccc; border-radius: 4px;
        font-size: 13px;">
      <b>{title}</b><br>
    """
    for lbl, col in zip(labels, colors):
        legend_html += f"""
        <div style="display:flex;align-items:center;margin:2px 0;">
          <div style="width:12px;height:12px;background:{col};margin-right:6px;border:1px solid #666;"></div>
          <span>{lbl}</span>
        </div>
        """
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))


# ----------------------------
# Stats: area and percentage per class
# ----------------------------
def compute_area_stats(image: ee.Image, region: ee.Geometry, scale: int = 30) -> dict:
    """
    Compute area per class (1..7) in m² within the region.
    Uses pixelArea and group reducer for robust area computation.
    """
    # Ensure it's a single-band classification named 'class'
    class_img = image.rename('class').toInt16()

    # Sum of pixelArea grouped by class
    grouped = ee.Image.pixelArea().addBands(class_img).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName='class'),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        tileScale=4
    )

    groups = ee.Dictionary(grouped.get('groups'))
    groups_list = groups.get('groups') if groups.contains('groups') else None

    # When EE returns list of dicts like [{'class': 1, 'sum': area_m2}, ...]
    result = {}
    if groups_list:
        records = ee.List(groups_list).getInfo()
        for rec in records:
            result[int(rec['class'])] = float(rec['sum'])
    else:
        # Some regions may produce empty results if no overlap
        result = {}

    return result


def to_percentage(area_by_class: dict) -> dict:
    total_area = sum(area_by_class.values()) if area_by_class else 0.0
    if total_area <= 0:
        return {c: 0.0 for c in range(1, 8)}
    return {c: (area_by_class.get(c, 0.0) / total_area) * 100.0 for c in range(1, 8)}


# ----------------------------
# Download clipped raster (GeoTIFF) without Drive export
# ----------------------------
def get_geotiff_bytes(image: ee.Image, region: ee.Geometry, scale: int = 30, crs: str = "EPSG:4326") -> tuple[bytes, str]:
    """
    Obtain a download URL and fetch bytes for a clipped classification raster as GeoTIFF.
    Avoids Export to Drive by using getDownloadURL.
    """
    params = {
        "scale": scale,
        "region": region,
        "crs": crs,
        "format": "GEO_TIFF",
    }
    url = image.toInt16().getDownloadURL(params)
    # Fetch the content – in some hosted environments outbound requests may be blocked
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    return r.content, url


# ----------------------------
# UI – Sidebar
# ----------------------------
with st.sidebar:
    st.header("Options")
    raster_choice = st.radio("Select raster", ["MIN", "MAX"], index=0, horizontal=True)
    scale = st.number_input("Analysis scale (meters/pixel)", min_value=10, max_value=500, value=30, step=10,
                            help="Use a scale close to the native resolution of the dataset.")
    uploaded_zip = st.file_uploader(
        "Upload shapefile (.zip with .shp, .shx, .dbf, .prj)",
        type=["zip"],
        accept_multiple_files=False
    )
    visualize_btn = st.button("Visualize on Map", use_container_width=True)
    stats_btn = st.button("Create Pie Chart (Stats %)", use_container_width=True)
    download_btn = st.button("Download Clipped Raster (GeoTIFF)", use_container_width=True)


# ----------------------------
# Main layout
# ----------------------------
col_map, col_plot = st.columns([1.6, 1.0])

# Store items in session_state to avoid recomputation
if "ee_geom" not in st.session_state:
    st.session_state.ee_geom = None
if "gdf" not in st.session_state:
    st.session_state.gdf = None
if "clipped_image" not in st.session_state:
    st.session_state.clipped_image = None
if "last_choice" not in st.session_state:
    st.session_state.last_choice = None


# ----------------------------
# Load shapefile and build clipped image
# ----------------------------
def prepare_data():
    if uploaded_zip is None:
        st.warning("Please upload a zipped shapefile to proceed.")
        return False

    try:
        ee_geom, gdf = load_shapefile_from_zip(uploaded_zip)
        st.session_state.ee_geom = ee_geom
        st.session_state.gdf = gdf
    except Exception as e:
        st.error(f"Error reading shapefile: {e}")
        return False

    try:
        img = get_fwc_image(raster_choice)
        st.session_state.last_choice = raster_choice
        st.session_state.clipped_image = img.clip(st.session_state.ee_geom)
    except Exception as e:
        st.error(f"Error building/clipping image: {e}")
        return False

    return True


# ----------------------------
# Visualize on Map
# ----------------------------
if visualize_btn:
    if prepare_data():
        with col_map:
            # Build base map zoomed to shapefile bounds
            bounds = st.session_state.gdf.total_bounds  # [minx, miny, maxx, maxy]
            center_lat = (bounds[1] + bounds[3]) / 2.0
            center_lon = (bounds[0] + bounds[2]) / 2.0

            m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB positron")

            # Add EE layer
            layer_name = f"HVFE {st.session_state.last_choice}"
            add_ee_layer(m, st.session_state.clipped_image, fwc_vis, layer_name)

            # Add a legend
            add_legend(m, "HVFE Classes", names, palette)

            # Add layer control
            folium.LayerControl(collapsed=False).add_to(m)

            st_folium(m, width=None, height=650)

        with col_plot:
            st.info("Tip: Click 'Create Pie Chart (Stats %)' to compute class percentages for the clipped area.")


# ----------------------------
# Stats & Pie chart
# ----------------------------
if stats_btn:
    if st.session_state.clipped_image is None or st.session_state.ee_geom is None:
        if not prepare_data():
            st.stop()

    with st.spinner("Computing class areas and percentages..."):
        try:
            areas = compute_area_stats(st.session_state.clipped_image, st.session_state.ee_geom, scale=scale)
            percentages = to_percentage(areas)

            # Prepare data for the pie
            class_ids = list(range(1, 8))
            labels = [names[i-1] for i in class_ids]
            values = [percentages.get(i, 0.0) for i in class_ids]

            # Color mapping label -> palette color
            color_map = {labels[i]: palette[i] for i in range(len(labels))}

            fig = px.pie(
                names=labels,
                values=values,
                title="Percentual por classe (clipped area)",
                color=labels,
                color_discrete_map=color_map,
                hole=0.3
            )
            fig.update_traces(textposition='inside', texttemplate='%{label}<br>%{value:.2f}%')
            fig.update_layout(showlegend=False, height=600)

            col_plot.plotly_chart(fig, use_container_width=True)

            # Show a compact table
            stats_rows = []
            for i, lbl in enumerate(labels, start=1):
                stats_rows.append({
                    "Classe": lbl,
                    "Área (m²)": round(areas.get(i, 0.0), 2),
                    "Percentual (%)": round(percentages.get(i, 0.0), 4),
                })
            col_plot.subheader("Tabela de estatísticas")
            col_plot.dataframe(stats_rows, use_container_width=True)
        except Exception as e:
            st.error(f"Error computing stats: {e}")


# ----------------------------
# Download (GeoTIFF)
# ----------------------------
if download_btn:
    if st.session_state.clipped_image is None or st.session_state.ee_geom is None:
        if not prepare_data():
            st.stop()

    with st.spinner("Preparing GeoTIFF for download..."):
        try:
            content, url = get_geotiff_bytes(st.session_state.clipped_image, st.session_state.ee_geom, scale=scale)
            now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            fname = f"HVFE_{st.session_state.last_choice}_clipped_{now}.tif"

            st.download_button(
                label="Download GeoTIFF",
                data=content,
                file_name=fname,
                mime="image/tiff",
                use_container_width=True
            )
            st.caption("If the button fails due to outbound HTTP restrictions, use the direct link below:")
            st.code(url, language="text")
        except Exception as e:
            st.error(f"Error fetching GeoTIFF: {e}")
            st.info("If you're running in a restricted environment, try running locally or use the direct URL approach via getDownloadURL.")
``

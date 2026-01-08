# filepath: /Users/eliowanner/Documents/local_stuff/hackathon_sbb_2026/test2/hackathon_sbb/src/frontend.py
# Streamlit frontend (self-contained, hardcoded sample data)
# run with "streamlit run src/frontend.py"

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import math
from pathlib import Path
import json
import subprocess
import sys
import urllib.parse
import urllib.request
import hashlib

# --- Optional OSRM routing (for Bus edges) ---
# Uses the public demo server by default. For production, run your own OSRM instance.
OSRM_BASE_URL = "https://router.project-osrm.org"


def _hex_color_from_string(s: str) -> str:
    """Deterministic bright-ish color for a given key."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    # Keep it a bit brighter by mixing with a baseline.
    r = (int(h[0:2], 16) // 2) + 80
    g = (int(h[2:4], 16) // 2) + 80
    b = (int(h[4:6], 16) // 2) + 80
    return f"#{r:02x}{g:02x}{b:02x}"


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _osrm_route_lon_lat(
    lon_a: float,
    lat_a: float,
    lon_b: float,
    lat_b: float,
    *,
    profile: str = "driving",
    overview: str = "full",
    geometries: str = "geojson",
    timeout_s: float = 10.0,
) -> list[tuple[float, float]]:
    """Fetch a route polyline from OSRM.

    Returns list of (lon, lat) points. Empty list on errors.
    """
    try:
        coords = f"{lon_a:.6f},{lat_a:.6f};{lon_b:.6f},{lat_b:.6f}"
        params = {
            "overview": overview,
            "geometries": geometries,
        }
        url = f"{OSRM_BASE_URL}/route/v1/{profile}/{coords}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "hackathon_sbb_frontend"})
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        routes = payload.get("routes")
        if not routes:
            return []
        geom = routes[0].get("geometry")
        if not geom:
            return []
        # With geometries=geojson, geometry is {"coordinates": [[lon,lat], ...], "type": "LineString"}
        coords_list = geom.get("coordinates") if isinstance(geom, dict) else None
        if not isinstance(coords_list, list):
            return []
        pts: list[tuple[float, float]] = []
        for p in coords_list:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except Exception:
                    continue
        return pts
    except Exception:
        return []

st.set_page_config(page_title="Schedule Visualizer", layout="wide")

st.title("Bio-Construction-Schedular")

# Load best_result from output/summary.json (fallback to a small hardcoded example)
project_root = Path(__file__).resolve().parents[1]
summary_json_path = project_root / "output" / "summary.json"

fallback_best_result = {"best_schedule": {
        "A1": 3.1712530261677268,
        "A2": 1.508604770429887,
        "A3": 0.0,
        "A4": 60.61600586577646,
    },
    # New schema: objectives
    "objectives": {
        "condition_penalty": 0.024060787510129356,
        "travel_penalty": 0.09585103276242014,
        "cost_penalty": 1134000.0,
    },
}

# ---- NSGA-II controls (sidebar) ----
with st.sidebar:
    st.header("Run NSGA-II")
    w1 = st.slider("Weight: condition", 0.0, 1.0, 0.3, 0.01)
    w2 = st.slider("Weight: travel time", 0.0, 1.0, 0.4, 0.01)
    w3 = st.slider("Weight: cost", 0.0, 1.0, 0.3, 0.01)
    # Normalize weights so they sum to 1 (avoid passing all zeros)
    _total_w = float(w1 + w2 + w3)
    if _total_w <= 0:
        weights = (0.33, 0.34, 0.33)
    else:
        weights = (w1 / _total_w, w2 / _total_w, w3 / _total_w)
    st.write(f"Normalized weights: {weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}")

    horizon = st.slider("Horizon (hours)", 1.0, 1000.0, 168.0, 1.0)
    population = st.slider("Population", 4, 1000, 40, 1)
    cx = st.slider("Crossover (cx)", 0.0, 1.0, 0.3, 0.01)
    mut = st.slider("Mutation (mut)", 0.0, 1.0, 0.1, 0.01)
    run_now = st.button("Run NSGA-II")
    status_area = st.empty()

    # Optionally show generations/seed inputs if needed
    generations = st.number_input("Generations", min_value=1, max_value=10000, value=30, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)

try:
    with open(summary_json_path, "r", encoding="utf-8") as f:
        best_result = json.load(f)
    if not isinstance(best_result, dict):
        raise ValueError("summary.json did not contain a JSON object")
except Exception as e:
    st.warning(f"Could not read {summary_json_path}: {e}. Falling back to hardcoded sample data.")
    best_result = fallback_best_result

# Backwards-compatible shim (if some older file still uses weighted_objectives)
if "objectives" not in best_result and "weighted_objectives" in best_result:
    best_result["objectives"] = best_result.get("weighted_objectives")

# Part 1: Map with nodes and edges
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Network map")

# Load nodes from CSV (data/nodes.csv). Format:
# node_id,location_x,location_y  (location_x=lat, location_y=lon)
data_dir = Path(__file__).resolve().parents[1] / "data"
nodes_csv_path = data_dir / "nodes_template.csv"
edges_csv_path = data_dir / "edges_template.csv"

fallback_nodes = [
    {"id": "N1", "lat": 47.3769, "lon": 8.5417},  # Zurich
    {"id": "N2", "lat": 46.9480, "lon": 7.4474},  # Bern
    {"id": "N3", "lat": 46.2044, "lon": 6.1432},  # Geneva
    {"id": "N4", "lat": 47.5596, "lon": 7.5886},  # Basel
    {"id": "N5", "lat": 46.5197, "lon": 6.6323},  # Lausanne
]

try:
    print(f"[DEBUG frontend] reading nodes from {nodes_csv_path}", flush=True)
    df_nodes = pd.read_csv(nodes_csv_path)
    required_cols = {"node_id", "location_x", "location_y"}
    if not required_cols.issubset(set(df_nodes.columns)):
        raise ValueError(
            f"nodes.csv must contain columns {sorted(required_cols)}; got {sorted(df_nodes.columns)}"
        )

    df_nodes = df_nodes.dropna(subset=["node_id", "location_x", "location_y"]).copy()
    df_nodes["location_x"] = pd.to_numeric(df_nodes["location_x"], errors="coerce")
    df_nodes["location_y"] = pd.to_numeric(df_nodes["location_y"], errors="coerce")
    df_nodes = df_nodes.dropna(subset=["location_x", "location_y"]).reset_index(drop=True)

    nodes = [
        {"id": str(r.node_id), "lat": float(r.location_x), "lon": float(r.location_y)}
        for r in df_nodes.itertuples(index=False)
    ]
    print(f"[DEBUG frontend] loaded {len(nodes)} nodes", flush=True)
    if len(nodes) == 0:
        nodes = fallback_nodes
except Exception as e:
    st.warning(f"Could not read nodes from {nodes_csv_path}: {e}. Falling back to hardcoded nodes.")
    nodes = fallback_nodes

# Load edges from CSV (data/edges.csv). For the map we only use start_node and end_node.
fallback_edges = [
    ("N1", "N2"),
    ("N1", "N4"),
    ("N2", "N5"),
    ("N3", "N5"),
    ("N4", "N1"),
]

def _parse_lon_lat_path(value) -> list[tuple[float, float]]:
    """Parse a polyline from CSV.

    Supported formats:
    - JSON array of [lon, lat] pairs: [[8.5, 47.3], [8.6, 47.35]]
    - JSON array of objects: [{"lon": 8.5, "lat": 47.3}, ...]

    Returns list of (lon, lat). Empty list on missing/invalid.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, (list, tuple)):
        raw = value
    else:
        s = str(value).strip()
        if not s:
            return []
        try:
            raw = json.loads(s)
        except Exception:
            return []

    pts: list[tuple[float, float]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    lon = float(item[0])
                    lat = float(item[1])
                    pts.append((lon, lat))
                except Exception:
                    continue
            elif isinstance(item, dict) and "lon" in item and "lat" in item:
                try:
                    pts.append((float(item["lon"]), float(item["lat"])))
                except Exception:
                    continue
    return pts


def _edge_key(a: str, b: str) -> str:
    return f"{a}->{b}"

try:
    print(f"[DEBUG frontend] reading edges from {edges_csv_path}", flush=True)
    df_edges = pd.read_csv(edges_csv_path)
    required_edge_cols = {"start_node", "end_node", "capacity_at_day"}
    if not required_edge_cols.issubset(set(df_edges.columns)):
        raise ValueError(
            f"edges.csv must contain columns {sorted(required_edge_cols)}; got {sorted(df_edges.columns)}"
        )
    df_edges = df_edges.dropna(subset=["start_node", "end_node"]).copy()
    df_edges["capacity_at_day"] = pd.to_numeric(df_edges["capacity_at_day"], errors="coerce")

    # Keep capacity + mode with each edge so we can style and optionally route it.
    edges = []
    for r in df_edges.itertuples(index=False):
        a = str(getattr(r, "start_node"))
        b = str(getattr(r, "end_node"))
        cap = getattr(r, "capacity_at_day", None)
        cap_val = float(cap) if cap is not None and not pd.isna(cap) else None
        mode_val = getattr(r, "mode", None) if "mode" in df_edges.columns else None
        edge_id_val = getattr(r, "edge_id", None) if "edge_id" in df_edges.columns else None
        edges.append(
            {
                "a": a,
                "b": b,
                "edge_id": (str(edge_id_val) if edge_id_val is not None and not pd.isna(edge_id_val) else _edge_key(a, b)),
                "capacity_at_day": cap_val,
                "mode": (str(mode_val) if mode_val is not None and not pd.isna(mode_val) else None),
            }
        )
    print(f"[DEBUG frontend] loaded {len(edges)} edges", flush=True)
    if len(edges) == 0:
        edges = fallback_edges
except Exception as e:
    st.warning(f"Could not read edges from {edges_csv_path}: {e}. Falling back to hardcoded edges.")
    edges = fallback_edges

# Load asset->edge mapping so the schedule timeline can share colors with edges.
assets_csv_path = data_dir / "assets_template.csv"
asset_edge_by_asset_id: dict[str, str] = {}
try:
    df_assets_for_edges = pd.read_csv(assets_csv_path)
    if {"asset_id", "edge_id"}.issubset(set(df_assets_for_edges.columns)):
        df_assets_for_edges = df_assets_for_edges.dropna(subset=["asset_id", "edge_id"]).copy()
        asset_edge_by_asset_id = {
            str(r.asset_id): str(r.edge_id) for r in df_assets_for_edges.itertuples(index=False)
        }
except Exception:
    asset_edge_by_asset_id = {}

# Deterministic color per edge_id for consistent map + schedule colors.
edge_color_by_edge_id: dict[str, str] = {}
if len(edges) > 0 and isinstance(edges[0], dict):
    for e in edges:
        eid = str(e.get("edge_id") or _edge_key(e.get("a", ""), e.get("b", "")))
        if eid not in edge_color_by_edge_id:
            edge_color_by_edge_id[eid] = _hex_color_from_string(eid)

# Optional: load train (Zug) polylines from data/edge_paths_train.csv
edge_paths_csv_path = data_dir / "edge_paths_train.csv"
train_paths: dict[str, list[tuple[float, float]]] = {}
_train_paths_load_error: str | None = None
try:
    if edge_paths_csv_path.exists():
        # NOTE: path_lon_lat contains commas. In CSV this MUST be quoted, e.g.
        # "[[8.53,47.04],[8.54,47.05]]"
        # We use the python engine + backslash escape to be more forgiving.
        try:
            df_paths = pd.read_csv(edge_paths_csv_path, engine="python", escapechar="\\")
        except Exception:
            # Fallback: allow a pipe-separated file if you prefer to avoid quoting JSON.
            df_paths = pd.read_csv(edge_paths_csv_path, sep="|", engine="python")

        required_path_cols = {"start_node", "end_node", "path_lon_lat"}
        if not required_path_cols.issubset(set(df_paths.columns)):
            raise ValueError(
                f"edge_paths_train.csv must contain columns {sorted(required_path_cols)}; got {sorted(df_paths.columns)}"
            )
        df_paths = df_paths.dropna(subset=["start_node", "end_node"]).copy()
        for r in df_paths.itertuples(index=False):
            a = str(getattr(r, "start_node"))
            b = str(getattr(r, "end_node"))
            path_val = getattr(r, "path_lon_lat", None)
            pts = _parse_lon_lat_path(path_val)
            if pts:
                train_paths[_edge_key(a, b)] = pts
except Exception as e:
    _train_paths_load_error = str(e)
    st.warning(f"Could not read train edge paths from {edge_paths_csv_path}: {e}. Falling back to straight lines.")

with st.sidebar:
    st.caption(
        "Train routes (mode=Zug) can optionally be drawn from data/edge_paths_train.csv "
        "(columns: start_node,end_node,path_lon_lat)."
    )
    st.caption(
        "If you use comma-separated CSV, wrap path_lon_lat in quotes, e.g. "
        "\"[[8.53,47.04],[8.54,47.05]]\". "
        "Alternatively you can use a pipe-separated file (same name) with rows like: "
        "start_node|end_node|path_lon_lat"
    )
    if _train_paths_load_error:
        st.warning(f"edge_paths_train.csv parse error: {_train_paths_load_error}")

# Build coordinate maps
coord = {n["id"]: (n["lon"], n["lat"]) for n in nodes}

# Filter edges that reference unknown nodes; if nothing remains, build a simple chain.
if len(edges) > 0 and isinstance(edges[0], dict):
    edges = [e for e in edges if e["a"] in coord and e["b"] in coord]
    if len(edges) == 0:
        node_ids = [n["id"] for n in nodes]
        edges = [
            {"a": a, "b": b, "capacity_at_day": None}
            for a, b in (list(zip(node_ids, node_ids[1:])) if len(node_ids) > 1 else [])
        ]
else:
    # fallback_edges list of tuples
    edges = [(a, b) for (a, b) in edges if a in coord and b in coord]
    if len(edges) == 0:
        node_ids = [n["id"] for n in nodes]
        edges = list(zip(node_ids, node_ids[1:])) if len(node_ids) > 1 else []

# Create a Plotly figure: lines for edges and scatter for nodes
edge_traces = []

# Precompute capacity range for nicer scaling (typical values: ~1000–5000)
_caps = []
if len(edges) > 0 and isinstance(edges[0], dict):
    _caps = [e.get("capacity_at_day") for e in edges if e.get("capacity_at_day") is not None]
_cap_min = float(min(_caps)) if _caps else None
_cap_max = float(max(_caps)) if _caps else None

def _edge_width_from_capacity(capacity_at_day: float | None) -> float:
    # Map capacity to a visually reasonable width range.
    # For your current data (about 1000–5000), linear scaling gives clearer differences.
    if capacity_at_day is None or capacity_at_day <= 0:
        return 2.0

    # If we don't have a range (missing data or constant capacity), fall back to a default.
    if _cap_min is None or _cap_max is None or _cap_min == _cap_max:
        return 4.0

    # Normalize to [0, 1]
    t = (float(capacity_at_day) - _cap_min) / (_cap_max - _cap_min)
    t = max(0.0, min(1.0, t))

    # Map to a width range that is visually distinct but not too thick.
    min_w, max_w = 2.0, 10.0
    return float(min_w + t * (max_w - min_w))

if len(edges) > 0 and isinstance(edges[0], dict):
    for e in edges:
        a, b = e["a"], e["b"]
        if a not in coord or b not in coord:
            continue

        # Use polyline only for trains and only if we have a path.
        pts: list[tuple[float, float]] = []
        mode = (e.get("mode") or "").strip()
        if mode in ("Zug", "Schiff"):
            pts = train_paths.get(_edge_key(a, b), [])
        elif mode == "Bus":
            lon_a, lat_a = coord[a]
            lon_b, lat_b = coord[b]
            pts = _osrm_route_lon_lat(lon_a, lat_a, lon_b, lat_b)

        # If a path exists, ensure it connects the endpoints (prepend/append if needed).
        if pts:
            lon_a, lat_a = coord[a]
            lon_b, lat_b = coord[b]
            if (abs(pts[0][0] - lon_a) > 1e-6) or (abs(pts[0][1] - lat_a) > 1e-6):
                pts = [(lon_a, lat_a)] + pts
            if (abs(pts[-1][0] - lon_b) > 1e-6) or (abs(pts[-1][1] - lat_b) > 1e-6):
                pts = pts + [(lon_b, lat_b)]

        if pts:
            lon = [p[0] for p in pts]
            lat = [p[1] for p in pts]
        else:
            lon_a, lat_a = coord[a]
            lon_b, lat_b = coord[b]
            lon = [lon_a, lon_b]
            lat = [lat_a, lat_b]

        w = _edge_width_from_capacity(e.get("capacity_at_day"))
        edge_id = str(e.get("edge_id") or _edge_key(a, b))
        edge_color = edge_color_by_edge_id.get(edge_id) or _hex_color_from_string(edge_id)
        edge_traces.append(
            go.Scattermapbox(
                lon=lon,
                lat=lat,
                mode="lines",
                line=dict(width=w, color=edge_color),
                hovertemplate=(
                    f"<b>{a}  {b}</b><br>"
                    f"edge_id: {edge_id}<br>"
                    f"mode: {mode}<br>"
                    f"capacity_at_day: {e.get('capacity_at_day')}<br>"
                    f"path_points: {len(pts) if pts else 0}<extra></extra>"
                ),
            )
        )
else:
    for a, b in edges:
        lon_a, lat_a = coord[a]
        lon_b, lat_b = coord[b]
        edge_traces.append(
            go.Scattermapbox(
                lon=[lon_a, lon_b],
                lat=[lat_a, lat_b],
                mode="lines",
                line=dict(width=2, color="blue"),
                hoverinfo="none",
            )
        )

node_trace = go.Scattermapbox(
    lon=[n["lon"] for n in nodes],
    lat=[n["lat"] for n in nodes],
    mode="markers+text",
    marker=go.scattermapbox.Marker(size=12, color="red"),
    text=[n["id"] for n in nodes],
    textposition="top center",
)

# Compose and display map. Use an open-access map style that doesn't require a Mapbox token.
fig_map = go.Figure(data=edge_traces + [node_trace])
if len(nodes) > 0:
    center_lat = sum(n["lat"] for n in nodes) / len(nodes)
    center_lon = sum(n["lon"] for n in nodes) / len(nodes)
else:
    center_lat, center_lon = 46.8, 8.0

fig_map.update_layout(
    mapbox=dict(
        # Black/white basemap
        style="carto-positron",
        center=dict(lat=float(center_lat), lon=float(center_lon)),
        zoom=10,
    ),
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
    height=450,
    showlegend=False,
)

with col_left:
    st.plotly_chart(fig_map, width="stretch")

# Part 2: Calendar / Gantt view for assets
with col_right:
    st.subheader("Construction schedule timeline")

# Assets to show. We'll derive start day from the best_result values.
assets = list(best_result["best_schedule"].keys())
start_values = best_result["best_schedule"]

# Base date for the visualization
base_date = date(2026, 1, 1)

# Load per-asset maintenance durations (half-days) from assets_template.csv
# (assets_csv_path already defined above)
DEFAULT_DURATION_DAYS = 4.0

duration_days_by_asset: dict[str, float] = {}
try:
    df_assets = pd.read_csv(assets_csv_path)
    required_asset_cols = {"asset_id", "maintenance_duration_half_days"}
    if not required_asset_cols.issubset(set(df_assets.columns)):
        raise ValueError(
            f"assets_template.csv must contain columns {sorted(required_asset_cols)}; got {sorted(df_assets.columns)}"
        )

    df_assets = df_assets.dropna(subset=["asset_id"]).copy()
    df_assets["maintenance_duration_half_days"] = pd.to_numeric(
        df_assets["maintenance_duration_half_days"], errors="coerce"
    )
    # Convert half-days -> days
    df_assets["duration_days"] = df_assets["maintenance_duration_half_days"] / 2.0

    duration_days_by_asset = {
        str(r.asset_id): (
            float(r.duration_days)
            if r.duration_days is not None and not pd.isna(r.duration_days) and float(r.duration_days) > 0
            else DEFAULT_DURATION_DAYS
        )
        for r in df_assets.itertuples(index=False)
    }
except Exception as e:
    st.warning(
        f"Could not read asset durations from {assets_csv_path}: {e}. Falling back to {DEFAULT_DURATION_DAYS} days."
    )

rows = []
for asset in assets:
    raw_start = start_values[asset]
    # For the mocked calendar, interpret the start as whole-day offset (floor)
    start_day = int(math.floor(raw_start))
    start_dt = base_date + timedelta(days=start_day)

    duration_days = float(duration_days_by_asset.get(asset, DEFAULT_DURATION_DAYS))
    end_dt = start_dt + timedelta(days=float(duration_days))

    # Use pandas timestamps (JSON-serializable through Plotly) and keep them separate from any timedelta objects
    rows.append(
        {
            "Asset": asset,
            "Start": pd.Timestamp(start_dt),
            "End": pd.Timestamp(end_dt),
            "raw_start": float(raw_start),
            "duration_days": float(duration_days),
        }
    )

df_schedule = pd.DataFrame(rows)

# Ensure correct dtypes
df_schedule["Start"] = pd.to_datetime(df_schedule["Start"], utc=False)
df_schedule["End"] = pd.to_datetime(df_schedule["End"], utc=False)
df_schedule["raw_start"] = df_schedule["raw_start"].astype(float)

# Plotly sometimes stumbles over non-primitive dtypes present in the dataframe.
# Pass a minimal dataframe to the chart to guarantee JSON-serializability.
df_schedule_plot = df_schedule[["Asset", "Start", "End"]].copy()
df_schedule_plot["Start"] = df_schedule_plot["Start"].dt.to_pydatetime()
df_schedule_plot["End"] = df_schedule_plot["End"].dt.to_pydatetime()

# Create a Gantt-like timeline (manual) to avoid datetime/timedelta JSON issues.
# We plot in "days since base_date" and format tick labels ourselves.
df_bar = df_schedule.copy()
df_bar["start_day"] = (df_bar["Start"].dt.normalize() - pd.Timestamp(base_date)).dt.days
# Use the computed duration (may be fractional) rather than re-deriving it from timestamps.
df_bar["duration_days"] = df_bar.get("duration_days", DEFAULT_DURATION_DAYS).astype(float)

fig_timeline = go.Figure()
for _, r in df_bar.iterrows():
    asset_id = str(r["Asset"])
    edge_id_for_asset = asset_edge_by_asset_id.get(asset_id)
    bar_color = (
        edge_color_by_edge_id.get(edge_id_for_asset)
        if edge_id_for_asset is not None
        else _hex_color_from_string(asset_id)
    )
    start_day = int(r["start_day"])
    dur_days = float(r["duration_days"])
    end_day = float(start_day + dur_days)
    start_date = base_date + timedelta(days=start_day)
    end_date = base_date + timedelta(days=end_day)

    fig_timeline.add_trace(
        go.Bar(
            y=[r["Asset"]],
            x=[dur_days],
            base=[start_day],
            marker=dict(color=bar_color),
            customdata=[
                [
                    start_day,
                    dur_days,
                    end_day,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    edge_id_for_asset,
                ]
            ],
            orientation="h",
            name=r["Asset"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Edge: %{customdata[5]}<br>"
                "Duration: %{customdata[1]:.1f} days<br>"
                "Start: %{customdata[3]}<br>"
                "End: %{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )

def _tick_label(day_offset: int) -> str:
    d = base_date + timedelta(days=int(day_offset))
    return d.isoformat()

max_end = float((df_bar["start_day"] + df_bar["duration_days"]).max())
tick_step = 7
max_tick = int(math.ceil(max_end / tick_step) * tick_step)
tickvals = list(range(0, max_tick + tick_step, tick_step))
ticktext = [_tick_label(v) for v in tickvals]

# Dynamically size the chart so all assets are visible.
# A fixed height clips rows when there are many assets.
row_height_px = 28
base_padding_px = 90
chart_height = int(base_padding_px + row_height_px * max(1, len(df_bar)))

fig_timeline.update_layout(
    barmode="stack",
    height=chart_height,
    margin={"l": 0, "r": 20, "t": 0, "b": 20},
    showlegend=False,
)
fig_timeline.update_yaxes(autorange="reversed")
fig_timeline.update_xaxes(title_text="Date", tickmode="array", tickvals=tickvals, ticktext=ticktext)

with col_right:
    st.plotly_chart(fig_timeline, width="stretch")

# Show a table of schedule values (below the chart)
with col_right:
    st.write("Schedule table (start date computed by flooring the provided start float):")
    st.dataframe(
        df_schedule.assign(
            Start=df_schedule.Start.dt.date.astype(str),
            End=df_schedule.End.dt.date.astype(str),
        )
    )

# Footer with weighted objectives
st.subheader("Objectives")
wo = best_result.get("objectives", {})
if isinstance(wo, dict) and wo:
    for k, v in wo.items():
        st.write(f"{k}: {v}")
else:
    st.write("No objectives found in result.")

st.subheader("Weighted objectives (mocked)")
# Some test fixtures of `best_result` may use the key `objectives` instead of
# `weighted_objectives`. Fall back safely and show sensible defaults.
if "weighted_objectives" in best_result and isinstance(best_result["weighted_objectives"], dict):
    wo = best_result["weighted_objectives"]
else:
    obj = best_result.get("objectives", {})
    wo = {
        "condition_penalty": obj.get("avg_condition", "N/A"),
        "travel_penalty": obj.get("avg_travel_time", "N/A"),
        "cost_penalty": obj.get("total_cost", "N/A"),
    }

st.write(f"Condition penalty: {wo.get('condition_penalty')}")
st.write(f"Travel penalty: {wo.get('travel_penalty')}")
st.write(f"Cost penalty: {wo.get('cost_penalty')}")

# Raw result at the bottom
st.divider()
with st.expander("Raw result JSON", expanded=False):
    st.json(best_result)

# If user clicked Run, execute src/main.py with provided arguments and show output
if 'run_now' in globals() and run_now:
    project_root = Path(__file__).resolve().parents[1]
    main_script = project_root / "src" / "main.py"
    cmd = [sys.executable, str(main_script),
           "--weights", f"{weights[0]}", f"{weights[1]}", f"{weights[2]}",
           "--horizon", str(float(horizon)),
           "--population", str(int(population)),
           "--cx", str(float(cx)),
           "--mut", str(float(mut)),
           "--generations", str(int(generations)),
           "--seed", str(int(seed))]

    # Show command for debugging
    st.sidebar.write("Running command:")
    st.sidebar.code(" ".join(cmd))
    print(f"[DEBUG frontend] running subprocess cmd: {' '.join(cmd)}", flush=True)

    # Update status area and run subprocess
    status_area.info("Queued — starting subprocess...")
    with st.spinner("Running NSGA-II (this may take a while)..."):
        try:
            status_area.info("Running NSGA-II...")
            proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
            rc = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except Exception as e:
            rc = None
            stdout = ""
            stderr = str(e)
            status_area.error(f"Failed to start subprocess: {e}")
    print(f"[DEBUG frontend] subprocess finished rc={rc}", flush=True)
    print(f"[DEBUG frontend] subprocess stdout sample:\n{stdout[:800]}", flush=True)
    print(f"[DEBUG frontend] subprocess stderr sample:\n{stderr[:800]}", flush=True)

    # Finalize status
    if rc == 0:
        status_area.success(f"Finished successfully (rc={rc}) ✅")
    else:
        status_area.error(f"Finished with return code={rc}")

    with st.expander("Stdout", expanded=True):
        st.text(stdout)
    with st.expander("Stderr", expanded=False):
        st.text(stderr)

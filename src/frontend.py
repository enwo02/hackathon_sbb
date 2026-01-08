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

st.set_page_config(page_title="Schedule Visualizer", layout="wide")

st.title("Hardcoded Frontend — Map + Calendar View")
st.markdown(
    "This is a self-contained Streamlit frontend that visualizes a mocked node/edge map and a simple calendar (Gantt) for assets."
)

# Hardcoded "best result" (mocked, matches the structure you showed)
best_result = {
    "best_schedule": {
        "A1": 3.1712530261677268,
        "A2": 1.508604770429887,
        "A3": 0.0,
        "A4": 60.61600586577646,
    },
    "weighted_objectives": {
        "condition_penalty": 0.024060787510129356,
        "travel_penalty": 0.09585103276242014,
        "cost_penalty": 1134000.0,
    },
}

st.subheader("Raw Result JSON")
st.json(best_result)

# Part 1: Map with nodes and edges (hardcoded 5 nodes)
st.subheader("Network map")

# Load nodes from CSV (data/nodes.csv). Format:
# node_id,location_x,location_y  (location_x=lat, location_y=lon)
data_dir = Path(__file__).resolve().parents[1] / "data"
nodes_csv_path = data_dir / "nodes.csv"
edges_csv_path = data_dir / "edges.csv"

fallback_nodes = [
    {"id": "N1", "lat": 47.3769, "lon": 8.5417},  # Zurich
    {"id": "N2", "lat": 46.9480, "lon": 7.4474},  # Bern
    {"id": "N3", "lat": 46.2044, "lon": 6.1432},  # Geneva
    {"id": "N4", "lat": 47.5596, "lon": 7.5886},  # Basel
    {"id": "N5", "lat": 46.5197, "lon": 6.6323},  # Lausanne
]

try:
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

try:
    df_edges = pd.read_csv(edges_csv_path)
    required_edge_cols = {"start_node", "end_node"}
    if not required_edge_cols.issubset(set(df_edges.columns)):
        raise ValueError(
            f"edges.csv must contain columns {sorted(required_edge_cols)}; got {sorted(df_edges.columns)}"
        )
    df_edges = df_edges.dropna(subset=["start_node", "end_node"]).copy()
    edges = [
        (str(r.start_node), str(r.end_node))
        for r in df_edges.itertuples(index=False)
    ]
    if len(edges) == 0:
        edges = fallback_edges
except Exception as e:
    st.warning(f"Could not read edges from {edges_csv_path}: {e}. Falling back to hardcoded edges.")
    edges = fallback_edges

# Build coordinate maps
coord = {n["id"]: (n["lon"], n["lat"]) for n in nodes}

# Filter edges that reference unknown nodes; if nothing remains, build a simple chain.
edges = [(a, b) for (a, b) in edges if a in coord and b in coord]
if len(edges) == 0:
    node_ids = [n["id"] for n in nodes]
    edges = list(zip(node_ids, node_ids[1:])) if len(node_ids) > 1 else []

# Create a Plotly figure: lines for edges and scatter for nodes
edge_traces = []
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
        style="open-street-map",
        center=dict(lat=float(center_lat), lon=float(center_lon)),
        zoom=10,
    ),
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
    height=450,
)

st.plotly_chart(fig_map, width='stretch')

# Part 2: Calendar / Gantt view for assets
st.subheader("Calendar view (assets)")

# Assets to show. We'll derive start day from the best_result values.
assets = list(best_result["best_schedule"].keys())
start_values = best_result["best_schedule"]

# Base date for the visualization
base_date = date(2026, 1, 1)

rows = []
for asset in assets:
    raw_start = start_values[asset]
    # For the mocked calendar, interpret the start as whole-day offset (floor)
    start_day = int(math.floor(raw_start))
    start_dt = base_date + timedelta(days=start_day)
    end_dt = start_dt + timedelta(days=4)  # fixed length of 4 days as requested
    # Use pandas timestamps (JSON-serializable through Plotly) and keep them separate from any timedelta objects
    rows.append(
        {
            "Asset": asset,
            "Start": pd.Timestamp(start_dt),
            "End": pd.Timestamp(end_dt),
            "raw_start": float(raw_start),
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

# Show a table of schedule values
st.write("Schedule table (start date computed by flooring the provided start float):")
st.dataframe(df_schedule.assign(Start=df_schedule.Start.dt.date.astype(str), End=df_schedule.End.dt.date.astype(str)))

# Create a Gantt-like timeline (manual) to avoid datetime/timedelta JSON issues.
# We plot in "days since base_date" and format tick labels ourselves.
df_bar = df_schedule.copy()
df_bar["start_day"] = (df_bar["Start"].dt.normalize() - pd.Timestamp(base_date)).dt.days
df_bar["duration_days"] = (df_bar["End"].dt.normalize() - df_bar["Start"].dt.normalize()).dt.days

fig_timeline = go.Figure()
for _, r in df_bar.iterrows():
    fig_timeline.add_trace(
        go.Bar(
            y=[r["Asset"]],
            x=[int(r["duration_days"])],
            base=[int(r["start_day"])],
            orientation="h",
            name=r["Asset"],
            hovertemplate=(
                "<b>%{y}</b><br>Start day: %{base}<br>Duration: %{x} days<extra></extra>"
            ),
        )
    )

def _tick_label(day_offset: int) -> str:
    d = base_date + timedelta(days=int(day_offset))
    return d.isoformat()

max_end = int((df_bar["start_day"] + df_bar["duration_days"]).max())
tick_step = 7
tickvals = list(range(0, max_end + tick_step, tick_step))
ticktext = [_tick_label(v) for v in tickvals]

fig_timeline.update_layout(
    title="Assets timeline (each has length = 4 days)",
    barmode="stack",
    height=300,
    margin={"l": 100, "r": 20, "t": 40, "b": 20},
    showlegend=False,
)
fig_timeline.update_yaxes(autorange="reversed")
fig_timeline.update_xaxes(title_text="Date", tickmode="array", tickvals=tickvals, ticktext=ticktext)

st.plotly_chart(fig_timeline, width="stretch")

# Footer with weighted objectives
st.subheader("Weighted objectives (mocked)")
wo = best_result["weighted_objectives"]
st.write(f"Condition penalty: {wo['condition_penalty']}")
st.write(f"Travel penalty: {wo['travel_penalty']}")
st.write(f"Cost penalty: {wo['cost_penalty']}")

st.info(
    "This frontend is fully hardcoded for demo purposes. Remove the mocks and load your CSVs from the data/ folder when you integrate it into the main app."
)

import streamlit as st
import json
from components.reactflow_editor import reactflow_editor



# Initialize example graph only once
if "fmu_graph" not in st.session_state:
    st.session_state.fmu_graph = {
        "nodes": [
            {
                "id": "Satellite",
                "data": {"label": "Satellite"},
                "position": {"x": 100, "y": 150}
            },
            {
                "id": "Controller",
                "data": {"label": "Controller"},
                "position": {"x": 400, "y": 150}
            }
        ],
        "edges": [
            {
                "id": "e1",
                "source": "Satellite",
                "target": "Controller",
                "label": "v -> v_meas",
                "animated": True
            }
        ]
    }

st.subheader("🔧 FMU Block Diagram")

# Render the interactive diagram
graph_data = st.session_state.fmu_graph
result = reactflow_editor(args=graph_data, key="reactflow")

# Store result from the editor
if result:
    st.session_state.fmu_graph = result

# JSON editor to view/edit the graph manually
st.subheader("📝 Edit FMU Graph JSON")

graph_json_str = st.text_area(
    "Graph JSON",
    value=json.dumps(st.session_state.fmu_graph, indent=2),
    height=300,
)
st.write("Sent to component:", graph_data)

# Update state if user modifies JSON
try:
    new_graph = json.loads(graph_json_str)
    st.session_state.fmu_graph = new_graph
    st.success("Graph updated from JSON editor.")
except json.JSONDecodeError:
    st.error("Invalid JSON format")


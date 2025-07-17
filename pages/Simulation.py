import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import json
from fmpy import read_model_description
from pathlib import Path


st.set_page_config(page_title="FMU Block Diagram", layout="wide")
st.title("📦 FMU Diagram + Simulation")

# Temporary directory for FMU storage
fmu_dir = tempfile.mkdtemp()

# Initialize session state
if "fmu_meta" not in st.session_state:
    st.session_state.fmu_meta = {}
if "fmu_graph" not in st.session_state:
    st.session_state.fmu_graph = {"nodes": [], "edges": []}

# Sidebar file uploader
st.sidebar.header("Upload FMUs")
uploaded_files = st.sidebar.file_uploader("Select FMUs", type="fmu", accept_multiple_files=True)

# Load and parse FMUs
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(fmu_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        try:
            model_description = read_model_description(file_path)
            inputs = [v.name for v in model_description.modelVariables if v.causality == "input"]
            outputs = [v.name for v in model_description.modelVariables if v.causality == "output"]

            st.session_state.fmu_meta[file.name] = {
                "inputs": inputs,
                "outputs": outputs,
                "modelIdentifier": model_description.modelExchange.modelIdentifier
            }

            # Add node to graph
            node_id = os.path.splitext(file.name)[0]
            st.session_state.fmu_graph["nodes"].append({
                "id": node_id,
                "label": node_id,
                "fmu_file": file.name,
                "position": {"x": 100 * len(st.session_state.fmu_graph["nodes"]), "y": 100}
            })

        except Exception as e:
            st.warning(f"Failed to read FMU '{file.name}': {e}")



st.subheader("📊 FMU Block Diagram (ReactFlow View)")

graph_data = st.session_state.fmu_graph

# Inject JSON into HTML file
html_template = Path("static/fmu_diagram.html").read_text()
html_with_graph = html_template.replace(
    "window.initialGraph || { nodes: [], edges: [] }",
    f"{{ nodes: {json.dumps(graph_data['nodes'])}, edges: {json.dumps(graph_data['edges'])} }}"
)

# Show the iframe
components.html(html_with_graph, height=600, scrolling=False)


# Display uploaded FMU metadata
if st.session_state.fmu_meta:
    st.subheader("Uploaded FMUs")
    for fname, meta in st.session_state.fmu_meta.items():
        with st.expander(fname):
            st.json(meta)

# Editable graph JSON
st.subheader("FMU Diagram (JSON Editor)")
st.markdown("Use this as a placeholder for block diagram editing.")

# Editable graph JSON area
graph_json = st.text_area(
    "FMU Graph JSON", 
    value=json.dumps(st.session_state.fmu_graph, indent=2),
    height=300
)

# Update graph from editor
try:
    st.session_state.fmu_graph = json.loads(graph_json)
except json.JSONDecodeError:
    st.error("Invalid JSON format in FMU graph editor")

# Placeholder for simulation
if st.button("Run Simulation"):
    st.info("Simulation engine not yet implemented.")
    st.json(st.session_state.fmu_graph)

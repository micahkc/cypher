import streamlit as st
import tempfile
import os
import json
from fmpy import read_model_description
from components.reactflow_editor import reactflow_editor
import hashlib

st.set_page_config(page_title="FMU Block Diagram", layout="wide")
st.title("Simulation - Coming Soon")

# Setup state
fmu_dir = tempfile.mkdtemp()
if "fmu_meta" not in st.session_state:
    st.session_state.fmu_meta = {}
if "fmu_graph" not in st.session_state:
    st.session_state.fmu_graph = {"nodes": [], "edges": []}
if "graph_update_counter" not in st.session_state:
    st.session_state.graph_update_counter = 0

# File uploader
st.sidebar.header("Upload FMUs")
uploaded_files = st.sidebar.file_uploader("Select FMUs", type="fmu", accept_multiple_files=True)

# Process FMUs
if uploaded_files:
    existing_ids = {node["id"] for node in st.session_state.fmu_graph["nodes"]}
    for file in uploaded_files:
        if file.name in st.session_state.fmu_meta:
            continue  # already processed

        file_path = os.path.join(fmu_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        try:
            model_description = read_model_description(file_path)
            inputs = [v.name for v in model_description.modelVariables if v.causality == "input"]
            outputs = [v.name for v in model_description.modelVariables if v.causality == "output"]

            node_id = os.path.splitext(file.name)[0]
            if node_id not in existing_ids:
                st.session_state.fmu_graph["nodes"].append({
                    "id": node_id,
                    "data": {
                        "label": node_id,
                        "inputs": inputs,
                        "outputs": outputs
                    },
                    "fmu_file": file.name,
                    "position": {"x": 100 * len(st.session_state.fmu_graph["nodes"]), "y": 100}
                })


            st.session_state.fmu_meta[file.name] = {
                "inputs": inputs,
                "outputs": outputs,
                "modelIdentifier": model_description.modelExchange.modelIdentifier
            }

            # Trigger graph refresh
            st.session_state.graph_update_counter += 1

        except Exception as e:
            st.warning(f"Failed to read FMU '{file.name}': {e}")

# Show diagram
st.subheader("Diagram Editor")

graph_data = st.session_state.fmu_graph
graph_key = hashlib.md5(json.dumps(graph_data, sort_keys=True).encode()).hexdigest()
unique_key = f"reactflow_fmu_{graph_key}_{st.session_state.graph_update_counter}"

result = reactflow_editor(args=graph_data, key=unique_key)
# if result:
#     st.session_state.fmu_graph = result
    # st.json(result)

# Only update session state if the result is different
if result and result != st.session_state.fmu_graph:
    st.session_state.fmu_graph = result
    st.rerun()  # optional: force Streamlit to rerun to update view mode


# # Metadata
# if st.session_state.fmu_meta:
#     st.subheader("📂 Uploaded FMUs")
#     for fname, meta in st.session_state.fmu_meta.items():
#         with st.expander(fname):
#             st.json(meta)

# Editor
with st.expander("Graph JSON Editor"):
    mode = st.radio("Editor Mode", ["View", "Edit"], horizontal=True)

    if mode == "View":
        viewer = st.empty()
        viewer.code(json.dumps(st.session_state.fmu_graph, indent=2), language="json")
    else:
        graph_json = st.text_area(
            "Edit Graph JSON below (nodes + edges):",
            value=json.dumps(st.session_state.fmu_graph, indent=2),
            height=300,
            key="graph_editor_text"  # use key so updates reflect correctly
        )


        if st.button("Refresh Display"):
            try:
                st.session_state.fmu_graph = json.loads(graph_json)
                st.session_state.graph_update_counter += 1
                st.rerun()
                st.success("Graph updated from editor!")
            except json.JSONDecodeError:
                st.error("Invalid JSON format.")

    st.download_button(
        label="Export Graph",
        data=json.dumps(st.session_state.fmu_graph, indent=2),
        file_name="fmu_graph.json",
        mime="application/json"
    )

# Placeholder
if st.button("▶️ Run Simulation"):
    st.info("Simulation not yet implemented.")
    st.json(st.session_state.fmu_graph)

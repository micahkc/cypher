import streamlit as st
import subprocess
from pathlib import Path
import importlib.util
import sys
import matplotlib.pyplot as plt
import numpy as np
import io
import os

st.set_page_config(layout="wide")
st.title("Rumoca")

generated_dir = Path("generated")
generated_dir.mkdir(exist_ok=True)

# Initialize session state keys if missing
for key in ["modelica_path", "generated_path", "export_format", "generated_code", "modelica_code"]:
    if key not in st.session_state:
        st.session_state[key] = None

# File uploader
uploaded_file = st.file_uploader("Upload a Modelica (.mo) file", type=["mo"])

# Export format options
export_formats = {
    "SymPy": "resources/sympy.jinja",
    "CasADi": "resources/casadi_dae.jinja",
    "JSON": "json"
}

selected_format = None
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Export to SymPy"):
        selected_format = "SymPy"
with col2:
    if st.button("Export to CasADi"):
        selected_format = "CasADi"
with col3:
    if st.button("Export to JSON"):
        selected_format = "JSON"

# When file is uploaded, display editable Modelica code
if uploaded_file:
    modelica_path = generated_dir / uploaded_file.name
    original_code = uploaded_file.read().decode("utf-8")

    # Save and store
    st.session_state.modelica_path = modelica_path
    st.session_state.modelica_code = original_code

    with st.expander("View/Edit Modelica Code", expanded=True):
        edited_code = st.text_area("Modelica File Contents", original_code, height=400, key="modelica_editor")
        modelica_path.write_text(edited_code)
        st.session_state.modelica_code = edited_code

# Export and code generation
if st.session_state.modelica_path and selected_format:
    output_py = generated_dir / f"{st.session_state.modelica_path.stem}_{selected_format}.py"
    template_path = Path(export_formats[selected_format]).resolve()
    manifest_path = Path("../rumoca").resolve() / "Cargo.toml"

    cmd = [
        "cargo", "run",
        "--manifest-path", str(manifest_path),
        "--",
        str(st.session_state.modelica_path),
        "-t", str(template_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_py.write_text(result.stdout)

        st.session_state.generated_path = output_py
        st.session_state.generated_code = result.stdout
        st.session_state.export_format = selected_format

        st.success(f"{selected_format} export successful!")

    except subprocess.CalledProcessError as e:
        st.error(f"Export failed:\n{e.stderr}")

# Show generated code
if st.session_state.generated_code:
    with st.expander("View/Edit Generated Code", expanded=False):
        updated_code = st.text_area("Generated Python Code", st.session_state.generated_code, height=400, key="generated_editor")
        Path(st.session_state.generated_path).write_text(updated_code)
        st.session_state.generated_code = updated_code

    st.download_button(
        label=f"Download {st.session_state.export_format} Output",
        data=Path(st.session_state.generated_path).read_bytes(),
        file_name=Path(st.session_state.generated_path).name,
        mime="text/plain"
    )

# Simulation section (only for CasADi export)
if st.session_state.generated_path and st.session_state.export_format == "CasADi":
    with st.expander("Run Simulation"):
        t0 = st.number_input("Start Time (t0)", value=0.0, format="%.5f")
        tf = st.number_input("End Time (tf)", value=8.0, format="%.5f")
        dt = st.number_input("Time Step (dt)", value=0.01, format="%.5f")

        if st.button("Run Simulation"):
            try:
                # Import generated module dynamically
                spec = importlib.util.spec_from_file_location("sim_module", str(st.session_state.generated_path))
                sim_module = importlib.util.module_from_spec(spec)
                sys.modules["sim_module"] = sim_module
                spec.loader.exec_module(sim_module)

                if hasattr(sim_module, "Model"):
                    model = sim_module.Model(st.session_state.modelica_path.stem)

                    tgrid, res = model.simulate(t0=t0, tf=tf, dt=dt)

                    st.success("Simulation completed!")

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(tgrid, res['xf'].T, label=[str(x) for x in model.dae.x()])
                    ax.set_xlabel("Time (s)", fontsize=14)
                    ax.set_ylabel("States", fontsize=14)
                    ax.legend(fontsize=12)
                    ax.grid(True)

                    st.pyplot(fig)

                    # Save plot to buffer and add download button
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)

                    st.download_button(
                        label="Download Plot as PNG",
                        data=buf,
                        file_name="simulation_plot.png",
                        mime="image/png"
                    )
                else:
                    st.error("No class named `Model` found in generated Python file.")
            except Exception as e:
                st.exception(f"Simulation error: {e}")

import streamlit as st
from streamlit_ace import st_ace
import subprocess
from pathlib import Path
import importlib.util
import sys
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import traceback


st.set_page_config(layout="wide")
st.title("Rumoca")

generated_dir = Path("generated")
generated_dir.mkdir(exist_ok=True)

# CSS
st.markdown("""
    <style>
    section > div:nth-child(3) [data-testid="stExpander"] {
        background-color: #e6f2ff; /* Import = light blue */
    }
    section > div:nth-child(4) [data-testid="stExpander"] {
        background-color: #fff4e6; /* Export = light orange */
    }
    </style>
""", unsafe_allow_html=True)




# Initialize session state keys if missing
for key in ["modelica_path", "generated_path", "export_format", "generated_code", "modelica_code"]:
    if key not in st.session_state:
        st.session_state[key] = None

HERE = Path(__file__).parent  # Points to web_interface/
TEMPLATE_DIR = HERE / "resources"

# export_formats = {
#     "sympy": TEMPLATE_DIR / "sympy.jinja",
#     "casadi": TEMPLATE_DIR / "casadi_dae.jinja",
# }

# template_path = export_formats[selected_format]
# Export format options
export_formats = {
    "SymPy": "resources/sympy.jinja",
    "CasADi": "resources/casadi_dae.jinja",
    "JSON": "json"
}

# File uploader
with st.expander("Modelica Model", expanded=True):
    uploaded_file = st.file_uploader("Upload a Modelica (.mo) file", type=["mo"])

    if uploaded_file:
        modelica_path = generated_dir / uploaded_file.name
        original_code = uploaded_file.read().decode("utf-8")

        if "last_uploaded_name" not in st.session_state:
            st.session_state.last_uploaded_name = None

        # If a new file is uploaded, reset the editor and path
        if uploaded_file.name != st.session_state.last_uploaded_name:
            st.session_state.modelica_code = original_code
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.modelica_path = modelica_path
            modelica_path.write_text(original_code)

        edited_code = st_ace(
            value=st.session_state.modelica_code,
            language= "c_cpp", #"modelica",
            theme="github",
            key=f"modelica_editor_{uploaded_file.name}",
            height=400,
            show_gutter=True,
            show_print_margin=False,
            wrap=True
        )

        if edited_code != st.session_state.modelica_code:
            modelica_path.write_text(edited_code)
            st.session_state.modelica_code = edited_code




col1, col2 = st.columns([3, 1])
with col1:
    selected_format = st.selectbox("Select Export Format", options=list(export_formats.keys()))
# with col2:
#     generate_clicked = st.button("Generate")

# Export and code generation
if st.session_state.modelica_path and selected_format:
    output_py = generated_dir / f"{st.session_state.modelica_path.stem}_{selected_format}.py"
    # template_path = Path(export_formats[selected_format]).resolve()
    template_path = export_formats[selected_format]

    project_root = os.path.dirname(os.path.abspath(__file__))
    rumoca_path = os.path.join(project_root, "..", "..", "rumoca", "target/release/rumoca")
    
    # manifest_path = Path("../rumoca").resolve() / "Cargo.toml"

    # cmd = [
    #     "cargo", "run",
    #     "--manifest-path", str(rumoca_path),
    #     "--",
    #     str(st.session_state.modelica_path),
    #     "-t", str(template_path)
    # ]
    cmd = [
        str(rumoca_path),
        str(st.session_state.modelica_path),
        "-t", str(template_path)
    ]   

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_py.write_text(result.stdout)

        st.session_state.generated_path = output_py
        st.session_state.generated_code = result.stdout
        st.session_state.export_format = selected_format
        st.session_state.error_message = None  # Clear any previous errors

    except subprocess.CalledProcessError as e:
        st.session_state.generated_code = None
        st.session_state.generated_path = None
        st.session_state.error_message = {
            "cmd": ' '.join(cmd),
            "stdout": e.stdout,
            "stderr": e.stderr
        }

    if st.session_state.get("error_message"):
        st.error("Rumoca export failed!")
        st.code(f"Command:\n{st.session_state.error_message['cmd']}", language="bash")
        
        with st.expander("STDOUT"):
            st.code(st.session_state.error_message["stdout"] or "[no output]", language="bash")
        
        with st.expander("STDERR"):
            st.code(st.session_state.error_message["stderr"] or "[no error]", language="bash")


# Show generated code
if st.session_state.generated_code:


    with st.expander(f"Generated {st.session_state.export_format} Code", expanded=False):
        updated_code = st_ace(
            value=st.session_state.generated_code,  # This must be correct session state
            language="python",
            theme="monokai",
            key=f"generated_editor_{st.session_state.modelica_path.stem}_{st.session_state.export_format}",
            height=400,
            show_gutter=True,
            show_print_margin=False,
            wrap=True
        )

        # Only update if something actually changed
        if updated_code != st.session_state.generated_code:
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

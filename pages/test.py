import streamlit as st
import subprocess
from pathlib import Path

st.title("Test Rumoca with Known Inputs")

# Hardcoded paths
model_file = Path("resources/msd.mo").resolve()
template_file = Path("resources/sympy.jinja").resolve()
rumoca_manifest = Path("../rumoca/Cargo.toml").resolve()

# Check if files exist
if not model_file.exists():
    st.error(f"Missing model file: {model_file}")
elif not template_file.exists():
    st.error(f"Missing template file: {template_file}")
elif not rumoca_manifest.exists():
    st.error(f"Missing rumoca Cargo.toml: {rumoca_manifest}")
else:
    # Construct cargo run command
    cmd = [
        "cargo", "run",
        "--manifest-path", str(rumoca_manifest),
        "--",
        str(model_file),
        "-t", str(template_file)
    ]

    st.code(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        st.success("Rumoca ran successfully!")
        st.text_area("stdout", result.stdout, height=300)
        if result.stderr.strip():
            st.text_area("stderr", result.stderr, height=150)

    except subprocess.CalledProcessError as e:
        st.error("Rumoca failed.")
        st.text_area("stdout", e.stdout or "None")
        st.text_area("stderr", e.stderr or "None")
    except FileNotFoundError:
        st.error("Cargo not found. Is Rust installed and in your PATH?")

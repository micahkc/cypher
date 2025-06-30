import streamlit as st

st.set_page_config(page_title="CP Analyze", layout="wide")
st.title("CP Analyze")
st.markdown("""
**CP Analyze** is a unified platform for analyzing cyber-physical systems through modeling, simulation, and mathematical analysis.

It integrates:
- **Rumoca**: Translate Modelica models to other representations
- **CP Reach**: Perform reachability analysis using Lyapunov and contraction-based methods
- **CP Analyze Core**: Import FMUs, simulate dynamics, plot results, and run advanced verification
""")

st.header("Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.image("images/cp_analyze.png", use_container_width=True)
    st.subheader("CP Analyze Core")
    st.markdown("""
    Tools for analying cyber-physical systems including FMU import, simulation, plotting, and mathematical analysis.
    """)

with col2:
    st.image("images/rumoca.png", use_container_width=True)
    st.subheader("Rumoca")
    st.markdown("""
    A Modelica-to-target converter supporting **SymPy**, **CasADi**, and **JSON**.
    """)

with col3:
    st.image("images/cp_reach.png", use_container_width=True)
    st.subheader("CP Reach")
    st.markdown("""
    Reachability analysis engine using nonlinear control theory to bound what states a system may reach given various disturbances.
    """)

st.divider()
st.markdown("© 2025 CP Analyze | Purdue University")

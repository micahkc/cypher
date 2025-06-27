import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
# from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../')
import cp_reach as cp
# import cp_reach.quadrotor.log_linearized as qr

st.set_page_config(page_title="Quadrotor", page_icon="🚁", layout="wide")
st.markdown("<h1 style='text-align: center;'>Quadrotor</h1>", unsafe_allow_html=True)

# Default values for Quadrotor
default_data = {
    "thrust_disturbance": 2.0,
    "gyro_disturbance": 2.4
}

# Initialize session state for configuration if not already
if "quadrotor_config" not in st.session_state:
    st.session_state.quadrotor_config = default_data.copy()

# Create columns for layout
col1, col2 = st.columns([1,2])

# Configure quadrotor Parameters and File Uploader in the first column
with col1:
    st.markdown("### Upload Quadrotor Configuration File (JSON)")
    uploaded_file = st.file_uploader("Choose a JSON file", type=["json"])

    # If the file is uploaded, update the configuration
    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.read().decode("utf-8")
            loaded_json = json.loads(file_contents)
            # Update session state with the uploaded config
            st.session_state.quadrotor_config.update(loaded_json)
            st.success("JSON file uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to read JSON file: {e}")

    # Retrieve the config from session state
    config = st.session_state.quadrotor_config


   


    st.markdown("### Configure Quadrotor Parameters")

    config_keys = {
        "thrust_disturbance": "Thrust Disturbance (N)",
        "gyro_disturbance": "Gyro Disturbance (deg/s)"
    }

    # Ensure all expected keys are present in the config
    for key, default_val in default_data.items():
        if key not in config:
            config[key] = default_val

    # Input fields with values from session state
    for key, label in config_keys.items():
        config[key] = st.number_input(
            label,
            value=config.get(key, default_data[key]),
            key=key
        )

    # Input for custom file name
    custom_filename = st.text_input("Enter a custom file name (without extension):", "quadrotor_config")

    # Download configuration button
    config_json = json.dumps(config, indent=4)
    st.download_button(
        label="Download Configuration",
        data=config_json,
        file_name=f"{custom_filename}.json",
        mime="application/json"
    )


     # Upload Trajectory CSV
    st.markdown("### Upload Trajectory CSV")
    trajectory_file = st.file_uploader("Choose a CSV file", type=["csv"], key="trajectory")

    # Store trajectory data in session state
    if trajectory_file is not None:
        try:
            df_trajectory = pd.read_csv(trajectory_file)
            st.session_state.trajectory_data = df_trajectory
            st.success("Trajectory CSV uploaded successfully!")
            st.dataframe(df_trajectory.head())  # Show a preview of the uploaded CSV
        except Exception as e:
            st.error(f"Failed to read CSV file: {e}")


# Analyze quadrotor in the second column
with col2:
    with stylable_container(
        "green",
        css_styles="""
        button {
            background-color: #3CB371;
            color: black;
            font-size: 24px;
            padding: 20px 40px;
            border-radius: 10px;
            border: none;
            width: 100%;
            cursor: pointer;
        }
        button:hover {
            background-color: #2E8B57;
        }
        """,
    ):
        analyze_button = st.button("Analyze Quadrotor", key="analyze_quadrotor")

    def analyze_quadrotor(config):
        # Replace with trajectory generation from CSV
        ref = cp.sim.multirotor_plan.traj_3()

        # # Create Plot 1: 
        # # fig1, ax1 = plt.subplots(figsize=(6, 6), projection="3d")

        # fig1 = plt.figure(figsize=(6,6))
        # ax1 = fig1.add_subplot(111, projection="3d")

        # cp.sim.multirotor_plan.plot_trajectory3D(ref, ax1)
        # # cp.quadrotor.simple_turn.emi_disturbance(config, ax1)

        # img_bytes1 = BytesIO()
        # fig1.savefig(img_bytes1, format='png')
        # img_bytes1.seek(0)

        # Create Plot 2: Roll Over Analysis
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        # inv_points, mu_total, points, points_theta, ebeta, omegabound, sol_LMI = cp.quadrotor.log_linearized.disturbance(config, ref)
        inv_points, points_algebra, lower_bound, upper_bound, sol, omega_bound = cp.quadrotor.log_linearized.disturbance(config, ref)
        # cp.quadrotor.log_linearized.plot2DInvSet(points, inv_points, ax2)
        flowpipes_list, nominal_traj = cp.flowpipe.flowpipe.flowpipes(
            ref=ref,             # your dict with keys 'x', 'y', 'z'
            step=1,                # number of segments
            w1=config['thrust_disturbance'],              # linear disturbance (scalar or vector)
            omegabound = omega_bound,     # angular disturbance (scalar or vector)
            sol=sol,             # output of SE23LMIs
            axis='xy'            # 'xy' or 'xz'
        )

        cp.flowpipe.flowpipe.plot_flowpipes(nominal_traj, flowpipes_list, ax2, axis='xy')

        img_bytes2 = BytesIO()
        fig2.savefig(img_bytes2, format='png')
        img_bytes2.seek(0)

        # st.pyplot(fig1)
        st.pyplot(fig2)

        return img_bytes2

    if analyze_button:
        img_bytes2 = analyze_quadrotor(st.session_state.quadrotor_config)

        st.download_button(
            label="Download Plot 1: EMI Disturbance",
            data=img_bytes2,
            file_name="quadrotor_emi_dist.png",
            mime="image/png"
        )
        st.download_button(
            label="Download Plot 2: Roll Over",
            data=img_bytes2,
            file_name="quadrotor_roll_over.png",
            mime="image/png"
        )
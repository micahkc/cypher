import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import sys
sys.path.append("../")
from cp_reach.sim.multirotor_plan import planner2, find_cost_function
from cp_reach.flowpipe.flowpipe import flowpipes, plot_flowpipes
from cp_reach.quadrotor import log_linearized

st.set_page_config(page_title="Trajectory & Flowpipe Planner", layout="wide")
st.title("Interactive Trajectory + Flowpipe Generator")

# Constants
poly_deg = 7
min_deriv = 4
bc_deriv = 4
k_time = 1e5

# Initialize session state
if "waypoints" not in st.session_state:
    st.session_state.waypoints = [
        {"pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]},
        {"pos": [5.0, 5.0, 0.0], "vel": [0.0, 0.0, 0.0]},
    ]
if "T_legs" not in st.session_state:
    st.session_state.T_legs = [5.0]
if "ref_traj" not in st.session_state:
    st.session_state.ref_traj = None

# === Waypoint Editor ===

# Always enforce T_legs = len(waypoints) - 1
while len(st.session_state.T_legs) < len(st.session_state.waypoints) - 1:
    st.session_state.T_legs.append(5.0)
while len(st.session_state.T_legs) > len(st.session_state.waypoints) - 1:
    st.session_state.T_legs.pop()


st.subheader("Waypoints")
cols = st.columns([1, 1, 1, 1, 1, 1, 1])
cols[0].markdown("**#**")
cols[1].markdown("**X**")
cols[2].markdown("**Y**")
cols[3].markdown("**Z**")
cols[4].markdown("**Vx**")
cols[5].markdown("**Vy**")
cols[6].markdown("**Vz**")

for i, wp in enumerate(st.session_state.waypoints):
    cols = st.columns([1, 1, 1, 1, 1, 1, 1])
    cols[0].write(f"{i}")
    for j, key in enumerate(["pos", "vel"]):
        for k in range(3):
            idx = 1 + j * 3 + k
            wp[key][k] = cols[idx].number_input(
                f"{key}_{k}_{i}", value=wp[key][k], step=0.1, label_visibility="collapsed"
            )

col_a, col_b = st.columns([1, 1])
if col_a.button("➕ Add Waypoint"):
    st.session_state.waypoints.append({"pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]})
    st.rerun()  # force Streamlit to display the new state immediately

if col_b.button("➖ Remove Last", disabled=len(st.session_state.waypoints) <= 2):
    st.session_state.waypoints.pop()
    st.rerun()



# === Time Editor ===
st.subheader("Segment Durations (T_legs)")
for i in range(len(st.session_state.T_legs)):
    st.session_state.T_legs[i] = st.number_input(
        f"T_leg[{i}]", value=st.session_state.T_legs[i], step=0.1
    )


# === Trajectory Controls: Generate + Sample ===
col_traj, col_sample = st.columns([1, 1])

with col_traj:
    generate_traj = st.button("Generate Trajectory")

with col_sample:
    if st.button("Sample Trajectory"):
        sample_pos = [
            [0, 0, 0],
            [7.04, -0.76, 0],
            [10.04, 1.7, 0],
            [10.22, 6.6, 0],
            [13.33, 8.65, 0],
            [20.15, 8.14, 0],
            [19.6, -1.92, 0],
        ]
        sample_vel = [
            [0, 0, 0],
            [2.37, 0, 0],
            [0.15, 2.67, 0],
            [0.49, 2.28, 0],
            [2.85, -0.23, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        sample_T_legs = [4.67, 2.17, 1.84, 1.92, 5.5, 6.46]

        st.session_state.waypoints = [
            {"pos": [float(x) for x in p], "vel": [float(vv) for vv in v]}
            for p, v in zip(sample_pos, sample_vel)
        ]
        st.session_state.T_legs = [float(t) for t in sample_T_legs]

        st.rerun()





# === Trajectory Generation ===
if generate_traj:
    try:
        pos = [wp["pos"] for wp in st.session_state.waypoints]
        vel = [wp["vel"] for wp in st.session_state.waypoints]
        acc = [[0, 0, 0] for _ in range(len(pos))]
        jerk = [[0, 0, 0] for _ in range(len(pos))]

        bc = np.stack((pos, vel, acc, jerk))
        cost = find_cost_function(
            poly_deg=poly_deg,
            min_deriv=min_deriv,
            rows_free=[],
            n_legs=len(pos) - 1,
            bc_deriv=bc_deriv,
        )

        ref = planner2(
            bc,
            cost,
            len(pos) - 1,
            poly_deg,
            k_time,
            st.session_state.T_legs,
        )

        # Save in session
        st.session_state.ref_traj = ref

        # Plot and store figure — reduced size
        fig = plt.figure(figsize=(3, 2.4))  # smaller figure
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(ref["x"], ref["y"], ref["z"], "k-", label="Trajectory")
        ax.scatter(*zip(*pos), c="r", label="Waypoints")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        st.session_state.ref_fig = fig

        # st.markdown("### Reference Trajectory")
        # # st.pyplot(fig)
        # plt.close(fig)

        # # Trigger rerun to expose state for other blocks
        # st.rerun()

    except Exception as e:
        st.exception(f"Trajectory generation failed: {e}")




if "ref_fig" in st.session_state:
    st.markdown("### Reference Trajectory")
    col1, _ = st.columns([1, 1])  # half-width left column, empty right column
    with col1:
        st.pyplot(st.session_state.ref_fig)
        plt.close(st.session_state.ref_fig)


# === Flowpipe Generation ===
st.subheader("Generate Flowpipe")

thrust_d = st.number_input("Thrust Disturbance", value=2.0, step=0.1)
gyro_d = st.number_input("Gyro Disturbance", value=2.4, step=0.1)

if st.button("Generate Flowpipe"):
    if st.session_state.ref_traj is None:
        st.warning("Generate the trajectory first.")
    else:
        try:
            ref = st.session_state.ref_traj
            config = {"thrust_disturbance": thrust_d, "gyro_disturbance": gyro_d}
            _, _, _, _, sol, omega_bound = log_linearized.disturbance(config, ref)

            # === Flowpipe XY
            fig_xy, ax_xy = plt.subplots(figsize=(10,10))
            fp_xy, nom_xy = flowpipes(
                ref=ref,
                step=1,
                w1=thrust_d,
                omegabound=omega_bound,
                sol=sol,
                axis="xy"
            )
            plot_flowpipes(nom_xy, fp_xy, ax_xy, axis="xy")

            # === Flowpipe XZ
            fig_xz, ax_xz = plt.subplots(figsize=(10, 10))
            fp_xz, nom_xz = flowpipes(
                ref=ref,
                step=1,
                w1=thrust_d,
                omegabound=omega_bound,
                sol=sol,
                axis="xz"
            )
            plot_flowpipes(nom_xz, fp_xz, ax_xz, axis="xz")

            # # === Reference Trajectory (from session state)
            # if "ref_fig" in st.session_state:
            #     st.markdown("### Reference Trajectory")
            #     st.pyplot(st.session_state.ref_fig)
            #     plt.close(st.session_state.ref_fig)

            # === Display Flowpipes Side-by-Side
            st.markdown("### Flowpipe Projections")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**XY Plane**")
                st.pyplot(fig_xy)
                plt.close(fig_xy)

            with col2:
                st.markdown("**XZ Plane**")
                st.pyplot(fig_xz)
                plt.close(fig_xz)


        except Exception as e:
            st.exception(f"Flowpipe generation failed: {e}")


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
import json
import sys
sys.path.append("../")
from cp_reach.sim.multirotor_plan import planner2, find_cost_function
from cp_reach.flowpipe.flowpipe import flowpipes, plot_flowpipes
from cp_reach.quadrotor import log_linearized

# === Persistent State Directory ===
PERSIST_DIR = "generated"
os.makedirs(PERSIST_DIR, exist_ok=True)

def save_state_to_disk():
    with open(os.path.join(PERSIST_DIR, "waypoints.json"), "w") as f:
        json.dump(st.session_state.waypoints, f)
    with open(os.path.join(PERSIST_DIR, "T_legs.json"), "w") as f:
        json.dump(st.session_state.T_legs, f)
    if "ref_traj" in st.session_state and st.session_state.ref_traj is not None:
        np.savez(os.path.join(PERSIST_DIR, "ref_traj.npz"), **st.session_state.ref_traj)

def load_state_from_disk():
    try:
        with open(os.path.join(PERSIST_DIR, "waypoints.json"), "r") as f:
            st.session_state.waypoints = json.load(f)
        with open(os.path.join(PERSIST_DIR, "T_legs.json"), "r") as f:
            st.session_state.T_legs = json.load(f)
        path = os.path.join(PERSIST_DIR, "ref_traj.npz")
        if os.path.exists(path):
            data = np.load(path)
            st.session_state.ref_traj = {k: data[k] for k in data}
    except Exception as e:
        st.warning(f"Could not load persistent session state: {e}")

st.set_page_config(page_title="Trajectory & Flowpipe Planner", layout="wide")
st.title("Interactive Trajectory + Flowpipe Generator")

# === Initialize Session State ===
if "waypoints" not in st.session_state:
    st.session_state.waypoints = [
        {"pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]},
        {"pos": [5.0, 5.0, 0.0], "vel": [0.0, 0.0, 0.0]},
    ]
    st.session_state.T_legs = [5.0]
    st.session_state.ref_traj = None
    load_state_from_disk()

# === Enforce T_legs Length ===
while len(st.session_state.T_legs) < len(st.session_state.waypoints) - 1:
    st.session_state.T_legs.append(5.0)
while len(st.session_state.T_legs) > len(st.session_state.waypoints) - 1:
    st.session_state.T_legs.pop()

# === Waypoint Editor ===
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
                f"{key}_{k}_{i}", value=float(wp[key][k]), step=0.1, label_visibility="collapsed"
            )
save_state_to_disk()

col_a, col_b = st.columns([1, 1])
if col_a.button("➕ Add Waypoint"):
    st.session_state.waypoints.append({"pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]})
    save_state_to_disk()
    st.rerun()

if col_b.button("➖ Remove Last", disabled=len(st.session_state.waypoints) <= 2):
    st.session_state.waypoints.pop()
    save_state_to_disk()
    st.rerun()

# === Time Editor ===
st.subheader("Segment Durations (T_legs)")
for i in range(len(st.session_state.T_legs)):
    st.session_state.T_legs[i] = st.number_input(
        f"T_leg[{i}]", value=float(st.session_state.T_legs[i]), step=0.1
    )
save_state_to_disk()

# === Trajectory Controls ===
col_traj, col_sample = st.columns([1, 1])
with col_traj:
    generate_traj = st.button("Generate Trajectory")

with col_sample:
    if st.button("Sample Trajectory"):
        sample_pos = [
            [0, 0, 0], [7.04, -0.76, 0], [10.04, 1.7, 0], [10.22, 6.6, 0],
            [13.33, 8.65, 0], [20.15, 8.14, 0], [19.6, -1.92, 0]
        ]
        sample_vel = [
            [0, 0, 0], [2.37, 0, 0], [0.15, 2.67, 0], [0.49, 2.28, 0],
            [2.85, -0.23, 0], [0, 0, 0], [0, 0, 0]
        ]
        sample_T_legs = [4.67, 2.17, 1.84, 1.92, 5.5, 6.46]

        st.session_state.waypoints = [
            {"pos": p, "vel": v} for p, v in zip(sample_pos, sample_vel)
        ]
        st.session_state.T_legs = sample_T_legs
        save_state_to_disk()
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
            poly_deg=7,
            min_deriv=4,
            rows_free=[],
            n_legs=len(pos) - 1,
            bc_deriv=4,
        )

        ref = planner2(
            bc, cost, len(pos) - 1, 7, 1e5, st.session_state.T_legs
        )

        st.session_state.ref_traj = ref
        save_state_to_disk()

        fig = plt.figure(figsize=(3, 2.4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(ref["x"], ref["y"], ref["z"], "k-", label="Trajectory")
        ax.scatter(*zip(*pos), c="r", label="Waypoints")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        st.session_state.ref_fig = fig

    except Exception as e:
        st.exception(f"Trajectory generation failed: {e}")

# === Plot Trajectory ===
if "ref_fig" in st.session_state:
    st.markdown("### Reference Trajectory")
    col1, _ = st.columns([1, 1])
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

            fig_xy, ax_xy = plt.subplots(figsize=(10, 10))
            fp_xy, nom_xy = flowpipes(ref, 1, thrust_d, omega_bound, sol, "xy")
            plot_flowpipes(nom_xy, fp_xy, ax_xy, axis="xy")

            fig_xz, ax_xz = plt.subplots(figsize=(10, 10))
            fp_xz, nom_xz = flowpipes(ref, 1, thrust_d, omega_bound, sol, "xz")
            plot_flowpipes(nom_xz, fp_xz, ax_xz, axis="xz")

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

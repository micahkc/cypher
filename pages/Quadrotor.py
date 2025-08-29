import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import sys
import pandas as pd
sys.path.append("../")
import cp_reach
import cp_reach.quadrotor as quadrotor
import cp_reach.utils as utils






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

if "kin_sol" not in st.session_state:
    st.session_state.kin_sol = None

if "tracks" not in st.session_state:
    st.session_state.tracks = []  # list of dicts: {"name", "x", "y", "z", "n"}

    
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
        st.session_state.reset_sol = True
        pos = [wp["pos"] for wp in st.session_state.waypoints]
        vel = [wp["vel"] for wp in st.session_state.waypoints]
        acc = [[0, 0, 0] for _ in range(len(pos))]
        jerk = [[0, 0, 0] for _ in range(len(pos))]

        bc = np.stack((pos, vel, acc, jerk))
        cost = quadrotor.mission.find_cost_function(
            poly_deg=poly_deg,
            min_deriv=min_deriv,
            rows_free=[],
            n_legs=len(pos) - 1,
            bc_deriv=bc_deriv,
        )

        ref = quadrotor.mission.plan_trajectory(
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


vel_d = st.number_input("Velocity Disturbance", value=1.0, step=0.1)
thrust_d = st.number_input("Thrust Disturbance", value=2.0, step=0.1)
gyro_d = st.number_input("Gyro Disturbance", value=2.4, step=0.1)

# === Bound Error ====
st.subheader("Compute Error Bounds")
if st.button("Compute Error Bounds"):
    if st.session_state.ref_traj is None:
        st.warning("Generate the trajectory first.")
    else:
        try:
            ref = st.session_state.ref_traj
            if st.session_state.reset_sol == True:
                # Solve lmi
                ang_vel_points,lower_bound_omega,upper_bound_omega,omega_dist,dynamics_sol,inv_points,lower_bound,upper_bound,kinematics_sol = quadrotor.invariant_set.solve(vel_d, thrust_d, gyro_d, ref)
                st.session_state.kin_sol = kinematics_sol
                st.session_state.dyn_sol = dynamics_sol
                st.session_state.reset_sol = False
            else:
                # Use the saved LMI
                kin_sol = st.session_state.kin_sol
                dyn_sol = st.session_state.dyn_sol
                ang_vel_points,lower_bound_omega,upper_bound_omega,omega_dist,dynamics_sol,inv_points,lower_bound,upper_bound,kinematics_sol = quadrotor.invariant_set.solve(vel_d, thrust_d, gyro_d, ref, dyn_sol, kin_sol)

            # Labels and units
            components_se3 = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
            units_se3 = ['m', 'm', 'm', 'rad', 'rad', 'rad']

            components_omega = ['ωx', 'ωy', 'ωz']
            units_omega = ['rad/s'] * 3

            # Combine data
            components = components_se3 + components_omega
            lower_bounds = np.concatenate([lower_bound, lower_bound_omega])
            upper_bounds = np.concatenate([upper_bound, upper_bound_omega])
            units = units_se3 + units_omega

            # Build dataframe
            df_all = pd.DataFrame({
                "Component": components,
                "Lower Bound": lower_bounds,
                "Upper Bound": upper_bounds,
                "Units": units
            })

            st.dataframe(df_all, use_container_width=True)

        except Exception as e:
            st.exception(f"Calculating error bound failed: {e}")

# === Minimum Attack Search UI ===
st.subheader("Search: Minimum Attack to Push Off Trajectory")

left, right = st.columns([1, 1])

with left:
    st.markdown("**Count the norm along these axes as a violation:**")
    chk_x = st.checkbox("X", value=True, key="atk_chk_x")
    chk_y = st.checkbox("Y", value=True, key="atk_chk_y")
    chk_z = st.checkbox("Z", value=True, key="atk_chk_z")

with right:
    st.markdown("**Attack Type**")
    obj_axis = st.radio("Objective axis", ["Velocity", "Acceleration", "Angular Acceleration"], index=0, horizontal=True, key="atk_obj_axis")

thresh_m = st.number_input("Deviation threshold (meters)", value=0.50, step=0.05, min_value=0.0)
max_search = st.number_input("Max disturbance to search", value=10.0, step=0.5, min_value=0.1)
tol_vel = 0.01
max_iter = 30

def selected_norms(pos_lo, pos_hi, chk_x, chk_y, chk_z, ord=2):
    """
    pos_lo, pos_hi: shape (3,) arrays for [x,y,z] lower/upper bounds
    chk_*: booleans from the checkboxes
    ord: norm order (2 = Euclidean, 1 = L1, np.inf = Linf, etc.)
    """
    mask = np.array([chk_x, chk_y, chk_z], dtype=bool)
    if not mask.any():
        return 0.0, 0.0, 0.0  # nothing selected

    lo_sel = np.asarray(pos_lo)[mask]
    hi_sel = np.asarray(pos_hi)[mask]

    lo_norm = float(np.linalg.norm(lo_sel, ord=ord))
    hi_norm = float(np.linalg.norm(hi_sel, ord=ord))

    return lo_norm, hi_norm

# Helper: evaluate bounds for a given vel disturbance (reuses cached LMI if available)
def _eval_bounds(magnitude):
    ref = st.session_state.ref_traj
    if ref is None:
        return None

    if st.session_state["atk_obj_axis"] == "Velocity":
        vel_d, thrust_d, gyro_d = magnitude, 0.0001, 0.0001

    elif st.session_state["atk_obj_axis"] == "Acceleration":
        vel_d, thrust_d, gyro_d = 0.0001, magnitude, 0.0001

    elif st.session_state["atk_obj_axis"] == "Angular Acceleration":
        vel_d, thrust_d, gyro_d = 0, 0, magnitude

    # Reuse cached LMI solutions when possible
    kin_sol = st.session_state.get("kin_sol", None)
    dyn_sol = st.session_state.get("dyn_sol", None)
    try:
        (ang_vel_points, lower_bound_omega, upper_bound_omega, omega_dist,
        dynamics_sol, inv_points, lower_bound, upper_bound, kinematics_sol) = \
            quadrotor.invariant_set.solve(
                vel_d, thrust_d, gyro_d, ref, dyn_sol, kin_sol
            )
        # Refresh cache (once we have them)
        st.session_state.kin_sol = kinematics_sol
        st.session_state.dyn_sol = dynamics_sol

        # lower/upper for SE(3): [x,y,z, roll, pitch, yaw]
        # we only use the position components here
        pos_lo = lower_bound[:3]
        pos_hi = upper_bound[:3]

        lo_norm, hi_norm = selected_norms(
            pos_lo, pos_hi,
            chk_x=st.session_state["atk_chk_x"],
            chk_y=st.session_state["atk_chk_y"],
            chk_z=st.session_state["atk_chk_z"],
            ord=2,  # Euclidean
        )
        return np.max([lo_norm, hi_norm])
    
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return None

# Predicate: does this vel disturbance violate the threshold?
def _violates(dist_mag):
    pos_error = _eval_bounds(dist_mag)
    if pos_error is None:
        return False
    return pos_error >= thresh_m

if st.button("Search minimum attack"):
    if st.session_state.ref_traj is None:
        st.warning("Generate the trajectory first.")
    else:
        # Exponential search to bracket a violation, then bisection
        lo, hi = 0.0, max(0.1, min(max_search, 0.5))
        # Grow 'hi' until violation or we hit max cap
        while hi < max_search and not _violates(hi):
            lo, hi = hi, min(2.0 * hi, max_search)

        if not _violates(hi):
            st.info("No violation found up to the maximum search bound.")
        else:
            # Bisection on vel disturbance magnitude
            it = 0
            while (hi - lo) > tol_vel and it < max_iter:
                mid = 0.5 * (lo + hi)
                if _violates(mid):
                    hi = mid
                else:
                    lo = mid
                it += 1

            # Report the result and the achieved axis magnitudes at that vel disturbance
            mag_star = hi
            res_star = _eval_bounds(mag_star)
            pos_abs = res_star

            

            if st.session_state["atk_obj_axis"] == "Velocity":
                st.success(f"Minimum velocity disturbance that causes violation: {mag_star:.2f} m/s")

            elif st.session_state["atk_obj_axis"] == "Acceleration":
                st.success(f"Minimum acceleration disturbance that causes violation: {mag_star:.2f} m/s^2")

            elif st.session_state["atk_obj_axis"] == "Angular Acceleration":
                st.success(f"Minimum angular acceleration disturbance that causes violation: {mag_star:.2f} rad/s^2")



# === CSV Upload =======
st.subheader("Overlay CSV logs (x,y,z) on flowpipes")

left_u, right_u = st.columns([3, 2])
with left_u:
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

col_load_a, col_load_b = st.columns([1, 1])
if col_load_a.button("Add to overlays", help="Parse and cache the uploaded CSVs"):
    new_tracks = []
    for f in uploaded_files or []:
        try:
            df = pd.read_csv(f)
            x_col = 'quad_low.position_w_p_w[1].1'
            y_col = 'quad_low.position_w_p_w[2].1'
            z_col = 'quad_low.position_w_p_w[3].1'

            x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
            y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
            z = pd.to_numeric(df[z_col], errors="coerce").to_numpy()

            new_tracks.append({
                "name": f.name.rsplit(".", 1)[0],
                "x": x, "y": y, "z": z,
                "n": len(x),
            })
        except Exception as e:
            st.error(f"Failed to load {getattr(f, 'name', '<unknown>')}: {e}")

    if new_tracks:
        st.session_state.tracks.extend(new_tracks)
        st.success(f"Added {len(new_tracks)} overlay track(s).")

if col_load_b.button("Clear overlays", type="secondary"):
    st.session_state.tracks = []
    st.info("Cleared all overlay tracks.")

# Tiny summary table
if st.session_state.tracks:
    st.caption("Overlays:")
    st.dataframe(
        pd.DataFrame([{"File": t["name"], "Points": t["n"]} for t in st.session_state.tracks]),
        use_container_width=True,
        hide_index=True,
    )

# === Flowpipe Generation ===
st.subheader("Generate Flowpipe")


if st.button("Generate Flowpipe"):
    if st.session_state.kin_sol is None:
        st.warning("Calculate the error bound first.")
    else:
        try:
            kin_sol = st.session_state.kin_sol
            dyn_sol = st.session_state.dyn_sol
            ref = st.session_state.ref_traj
            ang_vel_points,lower_bound_omega,upper_bound_omega,omega_dist,dynamics_sol,inv_points,lower_bound,upper_bound,kinematics_sol = quadrotor.invariant_set.solve(vel_d, thrust_d, gyro_d, ref, dyn_sol, kin_sol)

            # === Flowpipe XY
            fig_xy, ax_xy = plt.subplots(figsize=(10,10))
            fp_xy, nom_xy = utils.plotting.flowpipes(
                ref=ref,
                step=1,
                vel_dist = vel_d,
                accel_dist=thrust_d,
                omega_dist=omega_dist,
                sol=kinematics_sol,
                axis="xy"
            )
            utils.plotting.plot_flowpipes(nom_xy, fp_xy, ax_xy, axis="xy")

            # === Flowpipe XZ
            fig_xz, ax_xz = plt.subplots(figsize=(10, 10))
            fp_xz, nom_xz = utils.plotting.flowpipes(
                ref=ref,
                step=1,
                vel_dist = vel_d,
                accel_dist=thrust_d,
                omega_dist=omega_dist,
                sol=kinematics_sol,
                axis="xz"
            )
            utils.plotting.plot_flowpipes(nom_xz, fp_xz, ax_xz, axis="xz")



            # Simulation overlay
            # --- After utils.plotting.plot_flowpipes(...), overlay any tracks ---

            tracks = st.session_state.get("tracks", [])

            # XY overlay
            for tr in tracks:
                try:
                    print(np.min(tr['x']), np.min(tr['y'][0]), np.min(tr['z'][0]))
                    ax_xy.plot(tr["x"], tr["y"], label=tr["name"], linewidth=1.5, alpha=0.9)
                except Exception as e:
                    st.warning(f"Could not plot XY overlay for {tr['name']}: {e}")

            # Keep legends tidy
            handles_xy, labels_xy = ax_xy.get_legend_handles_labels()
            if labels_xy:
                ax_xy.legend(loc="best", fontsize=9)

            # XZ overlay
            for tr in tracks:
                try:
                    ax_xz.plot(tr["x"], tr["z"], label=tr["name"], linewidth=1.5, alpha=0.9)
                except Exception as e:
                    st.warning(f"Could not plot XZ overlay for {tr['name']}: {e}")

            handles_xz, labels_xz = ax_xz.get_legend_handles_labels()
            if labels_xz:
                ax_xz.legend(loc="best", fontsize=9)






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


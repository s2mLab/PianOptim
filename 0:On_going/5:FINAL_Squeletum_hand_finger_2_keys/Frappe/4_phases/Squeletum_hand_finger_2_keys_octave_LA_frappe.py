"""
 !! Les axes du modèle ne sont pas les mêmes que ceux généralement utilisés en biomécanique : x axe de flexion, y supination/pronation, z vertical
 ici on a : Y -» X , Z-» Y et X -» Z
 """
from casadi import MX, sqrt, if_else, sin, vertcat
import time
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
import biorbd_casadi as biorbd
import pickle
from bioptim import (
    PenaltyNode,
    ObjectiveList,
    PhaseTransitionFcn,
    DynamicsList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    CostType,
    PlotType,
    PhaseTransitionList,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    PenaltyNodeList,
    QAndQDotBounds,
    OdeSolver,
    BiorbdInterface,
    Solver,
    Axis
)

# Constants from data collect
# velocities (vel) and standards deviations (stdev) of the right hand of subject 134

pos_x_0 = 0.10665989
pos_x_1 = 0.25553592
pos_x_2 = 0.26675006
pos_x_3 = 0.39711206
pos_x_4 = 0.38712035
pos_x_5 = 0.9867809

pos_y_0 = 0.40344472
pos_y_1 = 0.37350889
pos_y_2 = 0.38965598
pos_y_3 = 0.34155492
pos_y_4 = 0.33750396
pos_y_5 = 0.39703432

pos_z_0 = 2.76551207
pos_z_1 = 2.74265983
pos_z_2 = 2.76575107
pos_z_3 = 2.73511557
pos_z_4 = 2.72985087
pos_z_5 = 2.75654283

vel_x_0 = 0.52489776
vel_x_1 = -0.00321311
vel_x_2 = 0.64569518
vel_x_3 = 0.04628122
vel_x_4 = -0.01133422
vel_x_5 = 0.31271958

vel_y_0 = 0.54408214
vel_y_1 = 0.01283999
vel_y_2 = 0.28540601
vel_y_3 = 0.02580192
vel_y_4 = 0.09021791
vel_y_5 = 0.42298668

vel_z_0 = 0.7477114
vel_z_1 = 0.17580993
vel_z_2 = 0.6360936
vel_z_3 = 0.3468823
vel_z_4 = -0.03609537
vel_z_5 = 0.38915613

stdev_vel_x_0 = 0.12266391
stdev_vel_x_1 = 0.05459328
stdev_vel_x_2 = 0.08348852
stdev_vel_x_3 = 0.06236412
stdev_vel_x_4 = 0.06251115
stdev_vel_x_5 = 0.10486219

# stdev_vel_y_0 = 0.06590577
# stdev_vel_y_1 = 0.04433499
# stdev_vel_y_2 = 0.08251966
# stdev_vel_y_3 = 0.03813032
# stdev_vel_y_4 = 0.07607116
# stdev_vel_y_5 = 0.0713205

stdev_vel_z_0 = 0.11591871
stdev_vel_z_1 = 0.10771169
stdev_vel_z_2 = 0.081717
stdev_vel_z_3 = 0.09894744
stdev_vel_z_4 = 0.11820802
stdev_vel_z_5 = 0.1479469

mean_time_phase_0 = 0.36574653  # phase 0 is from first marker to second marker
mean_time_phase_1 = 0.10555556
mean_time_phase_2 = 0.40625
mean_time_phase_3 = 0.10387153
mean_time_phase_4 = 1.00338542  # phase 4 is from last marker to first marker


# Function to minimize the difference between transitions
def minimize_difference(all_pn: PenaltyNode):
    return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx


# def custom_func_track_finger_marker_key(all_pn: PenaltyNodeList, marker: str) -> MX:
#     finger_marker_idx = biorbd.marker_index(all_pn.nlp.model, marker)
#     markers = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
#     finger_marker = markers[:, finger_marker_idx]
#     key = ((0.005*sin(137*(finger_marker[1]+0.0129)))
#       / (sqrt(0.001**2 + sin(137*(finger_marker[1] + 0.0129))**2))-0.005)
#
#     # if_else( condition, si c'est vrai fait ca',  sinon fait ca)
#     markers_diff_key = if_else(
#         finger_marker[1] < 0.01,
#         finger_marker[2] - 0,
#         if_else(
#             finger_marker[1] < 0.033,  # condition
#             finger_marker[2] - key,  # True
#             finger_marker[2]-0,  # False
#         )
#     )
#     return markers_diff_key

def custom_func_track_finger_5_above_principal_finger(all_pn: PenaltyNodeList) -> MX:
    finger_marker_idx = biorbd.marker_index(all_pn.nlp.model, "finger_marker")
    markers = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    finger_marker_5_idx = biorbd.marker_index(all_pn.nlp.model, "finger_marker_5")
    markers_5 = BiorbdInterface.mx_to_cx("markers_5", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
    finger_marker_5 = markers_5[:, finger_marker_5_idx]

    markers_diff_key1 = finger_marker_5[2] - finger_marker[2]

    return markers_diff_key1


def custom_func_track_finger_5_on_the_right_of_principal_finger(all_pn: PenaltyNodeList) -> MX:
    finger_marker_idx = biorbd.marker_index(all_pn.nlp.model, "finger_marker")
    markers = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    finger_marker_5_idx = biorbd.marker_index(all_pn.nlp.model, "finger_marker_5")
    markers_5 = BiorbdInterface.mx_to_cx("markers_5", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
    finger_marker_5 = markers_5[:, finger_marker_5_idx]

    markers_diff_key2 = finger_marker[1] - finger_marker_5[1]

    return markers_diff_key2


def custom_func_track_principal_finger_and_finger5_above_bed_key(all_pn: PenaltyNodeList, marker: str) -> MX:
    finger_marker_idx3 = biorbd.marker_index(all_pn.nlp.model, marker)
    markers3 = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
    finger_marker3 = markers3[:, finger_marker_idx3]

    markers_diff_key3 = finger_marker3[2] - -0.01

    return markers_diff_key3


def custom_func_track_roty_principal_finger(all_pn: PenaltyNodeList) -> MX:

    model = all_pn.nlp.model
    rotation_matrix_index = biorbd.segment_index(model, "2proxph_2mcp_flexion")

    q = all_pn.nlp.states["q"].mx
    rotation_matrix = all_pn.nlp.model.globalJCS(q, rotation_matrix_index).to_mx()

    output = vertcat(rotation_matrix[1, 0], rotation_matrix[1, 2], rotation_matrix[0, 1], rotation_matrix[2, 1],
                     rotation_matrix[1, 1] - MX(1))

    rotation_matrix = BiorbdInterface.mx_to_cx("rot_mat", output, all_pn.nlp.states["q"])




    return rotation_matrix


def prepare_ocp(
        biorbd_model_path: str = "/home/lim/Documents/Stage Mathilde/PianOptim/0:On_going/5:FINAL_Squeletum_hand_finger_2_keys/Frappe/4_phases/Squeletum_hand_finger_3D_2_keys_octave_LA_frappe.bioMod",
        ode_solver: OdeSolver = OdeSolver.COLLOCATION(),
        long_optim: bool = False,
) -> OptimalControlProgram:

    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use
    long_optim: bool
        If the solver should solve the precise optimization (500 shooting points) or the approximate (50 points)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path))

    # Average of N frames by phase and the phases time, both measured with the motion capture datas.
    n_shooting = (30, 6, 9, 50)
    phase_time = (0.3, 0.027, 0.058, 0.5)
    tau_min, tau_max, tau_init = -200, 200, 0
    vel_pushing = -0.32298261

    # Find the number of the node at 75 % of the phase 0 and 3 in order to apply the vel_pushing at this node
    three_quarter_node_phase_1 = ceil(0.75 * n_shooting[1])

    # Add objective functions # Torques generated into articulations
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0, weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=1, weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=2, weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=3, weight=100)

    # EXPLANATION 1 on EXPLANATIONS_FILE
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=0, weight=0.0001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=1, weight=0.0001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=2, weight=0.0001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=3, weight=0.0001)

    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
                            target=vel_pushing, node=three_quarter_node_phase_1, phase=1, marker_index=4,
                            weight=1000)

    # objective_functions.add(custom_func_track_roty_principal_finger, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL,
    #                         phase=0, weight=10)
    # objective_functions.add(custom_func_track_roty_principal_finger, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL,
    #                         phase=1, weight=10)
    # objective_functions.add(custom_func_track_roty_principal_finger, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL,
    #                         phase=2, weight=10)
    # objective_functions.add(custom_func_track_roty_principal_finger, custom_type=ObjectiveFcn.Lagrange, node=Node.ALL,
    #                         phase=3, weight=10)

    objective_functions.add( # To minimize the difference between 0 and 1
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=1000,
        phase=1,
        quadratic=True,
    )
    objective_functions.add( # To minimize the difference between 1 and 2
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=1000,
        phase=2,
        quadratic=True,
    )
    objective_functions.add( # To minimize the difference between 2 and 3
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=1000,
        phase=3,
        quadratic=True,
    )


    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    # rajouter expend ?
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=3)

    # Constraints
    constraints = ConstraintList()

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.START, first_marker="finger_marker", second_marker="high_square", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="high_square", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="low_square", phase=1)
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=0, min_bound=0, phase=2)

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="high_square", phase=3)

    # constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
    #                 node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=0)
    # constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
    #                 node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=1)
    # constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
    #                 node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=2)
    # constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
    #                 node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=3)
    #
    # constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
    #                 node=Node.ALL, marker="finger_marker_5", min_bound=0, max_bound=10000, phase=0)
    # constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
    #                 node=Node.ALL, marker="finger_marker_5", min_bound=0, max_bound=10000, phase=1)
    # constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
    #                 node=Node.ALL, marker="finger_marker_5", min_bound=0, max_bound=10000, phase=2)
    # constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
    #                 node=Node.ALL, marker="finger_marker_5", min_bound=0, max_bound=10000, phase=3)
    #
    # constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
    #                 node=Node.ALL, min_bound=0, max_bound=10000, phase=0)
    # constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
    #                 node=Node.ALL, min_bound=0, max_bound=10000, phase=1)
    # constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
    #                 node=Node.ALL, min_bound=0, max_bound=10000, phase=2)
    # constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
    #                 node=Node.ALL, min_bound=0, max_bound=10000, phase=3)
    #
    # constraints.add(custom_func_track_finger_5_above_principal_finger,
    #                 node=Node.ALL, min_bound=0, max_bound=10000, phase=0)
    # constraints.add(custom_func_track_finger_5_above_principal_finger,
    #                 node=Node.ALL, min_bound=0, max_bound=10000, phase=1)
    # constraints.add(custom_func_track_finger_5_above_principal_finger,
    #                 node=Node.ALL, min_bound=0, max_bound=10000, phase=2)
    # constraints.add(custom_func_track_finger_5_above_principal_finger,
    #                 node=Node.ALL, min_bound=0, max_bound=10000, phase=3)

    # constraints.add(custom_func_track_finger_marker_key,
    #                 node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=2)

    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transition,
        ode_solver=ode_solver,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """

    ocp = prepare_ocp()
    ocp.add_plot_penalty(CostType.ALL)

    # # --- Solve the program --- # #

    solv = Solver.IPOPT(show_online_optim=True)
    solv.set_maximum_iterations(1000)
    solv.set_linear_solver("ma57")
    tic = time.time()
    sol = ocp.solve(solv)

    print('temps de resolution : ', time.time() - tic, 's')
    ocp.print(to_console=False, to_graph=False)

    # # --- Show important markers states --- # #

    # All state for each phase
    q0 = sol.states[0]["q"]
    q1 = sol.states[1]["q"]
    q2 = sol.states[2]["q"]
    q3 = sol.states[3]["q"]

    # States q for Finger_Marker_5 and Finger_marker
    q_finger_marker_5_idx_1 = []
    q_finger_marker_idx_5 = []
    phase_shape = []
    phase_time = []
    for k in [1, 5]:
        for i in range(4):
            # Number of nodes per phase : 0=151, 1=31, 2=46, 3=251 (bc COLLOCATION)
            for j in range(sol.states[i]["q"].shape[1]):
                q_all_markers = BiorbdInterface.mx_to_cx("markers", sol.ocp.nlp[i].model.markers, sol.states[i]["q"][:, j])  # q_markers = 3 * 10
                q_marker = q_all_markers["o0"][:, k]  # q_marker_1_one_node = 3 * 1
                if k == 1:
                    q_finger_marker_5_idx_1.append(q_marker)
                elif k == 5:
                    q_finger_marker_idx_5.append(q_marker)
            if k == 1:
                phase_time.append(ocp.nlp[i].tf)
                phase_shape.append(sol.states[i]["q"].shape[1])

    q_finger_marker_5_idx_1 = np.array(q_finger_marker_5_idx_1)
    q_finger_marker_5_idx_1 = q_finger_marker_5_idx_1.reshape((479, 3))

    q_finger_marker_idx_5 = np.array(q_finger_marker_idx_5)
    q_finger_marker_idx_5 = q_finger_marker_idx_5.reshape((479, 3))

    # Plot curves
    t = np.linspace(0, sum(phase_time), sum(phase_shape))
    figQ, axs = plt.subplots(2, 3)

    axs[0, 0].set_title("X\n", color='green')
    axs[0, 0].set(ylabel="Position (m)")
    axs[0, 0].plot(t, q_finger_marker_5_idx_1[:, 0], color='green')
    axs[0, 0].plot(t, q_finger_marker_idx_5[:, 0], color='r', linestyle='--',
                   label="C1 : Finger 5 at the top right of the principal finger\n"
                         "- in y : Finger 5 ⩽ Principal finger\n"
                         "- in z : Finger 5 ⩾ Principal finger")

    axs[0, 1].set_title("SMALL ONE Finger_Marker_5\nY", color='green')
    axs[0, 1].plot(t, q_finger_marker_5_idx_1[:, 1], color='green')
    axs[0, 1].plot(t, q_finger_marker_idx_5[:, 1], color='r', linestyle='--')  # "C1 : Finger 5 at the top right of the principal finger"

    axs[0, 2].set_title("Z\n", color='green')
    axs[0, 2].plot(t, q_finger_marker_5_idx_1[:, 2], color='green')
    axs[0, 2].plot(t, q_finger_marker_idx_5[:, 2], color='r', linestyle='--')  # "C1 : Finger 5 at the top right of the principal finger"
    axs[0, 2].axhline(y=0.07808863830566405-0.01, color='b', linestyle='--',
                      label="C2 : Finger 5 and principal finger above the bed key\n"
                            "- bed Key")

    axs[1, 0].set_title("X\n", color='red')
    axs[1, 0].set(ylabel='Position (m)')
    axs[1, 0].plot(t, q_finger_marker_idx_5[:, 0], color='red')

    axs[1, 1].set_title("PRINCIPAL Finger_Marker\nY", color='red')
    axs[1, 1].set(xlabel='Time (s) \n')
    axs[1, 1].plot(t, q_finger_marker_idx_5[:, 1], color='red')
    axs[1, 1].axhline(y=0, color='m', linestyle='--',
                      label="Obj : Principal Finger just in rotation around y \n"
                            "- no translation in y")

    axs[1, 2].set_title("Z\n", color='red')
    axs[1, 2].plot(t, q_finger_marker_idx_5[:, 2], color='red')
    axs[1, 2].axhline(y=0.07808863830566405-0.01, color='b', linestyle='--')  # "C2 : Finger 5 and principal finger above the bed key"

    figQ.suptitle('State translations q for important markers', fontsize=16)
    figQ.legend(loc="upper right", borderaxespad=0, prop={"size": 8}, title="Spacial constraints and objectives for Markers :")
    for i in range(0, 2):
        for j in range(0, 3):
            axs[i, j].axvline(x=0.30, color='gray', linestyle='--')
            axs[i, j].axvline(x=0.30+0.027, color='gray', linestyle='--')
            axs[i, j].axvline(x=0.30+0.027+0.058, color='gray', linestyle='--')
            axs[i, j].axvline(x=0.30+0.027+0.058+0.5 , color='gray', linestyle='--')

    # # --- Download datas --- #

    data = dict(
        states=sol.states, controls=sol.controls, parameters=sol.parameters,
        iterations=sol.iterations,
        cost=np.array(sol.cost)[0][0], detailed_cost=sol.detailed_cost,
        real_time_to_optimize=sol.real_time_to_optimize,
        param_scaling=[nlp.parameters.scaling for nlp in ocp.nlp]
    )

    with open("results_download/piano_results_4_phases_9_no_CandO", "wb") as file:
        pickle.dump(data, file)

    # # --- Show results --- # #

    sol.print_cost()
    sol.graphs(show_bounds=True)
    plt.show()

    # sol.animate(markers_size=0.0010, contacts_size=0.0010, show_floor=False,
    #             show_segments_center_of_mass=True, show_global_ref_frame=True,
    #             show_local_ref_frame=False, )


if __name__ == "__main__":
    main()



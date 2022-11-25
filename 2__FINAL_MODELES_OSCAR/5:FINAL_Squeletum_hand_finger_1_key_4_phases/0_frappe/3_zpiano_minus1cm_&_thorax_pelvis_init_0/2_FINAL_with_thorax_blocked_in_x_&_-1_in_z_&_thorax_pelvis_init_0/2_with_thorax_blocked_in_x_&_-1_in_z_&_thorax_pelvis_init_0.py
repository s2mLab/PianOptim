"""
 !! Les axes du modèle ne sont pas les mêmes que ceux généralement utilisés en biomécanique : x axe de flexion, y supination/pronation, z vertical
 ici on a : Y -» X , Z-» Y et X -» Z
 """
from casadi import MX, acos, vertcat, dot, pi
import time
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
import biorbd_casadi as biorbd
import pickle
from pickle import dump
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
#     markers_diff_key1 = if_else(
#         finger_marker[1] < 0.01,
#         finger_marker[2] - 0,
#         if_else(
#             finger_marker[1] < 0.033,  # condition
#             finger_marker[2] - key,  # True
#             finger_marker[2]-0,  # False
#         )
#     )
#     return markers_diff_key1


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
    finger_marker_idx = biorbd.marker_index(all_pn.nlp.model, marker)
    markers = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    markers_diff_key3 = finger_marker[2] - (0.07808863830566405-0.02)

    return markers_diff_key3


# def custom_func_track_roty_principal_finger(all_pn: PenaltyNodeList, ) -> MX:
#
#     model = all_pn.nlp.model
#     rotation_matrix_index = biorbd.segment_index(model, "2proxph_2mcp_flexion")
#     q = all_pn.nlp.states["q"].mx
#
#     rotation_matrix = all_pn.nlp.model.globalJCS(q, rotation_matrix_index).to_mx()
#
#     output = vertcat(rotation_matrix[1, 0], rotation_matrix[1, 2], rotation_matrix[0, 1], rotation_matrix[2, 1],
#                      rotation_matrix[1, 1] - MX(1))
#     rotation_matrix_output = BiorbdInterface.mx_to_cx("rot_mat", output, all_pn.nlp.states["q"])
#
#     return rotation_matrix_output


def custom_func_track_principal_finger_pi_in_two_global_axis(all_pn: PenaltyNodeList, segment: str) -> MX:
    model = all_pn.nlp.model
    rotation_matrix_index = biorbd.segment_index(model, segment)
    q = all_pn.nlp.states["q"].mx
    # global JCS gives the local matrix according to the global matrix
    principal_finger_axis = all_pn.nlp.model.globalJCS(q, rotation_matrix_index).to_mx()  # x finger = y global
    y = MX.zeros(4)
    y[:4] = np.array([0, 1, 0, 1])
    # @ x : pour avoir l'orientation du vecteur x du jcs local exprimé dans le global
    # @ produit matriciel
    principal_finger_y = principal_finger_axis @ y
    principal_finger_y = principal_finger_y[:3, :]

    global_y = MX.zeros(3)
    global_y[:3] = np.array([0, 1, 0])

    teta = acos(dot(principal_finger_y, global_y[:3]))
    output_casadi = BiorbdInterface.mx_to_cx("scal_prod", teta, all_pn.nlp.states["q"])

    return output_casadi


def prepare_ocp(
        biorbd_model_path: str = "/home/lim/Documents/Stage Mathilde/PianOptim/0:On_going/5:FINAL_Squeletum_hand_finger_2_keys/frappe_&_pressed/4_phases/Squeletum_hand_finger_3D_2_keys_octave_LA_frappe_10_ddl.bioMod",
        ode_solver: OdeSolver = OdeSolver.COLLOCATION(polynomial_degree=4),
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
    n_shooting = (30, 6, 9, 30)
    phase_time = (0.3, 0.027, 0.058, 0.3)
    tau_min, tau_max, tau_init = -200, 200, 0

    vel_push_array2 = [[-0.698417100906372, -0.474601301515033, -0.368024758139809, -0.357349785081633, -0.367995643393795, -0.277969583506421, 0]]

    pi_sur_2_phase_0 = np.full((1, n_shooting[0]+1), pi/2)
    pi_sur_2_phase_1 = np.full((1, n_shooting[1]+1), pi/2)
    pi_sur_2_phase_2 = np.full((1, n_shooting[2]+1), pi/2)
    pi_sur_2_phase_3 = np.full((1, n_shooting[3]+1), pi/2)

    # Add objective functions # Torques generated into articulations
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0, weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=1, weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=2, weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=3, weight=100)

    # EXPLANATION 1 on EXPLANATIONS_FILE
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=0, weight=0.0001)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=1, weight=0.0001)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=2, weight=0.0001)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=3, weight=0.0001)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=0.0001,
                            index=[0, 1, 2, 3, 4, 5, 6, 7])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=1, weight=0.0001,
                            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=2, weight=0.0001,
                            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=0.0001,
                            index=[0, 1, 2, 3, 4, 5, 6, 7])

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=100,
                            index=[8, 9], derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=100,
                            index=[8, 9], derivative=True)

    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
                            target=vel_push_array2, node=Node.ALL, phase=1, marker_index=4,
                            weight=10000)

    objective_functions.add(custom_func_track_principal_finger_pi_in_two_global_axis, custom_type=ObjectiveFcn.Lagrange,
                            node=Node.ALL, phase=0, weight=1000, quadratic=True, target=pi_sur_2_phase_0,
                            segment="2proxph_2mcp_flexion")
    objective_functions.add(custom_func_track_principal_finger_pi_in_two_global_axis, custom_type=ObjectiveFcn.Lagrange,
                            node=Node.ALL, phase=1, weight=100000, quadratic=True, target=pi_sur_2_phase_1,
                            segment="2proxph_2mcp_flexion")
    objective_functions.add(custom_func_track_principal_finger_pi_in_two_global_axis, custom_type=ObjectiveFcn.Lagrange,
                            node=Node.ALL, phase=2, weight=100000, quadratic=True, target=pi_sur_2_phase_2,
                            segment="2proxph_2mcp_flexion")
    objective_functions.add(custom_func_track_principal_finger_pi_in_two_global_axis, custom_type=ObjectiveFcn.Lagrange,
                            node=Node.ALL, phase=3, weight=1000, quadratic=True, target=pi_sur_2_phase_3,
                            segment="2proxph_2mcp_flexion")

    objective_functions.add(custom_func_track_principal_finger_pi_in_two_global_axis, custom_type=ObjectiveFcn.Lagrange,
                            node=Node.ALL, phase=0, weight=1000, quadratic=True, target=pi_sur_2_phase_0,
                            segment="secondmc")
    objective_functions.add(custom_func_track_principal_finger_pi_in_two_global_axis, custom_type=ObjectiveFcn.Lagrange,
                            node=Node.ALL, phase=1, weight=100000, quadratic=True, target=pi_sur_2_phase_1,
                            segment="secondmc")
    objective_functions.add(custom_func_track_principal_finger_pi_in_two_global_axis, custom_type=ObjectiveFcn.Lagrange,
                            node=Node.ALL, phase=2, weight=100000, quadratic=True, target=pi_sur_2_phase_2,
                            segment="secondmc")
    objective_functions.add(custom_func_track_principal_finger_pi_in_two_global_axis, custom_type=ObjectiveFcn.Lagrange,
                            node=Node.ALL, phase=3, weight=1000, quadratic=True, target=pi_sur_2_phase_3,
                            segment="secondmc")

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
    # dynamics = DynamicsList()
    # expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
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

    constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
                    node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=0)
    constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
                    node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=1)
    constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
                    node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=2)
    constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
                    node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=3)

    constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
                    node=Node.ALL, marker="finger_marker_5", min_bound=0, max_bound=10000, phase=0)
    constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
                    node=Node.ALL, marker="finger_marker_5", min_bound=0, max_bound=10000, phase=1)
    constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
                    node=Node.ALL, marker="finger_marker_5", min_bound=0, max_bound=10000, phase=2)
    constraints.add(custom_func_track_principal_finger_and_finger5_above_bed_key,
                    node=Node.ALL, marker="finger_marker_5", min_bound=0, max_bound=10000, phase=3)

    constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
                    node=Node.ALL, min_bound=0.00001, max_bound=10000, phase=0)
    constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
                    node=Node.ALL, min_bound=0.00001, max_bound=10000, phase=1)
    constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
                    node=Node.ALL, min_bound=0.00001, max_bound=10000, phase=2)
    constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
                    node=Node.ALL, min_bound=0.00001, max_bound=10000, phase=3)

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

    x_bounds[0][[0, 1, 2], 0] = 0
    x_bounds[3][[0, 1, 2], 2] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    for i in range(4):
        x_init[i][4, 0] = 0.0
        x_init[i][5, 0] = 0.67
        x_init[i][6, 0] = 1.11
        x_init[i][7, 0] = 1.48
        x_init[i][9, 0] = 0.17

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
    solv.set_maximum_iterations(100000)
    solv.set_linear_solver("ma57")
    tic = time.time()
    sol = ocp.solve(solv)

    # # --- Take important states for Finger_Marker_5 and Finger_marker --- # #

    q_finger_marker_5_idx_1 = []
    q_finger_marker_idx_4 = []
    phase_shape = []
    phase_time = []
    for k in [1, 4]:
        for i in range(4):
            # Number of nodes per phase : 0=151, 1=31, 2=46, 3=151 (251) (bc COLLOCATION)
            for j in range(sol.states[i]["q"].shape[1]):
                q_all_markers = BiorbdInterface.mx_to_cx("markers", sol.ocp.nlp[i].model.markers, sol.states[i]["q"][:, j])  # q_markers = 3 * 10
                q_marker = q_all_markers["o0"][:, k]  # q_marker_1_one_node = 3 * 1
                if k == 1:
                    q_finger_marker_5_idx_1.append(q_marker)
                elif k == 4:
                    q_finger_marker_idx_4.append(q_marker)
            if k == 1:
                phase_time.append(ocp.nlp[i].tf)
                phase_shape.append(sol.states[i]["q"].shape[1])

    q_finger_marker_5_idx_1 = np.array(q_finger_marker_5_idx_1)
    q_finger_marker_5_idx_1 = q_finger_marker_5_idx_1.reshape((379, 3))

    q_finger_marker_idx_4 = np.array(q_finger_marker_idx_4)
    q_finger_marker_idx_4 = q_finger_marker_idx_4.reshape((379, 3))

    # # --- Download datas --- #

    data = dict(
        states=sol.states, controls=sol.controls, parameters=sol.parameters,
        iterations=sol.iterations,
        cost=np.array(sol.cost)[0][0], detailed_cost=sol.detailed_cost,
        real_time_to_optimize=sol.real_time_to_optimize,
        param_scaling=[nlp.parameters.scaling for nlp in ocp.nlp],
        phase_time=phase_time, phase_shape=phase_shape,
        q_finger_marker_5_idx_1=q_finger_marker_5_idx_1,
        q_finger_marker_idx_4=q_finger_marker_idx_4,
    )

    with open(
            "/0:On_going/5:FINAL_Squeletum_hand_finger_2_keys/frappe_&_pressed/4_phases/0_frappe/results/3_zpiano_minus1cm_&_thorax_pelvis_init_0/2_with_thorax_blocked_in_x_&_-1_in_z_&_thorax_pelvis_init_0/1.pckl", "wb") as file:
        pickle.dump(data, file)

    print("results saved")
    print('temps de resolution : ', time.time() - tic, 's')
    ocp.print(to_console=False, to_graph=False)
    sol.graphs(show_bounds=True)
    ocp.print(to_console=False, to_graph=False)
    sol.print_cost()


if __name__ == "__main__":
    main()

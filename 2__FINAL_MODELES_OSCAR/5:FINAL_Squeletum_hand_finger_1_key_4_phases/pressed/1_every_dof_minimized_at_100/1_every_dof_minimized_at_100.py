"""
 !! Les axes du modèle ne sont pas les mêmes que ceux généralement utilisés en biomécanique : x axe de flexion, y supination/pronation, z vertical
 ici on a : Y -» X , Z-» Y et X -» Z
 """
from casadi import MX, acos, vertcat, dot, pi
import time
import numpy as np
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
)


def minimize_difference(all_pn: PenaltyNode):
    return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx


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


def custom_func_track_roty_principal_finger(all_pn: PenaltyNodeList, ) -> MX:

    model = all_pn.nlp.model
    rotation_matrix_index = biorbd.segment_index(model, "2proxph_2mcp_flexion")
    q = all_pn.nlp.states["q"].mx

    rotation_matrix = all_pn.nlp.model.globalJCS(q, rotation_matrix_index).to_mx()

    output = vertcat(rotation_matrix[1, 0], rotation_matrix[1, 2], rotation_matrix[0, 1], rotation_matrix[2, 1],
                     rotation_matrix[1, 1] - MX(1))
    rotation_matrix_output = BiorbdInterface.mx_to_cx("rot_mat", output, all_pn.nlp.states["q"])

    return rotation_matrix_output


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
        biorbd_model_path: str = "/home/lim/Documents/Stage Mathilde/PianOptim/0__On_going/Resultats_FINAL/pressed/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA_frappe_10_ddl.bioMod",
        ode_solver: OdeSolver = OdeSolver.COLLOCATION(polynomial_degree=4),
) -> OptimalControlProgram:

    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path))

    # Average of N frames by phase ; Average of phases time ; both measured with the motion capture datas.
    n_shooting = (30, 7, 7, 35)
    phase_time = (0.3, 0.044, 0.051, 0.35)
    tau_min, tau_max, tau_init = -200, 200, 0
    # Velocity profile found thanks to the motion capture datas.
    vel_push_array2 = [[0, -0.113772161006927, -0.180575996580578, -0.270097219830468,
                        -0.347421549388341, -0.290588704744975, -0.0996376128423782, 0]]

    pi_sur_2_phase_0 = np.full((1, n_shooting[0]+1), pi/2)
    pi_sur_2_phase_1 = np.full((1, n_shooting[1]+1), pi/2)
    pi_sur_2_phase_2 = np.full((1, n_shooting[2]+1), pi/2)
    pi_sur_2_phase_3 = np.full((1, n_shooting[3]+1), pi/2)

    # Objectives
    # Minimize Torques generated into articulations
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0, weight=100,
                            index=[0, 1, 2, 3, 4, 5])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=1, weight=100,
                            index=[0, 1, 2, 3, 4, 5])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=2, weight=100,
                            index=[0, 1, 2, 3, 4, 5])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=3, weight=100,
                            index=[0, 1, 2, 3, 4, 5])

    # Special articulations called individually in order to see, in the results, the individual objectives cost of each.
    for j in [6, 7, 8, 9]:
        for i in [0, 1, 2, 3]:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=i, weight=100,
                                    index=j)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=0.0001,
                            index=[0, 1, 2, 3, 4, 5, 6, 7])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=1, weight=0.0001,
                            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=2, weight=0.0001,
                            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=0.0001,
                            index=[0, 1, 2, 3, 4, 5, 6, 7])

    # To block ulna rotation before the key pressing.
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=100000,
                            index=[7])

    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
                            target=vel_push_array2, node=Node.ALL, phase=1, marker_index=4,
                            weight=10000)

    # To keep the hand/index perpendicular of the key piano all long the attack.
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

    # To avoid the apparition of "noise" caused by the objective function just before.
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=100,
                            index=[8, 9], derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=100,
                            index=[8, 9], derivative=True)

    # To minimize the difference between 0 and 1
    objective_functions.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=1000,
        phase=1,
        quadratic=True,
    )
    # To minimize the difference between 1 and 2
    objective_functions.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=1000,
        phase=2,
        quadratic=True,
    )
    # To minimize the difference between 2 and 3
    objective_functions.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=1000,
        phase=3,
        quadratic=True,
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=3)

    # Constraints
    constraints = ConstraintList()

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.ALL, first_marker="finger_marker", second_marker="high_square", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="low_square", phase=1)
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=0, min_bound=-5, max_bound=5, phase=2)
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=1, min_bound=-5, max_bound=5, phase=2)
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=2, min_bound=0, max_bound=30, phase=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="high_square", phase=3)

    # To keep the index and the small finger above the bed key.
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

    # To keep the small finger on the right of the principal finger.
    constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
                    node=Node.ALL, min_bound=0.00001, max_bound=10000, phase=0)
    constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
                    node=Node.ALL, min_bound=0.00001, max_bound=10000, phase=1)
    constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
                    node=Node.ALL, min_bound=0.00001, max_bound=10000, phase=2)
    constraints.add(custom_func_track_finger_5_on_the_right_of_principal_finger,
                    node=Node.ALL, min_bound=0.00001, max_bound=10000, phase=3)

    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    # EXPLANATION
    # ex: x_bounds[0][3, 0] = vel_pushing
    # [ phase 0 ]
    # [indice du ddl (0 et 1 position y z, 2 et 3 vitesse y z),
    # time (0 =» 1st point, 1 =» all middle points, 2 =» last point)]

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
        x_init[i][4, 0] = 0.08
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
    solv.set_maximum_iterations(1)
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
            # Number of nodes per phase : 0=151, 1=36, 2=36, 3=176 (399) (bc COLLOCATION)
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
    q_finger_marker_5_idx_1 = q_finger_marker_5_idx_1.reshape((399, 3))

    q_finger_marker_idx_4 = np.array(q_finger_marker_idx_4)
    q_finger_marker_idx_4 = q_finger_marker_idx_4.reshape((399, 3))

    # # --- Download datas on a .pckl file --- #

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
            "/0__On_going/Resultats_FINAL/pressed/3_FINAL_with_thorax_blocked_in_x_&_-1_in_z_&_thorax_pelvis_init_0/1_every_dof_100/test_with_max_bound_contact/asupprimer.pckl", "wb") as file:
        pickle.dump(data, file)

    # # --- Print results --- # #

    print("Tesults saved")
    print('Temps de resolution : ', time.time() - tic, 's')
    ocp.print(to_console=False, to_graph=False)
    sol.graphs(show_bounds=True)
    sol.print_cost()


if __name__ == "__main__":
    main()

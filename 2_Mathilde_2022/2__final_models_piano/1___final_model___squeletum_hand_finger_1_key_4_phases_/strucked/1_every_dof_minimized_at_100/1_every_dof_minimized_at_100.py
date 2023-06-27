"""
 !! Les axes du modèle ne sont pas les mêmes que ceux généralement utilisés en biomécanique : x axe de flexion, y supination/pronation, z vertical
 ici on a : Y -» X , Z-» Y et X -» Z
 """
from casadi import MX, acos, dot, pi
import time
import numpy as np
import biorbd_casadi as biorbd
import pickle

from bioptim import (
    BiorbdModel,
    PenaltyController,
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
    OdeSolver,
    Solver,
    MultinodeObjectiveList,
)

def minimize_difference(controllers: list[PenaltyController, PenaltyController]):
    pre, post = controllers
    return pre.controls.cx_end - post.controls.cx

def custom_func_track_finger_5_on_the_right_of_principal_finger(controller: PenaltyController) -> MX:
    finger_marker_idx = biorbd.marker_index(controller.model.model, "finger_marker")
    markers = controller.mx_to_cx("markers", controller.model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    finger_marker_5_idx = biorbd.marker_index(controller.model.model, "finger_marker_5")
    markers_5 = controller.mx_to_cx("markers_5", controller.model.markers, controller.states["q"])
    finger_marker_5 = markers_5[:, finger_marker_5_idx]

    markers_diff_key2 = finger_marker[1] - finger_marker_5[1]

    return markers_diff_key2

def custom_func_track_principal_finger_and_finger5_above_bed_key(controller: PenaltyController, marker_name: str) -> MX:
    biorbd_model = controller.model
    finger_marker_idx = biorbd.marker_index(biorbd_model.model, marker_name)
    markers = controller.mx_to_cx("markers", biorbd_model.markers, controller.states["q"])
    finger_marker = markers[:, finger_marker_idx]

    markers_diff_key3 = finger_marker[2] - (0.07808863830566405 - 0.02)

    return markers_diff_key3

def custom_func_track_principal_finger_pi_in_two_global_axis(controller: PenaltyController, segment: str) -> MX:
    rotation_matrix_index = biorbd.segment_index(controller.model.model, segment)
    q = controller.states["q"].mx
    # global JCS gives the local matrix according to the global matrix
    principal_finger_axis= controller.model.model.globalJCS(q, rotation_matrix_index).to_mx()  # x finger = y global
    y = MX.zeros(4)
    y[:4] = np.array([0, 1, 0, 1])
    # @ x : pour avoir l'orientation du vecteur x du jcs local exprimé dans le global
    # @ produit matriciel
    principal_finger_y = principal_finger_axis @ y
    principal_finger_y = principal_finger_y[:3, :]

    global_y = MX.zeros(3)
    global_y[:3] = np.array([0, 1, 0])

    teta = acos(dot(principal_finger_y, global_y[:3]))
    output_casadi = controller.mx_to_cx("scal_prod", teta, controller.states["q"])

    return output_casadi

    # principal_finger_axis = all_pn.nlp.model.globalJCS(q, rotation_matrix_index).to_mx()  # x finger = y global

def prepare_ocp(
    biorbd_model_path: str = "/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod",
    ode_solver: OdeSolver = OdeSolver.COLLOCATION(polynomial_degree=4),
    # assume_phase_dynamics: bool = True,
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

    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    # Average of N frames by phase ; Average of phases time ; both measured with the motion capture datas.
    n_shooting = (30, 6, 9, 30)
    phase_time1 = (0.3, 0.027, 0.058, 0.3)
    tau_min, tau_max, tau_init = -200, 200, 0
    # Velocity profile found thanks to the motion capture datas.
    vel_push_array2 = [
        [
            -0.698417100906372,
            -0.474601301515033,
            -0.368024758139809,
            -0.357349785081633,
            -0.367995643393795,
            -0.277969583506421,
            0,
        ]
    ]

    pi_sur_2_phase_0 = np.full((1, n_shooting[0] + 1), pi / 2)
    pi_sur_2_phase_1 = np.full((1, n_shooting[1] + 1), pi / 2)
    pi_sur_2_phase_2 = np.full((1, n_shooting[2] + 1), pi / 2)
    pi_sur_2_phase_3 = np.full((1, n_shooting[3] + 1), pi / 2)

    # Objectives
    # Minimize Torques generated into articulations
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0, weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=1, weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=2, weight=100)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=3, weight=100)

    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", phase=0, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=1, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=2, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=0.0001, index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
        target=vel_push_array2,
        node=Node.ALL,
        phase=1,
        marker_index=4,
        weight=10000,
    )

    # To keep the hand/index perpendicular of the key piano all long the attack.
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=0,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_0,
        segment="2proxph_2mcp_flexion",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=1,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_1,
        segment="2proxph_2mcp_flexion",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=2,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_2,
        segment="2proxph_2mcp_flexion",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=3,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_3,
        segment="2proxph_2mcp_flexion",
    )

    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=0,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_0,
        segment="secondmc",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=1,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_1,
        segment="secondmc",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=2,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_2,
        segment="secondmc",
    )
    objective_functions.add(
        custom_func_track_principal_finger_pi_in_two_global_axis,
        custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL,
        phase=3,
        weight=100000,
        quadratic=True,
        target=pi_sur_2_phase_3,
        segment="secondmc",
    )

    # To avoid the apparition of "noise" caused by the objective function just before.
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=0, weight=100, index=[8, 9], derivative=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=100, index=[8, 9], derivative=True
    )

    multinode_objectives = MultinodeObjectiveList()

    # To minimize the difference between 0 and 1
    multinode_objectives.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        nodes_phase=(0, 1),
        nodes=(Node.END, Node.START),
        quadratic=True,
    )
    # # To minimize the difference between 0 and 1
    multinode_objectives.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        nodes_phase=(1, 2),
        nodes=(Node.END, Node.START),
        quadratic=True,
    )
    # # To minimize the difference between 2 and 3
    multinode_objectives.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        nodes_phase=(2, 3),
        nodes=(Node.END, Node.START),
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

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.START,
        first_marker="finger_marker",
        second_marker="high_square",
        phase=0,
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="high_square",
        phase=0,
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="low_square",
        phase=1,
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES, node=Node.ALL, contact_index=0, min_bound=-5, max_bound=5, phase=2
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES, node=Node.ALL, contact_index=1, min_bound=-5, max_bound=5, phase=2
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES, node=Node.ALL, contact_index=2, min_bound=0, max_bound=30, phase=2
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="high_square",
        phase=3,
    )

    # To keep the index and the small finger above the bed key.
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker_name="finger_marker",
        min_bound=0,
        max_bound=10000,
        phase=0,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker_name="finger_marker",
        min_bound=0,
        max_bound=10000,
        phase=1,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker_name="finger_marker",
        min_bound=0,
        max_bound=10000,
        phase=2,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker_name="finger_marker",
        min_bound=0,
        max_bound=10000,
        phase=3,
    )

    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker_name="finger_marker_5",
        min_bound=0,
        max_bound=10000,
        phase=0,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker_name="finger_marker_5",
        min_bound=0,
        max_bound=10000,
        phase=1,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker_name="finger_marker_5",
        min_bound=0,
        max_bound=10000,
        phase=2,
    )
    constraints.add(
        custom_func_track_principal_finger_and_finger5_above_bed_key,
        node=Node.ALL,
        marker_name="finger_marker_5",
        min_bound=0,
        max_bound=10000,
        phase=3,
    )

    # To keep the small finger on the right of the principal finger.
    constraints.add(
        custom_func_track_finger_5_on_the_right_of_principal_finger,
        node=Node.ALL,
        min_bound=0.00001,
        max_bound=10000,
        phase=0,
    )
    constraints.add(
        custom_func_track_finger_5_on_the_right_of_principal_finger,
        node=Node.ALL,
        min_bound=0.00001,
        max_bound=10000,
        phase=1,
    )
    constraints.add(
        custom_func_track_finger_5_on_the_right_of_principal_finger,
        node=Node.ALL,
        min_bound=0.00001,
        max_bound=10000,
        phase=2,
    )
    constraints.add(
        custom_func_track_finger_5_on_the_right_of_principal_finger,
        node=Node.ALL,
        min_bound=0.00001,
        max_bound=10000,
        phase=3,
    )

    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)

    # EXPLANATION
    # ex: x_bounds[0][3, 0] = vel_pushing
    # [ phase 0 ]
    # [indice du ddl (0 et 1 position y z, 2 et 3 vitesse y z),
    # time (0 =» 1st point, 1 =» all middle points, 2 =» last point)]

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))

    x_bounds[0][[0, 1, 2], 0] = 0
    x_bounds[3][[0, 1, 2], 2] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nb_q + biorbd_model[0].nb_qdot))
    x_init.add([0] * (biorbd_model[0].nb_q + biorbd_model[0].nb_qdot))
    x_init.add([0] * (biorbd_model[0].nb_q + biorbd_model[0].nb_qdot))
    x_init.add([0] * (biorbd_model[0].nb_q + biorbd_model[0].nb_qdot))

    for i in range(4):
        x_init[i][4, 0] = 0.08
        x_init[i][5, 0] = 0.67
        x_init[i][6, 0] = 1.11
        x_init[i][7, 0] = 1.48
        x_init[i][9, 0] = 0.17

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nb_tau, [tau_max] * biorbd_model[0].nb_tau)
    u_bounds.add([tau_min] * biorbd_model[0].nb_tau, [tau_max] * biorbd_model[0].nb_tau)
    u_bounds.add([tau_min] * biorbd_model[0].nb_tau, [tau_max] * biorbd_model[0].nb_tau)
    u_bounds.add([tau_min] * biorbd_model[0].nb_tau, [tau_max] * biorbd_model[0].nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nb_tau)
    u_init.add([tau_init] * biorbd_model[0].nb_tau)
    u_init.add([tau_init] * biorbd_model[0].nb_tau)
    u_init.add([tau_init] * biorbd_model[0].nb_tau)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        phase_time1,
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
    solv.set_maximum_iterations(10000)
    solv.set_linear_solver("ma57")
    tic = time.time()
    sol = ocp.solve(solv)
    #
    # # # # --- Take important states for Finger_Marker_5 and Finger_marker --- # #

    # # # --- Download datas on a .pckl file --- #

    data = dict(
        states=sol.states,
        states_no_intermediate=sol.states_no_intermediate,
        controls=sol.controls,
        parameters=sol.parameters,
        iterations=sol.iterations,
        cost=np.array(sol.cost)[0][0],
        detailed_cost=sol.detailed_cost,
        real_time_to_optimize=sol.real_time_to_optimize,
        param_scaling=[nlp.parameters.scaling for nlp in ocp.nlp],
        phase_time=sol.phase_time,
        Time=sol.time,

    )

    with open(
            "/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/strucked/Results/alldofs_struckTouch_1.pckl", "wb") as file:
        pickle.dump(data, file)

    # # --- Print results --- # #

    print("Results saved")
    print("Temps de resolution : ", time.time() - tic, "s")
    ocp.print(to_console=False, to_graph=False)
    # sol.graphs(show_bounds=True)
    sol.print_cost()
    sol.animate(show_floor=False, show_global_center_of_mass=False, show_segments_center_of_mass=False, show_global_ref_frame=False, show_local_ref_frame=False, show_markers=False, n_frames=500)


if __name__ == "__main__":
    main()

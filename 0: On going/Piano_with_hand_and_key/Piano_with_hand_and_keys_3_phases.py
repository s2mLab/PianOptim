"""
 !! Les axes du modèle ne sont pas les mêmes que ceux généralement utilisés en biomécanique : x axe de flexion, y supination/pronation, z vertical
 ici on a : Y -» X , Z-» Y et X -» Z
 """
from casadi import MX, sqrt, if_else, sin
import time
import biorbd_casadi as biorbd
from bioptim import (
    PenaltyNode,
    Axis,
    ObjectiveList,
    DynamicsList,
    ConstraintFcn,
    BoundsList,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    QAndQDotBounds,
    OdeSolver,
    Solver,
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

# def custom_func_track_markers():
#   return markers_diff


def custom_func_track_markers(all_pn: PenaltyNodeList, marker: str) -> MX:
    finger_marker_idx = biorbd.marker_index(all_pn.nlp.model, marker)
    markers = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
    finger_marker = markers[:, finger_marker_idx]
    key = ((0.005*sin(137*(finger_marker[1]+0.0129))) / (sqrt(0.001**2 + sin(137*(finger_marker[1] + 0.0129))**2))-0.005)

    # if_else( condition, si c'est vrai fait ca',  sinon fait ca)
    markers_diff_key = if_else(
        finger_marker[1] < 0.01,
        finger_marker[2] - 0,
        if_else(
            finger_marker[1] < 0.033,  # condition
            finger_marker[2] - key,  # True
            finger_marker[2]-0,  # False
        )
    )
    return markers_diff_key


# def custom_func_track_markers2(all_pn: PenaltyNodeList, marker: str) -> MX:
#     finger_marker_idx = biorbd.marker_index(all_pn.nlp.model, marker)
#     markers = BiorbdInterface.mx_to_cx("markers", all_pn.nlp.model.markers, all_pn.nlp.states["q"])
#     finger_marker = markers[:, finger_marker_idx]
#
#     # if_else( condition, si c'est vrai fait ca',  sinon fait ca)
#     markers_diff_key2 = if_else(
#         finger_marker[0] < 0,  # condition
#         finger_marker[0] - 0,  # True
#         finger_marker[0] - 0,  # False
#         )
#     return markers_diff_key2


def prepare_ocp(
        biorbd_model_path: str = "/home/mickaelbegon/Documents/Stage_Mathilde/programation/PianOptim/0: On going/Piano_with_hand_and_key/Piano_with_hand_and_keys.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK4(),
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
                    biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Average of N frames by phase and the phases time, both measured with the motion capture datas.
    # Name of the datas file : MotionCaptureDatas_Frames.xlsx
    n_shooting = (7*2, 7*2, 30*2, 7*2, 7*2)
    final_time = (0.044*2, 0.051*2, 0.2*2, 0.044*2, 0.051*2)
    tau_min, tau_max, tau_init = -100, 100, 0
    vel_pushing = 0.00372

    # Add objective functions # Torques generated into articulations
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=4)


    # Objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q",weight=100)

    # objective_functions.add( # To minimize the difference between 0 and 1
    #     minimize_difference,
    #     custom_type=ObjectiveFcn.Mayer,
    #     node=Node.TRANSITION,
    #     weight=1000,
    #     phase=1,
    #     quadratic=True,
    # )
    # objective_functions.add( # To minimize the difference between 1 and 2
    #     minimize_difference,
    #     custom_type=ObjectiveFcn.Mayer,
    #     node=Node.TRANSITION,
    #     weight=1000,
    #     phase=2,
    #     quadratic=True,
    # )
    # objective_functions.add( # To minimize the difference between 2 and 3
    #     minimize_difference,
    #     custom_type=ObjectiveFcn.Mayer,
    #     node=Node.TRANSITION,
    #     weight=1000,
    #     phase=3,
    #     quadratic=True,
    # )
    # objective_functions.add( # To minimize the difference between 3 and 4
    #     minimize_difference,
    #     custom_type=ObjectiveFcn.Mayer,
    #     node=Node.TRANSITION,
    #     weight=1000,
    #     phase=4,
    #     quadratic=True,
    # )
    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    # rajouter expend ?
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=4)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.START, first_marker="finger_marker", second_marker="high_square", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="low_square", phase=0)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY,
                    target=0, node=Node.START, phase=0, marker_index=4)
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=0, min_bound=0, phase=1)  # contact index : axe du contact

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="high_square2", phase=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="low_square2", phase=3)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY,
                    target=0, node=Node.START, phase=3, marker_index=4)
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=0, min_bound=0, phase=4)
    constraints.add(custom_func_track_markers,
                    node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=2)

    # constraints.add(custom_func_track_markers2,
    #                 node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=0, phase=1)
    # constraints.add(custom_func_track_markers2,
    #                 node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=0, phase=4)


    # Path constraint # x_bounds = limit conditions
    # [ phase 0 ] [indice du ddl (0 et 1 position y z, 2 et 3 vitesse y z), time]
    # (0 =» 1st point, 1 =» all middle points, 2 =» last point)
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds[0][5, 0] = vel_pushing
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds[2][4, 2] = 0
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds[3][5, 0] = vel_pushing
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
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
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
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

    # --- Solve the program --- #
    solv = Solver.IPOPT(show_online_optim=True)
    solv.set_linear_solver("ma57")
    tic = time.time()
    sol = ocp.solve(solv)

    print('temps de resolution : ', time.time() - tic)
    ocp.print(to_console=False, to_graph=False)


    # # --- Show results --- #
    sol.animate(markers_size=0.0010, contacts_size=0.0010, show_floor=False,
                show_segments_center_of_mass=True, show_global_ref_frame=True,
                show_local_ref_frame=False,)
    sol.print_cost()
    sol.graphs(show_bounds=True)


    # data = dict(
    #     states=sol.states, controls=sol.controls, parameters=sol.parameters,
    #     iterations=sol.iterations,
    #     cost=np.array(sol.cost)[0][0], detailed_cost=sol.detailed_cost,
    #     real_time_to_optimize=sol.real_time_to_optimize,
    #     param_scaling=[nlp.parameters.scaling for nlp in ocp.nlp]
    # )
    #
    # with open("Piano_results_3_phases_without_pelvis_rotZ_and_thorax.pckl", "wb") as file:
    #     pickle.dump(data, file)


if __name__ == "__main__":
    main()
from casadi import MX, sqrt, if_else, sin
import biorbd_casadi as biorbd
import numpy as np
from math import ceil
from bioptim import (
    ObjectiveList,
    ObjectiveFcn,
    PhaseTransitionFcn,
    PenaltyNodeList,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Node,
    Solver,
    CostType,
    PhaseTransitionList,
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    PenaltyNodeList,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    OdeSolver,
    BiorbdInterface,
    Solver,
    Axis

)


# PAR DEFAULT : M et S
def custom_func_track_finger_marker_key(all_pn: PenaltyNodeList, marker: str) -> MX:
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
            finger_marker[2] - 0,  # False
        )
    )
    return markers_diff_key


def prepare_ocp(biorbd_model_path: str = "/home/lim/Documents/Stage Mathilde/PianOptim/2:FINAL_MODELES/2:FINAL_Finger_2_keys_simulation/FINAL_Finger_2_keys_simulation.bioMod",
                ode_solver: OdeSolver = OdeSolver.COLLOCATION()
) -> OptimalControlProgram:
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Average of N frames by phase and the phases time, both measured with the motion capture datas.
    # Name of the datas file : MotionCaptureDatas_Frames.xlsx
    n_shooting = (7*2, 7*2, 20*2, 7*2, 7*2)
    phase_time = (0.044, 0.051, 0.5, 0.044, 0.051)
    tau_min, tau_max, tau_init = -200, 200, 0
    vel_pushing = 0.372

    # Find the number of the node at 75 % of the phase 0 and 3 in order to apply the vel_pushing at this node
    three_quarter_node_phase_0 = ceil(0.75 * n_shooting[0])
    three_quarter_node_phase_3 = ceil(0.75 * n_shooting[3])

    # Multiples vel_pushing to apply this velocity on multiples nodes.
    # 14 : -1 because Node.INTERMEDIATES doesn't count the last node, and -1 bc the first point can't have a velocity
    vel_push_array = np.zeros((1, 12))
    vel_push_array[0, :] = vel_pushing

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0, weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=1, weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=2, weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=3, weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=4, weight=1)

    # EXPLANATION 1 on EXPLANATIONS_FILE
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=0, weight=0.0001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=1, weight=0.0001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=2, weight=0.0001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=3, weight=0.0001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", index=1, phase=4, weight=0.0001)

    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
                            target=[0, 0, 0], node=Node.START, phase=0, marker_index=0,
                            weight=1000)
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
                            target=[0, 0, 0], node=Node.START, phase=3, marker_index=0,
                            weight=1000)

    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
                            target=vel_pushing, node=three_quarter_node_phase_0, phase=0, marker_index=0
                            , weight=1000)
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
                            target=vel_pushing, node=three_quarter_node_phase_3, phase=3, marker_index=0
                            , weight=1000)
    # Dynamics
    dynamics = DynamicsList()
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
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=0, min_bound=0, phase=1)  # contact index : axe du contact

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="high_square2", phase=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="low_square2", phase=3)
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=0, min_bound=0, phase=4)  # contact index : axe du contact

    constraints.add(custom_func_track_finger_marker_key,
                    node=Node.ALL, marker="finger_marker", min_bound=0, max_bound=10000, phase=2)

    phase_transition = PhaseTransitionList()
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)
    phase_transition.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
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

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True)
    solver.set_linear_solver("ma57")
    sol = ocp.solve(solver)

    # --- Show results --- #
    # show_segments_center_of_mass : origin du marker
    sol.animate(markers_size=0.0010, contacts_size=0.0010, show_floor=False,
                show_segments_center_of_mass=True, show_global_ref_frame=True,
                show_local_ref_frame=False),
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()

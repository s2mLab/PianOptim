
import math
from casadi import MX, sqrt, if_else, sin
from math import ceil
import biorbd_casadi as biorbd
import numpy as np
from bioptim import (
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

def prepare_ocp(
        biorbd_model_path: str = "/home/lim/Documents/Stage Mathilde/PianOptim/0:On going/5:FINAL_Finger_hand_2_keys_pressé_frappé/Frappe/FINAL_Frappe_Finger_hand_2_keys_VELOCITY_PROFIL/FINAL_Frappe_Finger_hand_2_keys_VELOCITY_PROFIL.bioMod",
        ode_solver: OdeSolver = OdeSolver.COLLOCATION()
) -> OptimalControlProgram:
    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path),
                    )

    # Average of N frames by phase and the phases time, both measured with the motion capture datas.
    # Name of the datas file : MotionCaptureDatas_Frames.xlsx
    # normally 45 phases for phase 0
    n_shooting = (49, 9, 100, 49, 9)
    phase_time = (0.325, 0.058, 0.5, 0.325, 0.058)
    tau_min, tau_max, tau_init = -200, 200, 0

    vel_push_array = [[-0.05268651580810547, -0.05882426764040576, -0.06465352786317163, -0.07001178476761797,
                      -0.07561809259531448, -0.08157519733662508, -0.08774206714240873, -0.09290024675641742,
                      -0.09725865718296595, -0.09996036249277542, -0.10290956660679407, -0.10682649339948382,
                      -0.1098874180073641, -0.11408276086924026, -0.1165207312447684, -0.11766088929468269,
                      -0.12080523012122332, -0.122764197524713, -0.12206943605384048, -0.12212927884471661,
                      -0.1218079194049446, -0.11944992828369139, -0.11881362697056362, -0.118992618638642,
                      -0.11928289779351682, -0.1192552624138034, -0.12023685720015547, -0.12216469433842873,
                      -0.12442561667306083, -0.12763144076600364, -0.13270834957823462, -0.140096222235232,
                      -0.1462007521025988, -0.15073194916394294, -0.15755177540681797, -0.16379577091761996,
                      -0.16884862751863443, -0.1743981194009586, -0.1800907745361328, -0.18716748280427896,
                      -0.19910029477489235, -0.22101017200703532, -0.2531227471487864, -0.2905638077405034,
                      -0.31946766039789946, -0.30941753512012707, -0.22911613059530464, -0.09612818527221667,
                      0.03924329742120289, 0.14856787490844727]]


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
                            target=vel_push_array, node=Node.ALL, phase=0, marker_index=4,
                            weight=1000)
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS_VELOCITY,
                            target=vel_push_array, node=Node.ALL, phase=3, marker_index=4,
                            weight=1000)
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
                    node=Node.END, first_marker="finger_marker", second_marker="low_square", phase=0)
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=0, min_bound=0, phase=1)

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END, first_marker="finger_marker", second_marker="low_square2", phase=3)
    constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES,
                    node=Node.ALL, contact_index=0, min_bound=0, phase=4)

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
    sol.animate(markers_size=0.0010, contacts_size=0.0010, show_floor=False,
                show_segments_center_of_mass=True, show_global_ref_frame=True,
                show_local_ref_frame=False,),
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()

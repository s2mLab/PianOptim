"""
This trivial spring example targets to have the highest upward velocity. It is however only able to load a spring by
pulling downward and afterward to let it go so it gains velocity. It is designed to show how one can use the external
forces to interact with the body.
"""

from casadi import MX, vertcat
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    ConfigureProblem,
    DynamicsFunctions,
    NonLinearProgram,
    DynamicsEvaluation,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    QAndQDotBounds,
    Node,
    PenaltyNode,
    InitialGuessList,
    Solver,
)


def custom_dynamic(states: MX, controls: MX, parameters: MX, nlp: NonLinearProgram) -> DynamicsEvaluation:
    """
    The dynamics of the system using an external force (see custom_dynamics for more explanation)

    Parameters
    ----------
    states: MX
        The current states of the system
    controls: MX
        The current controls of the system
    parameters: MX
        The current parameters of the system
    nlp: NonLinearProgram
        A reference to the phase of the ocp

    Returns
    -------
    The state derivative
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    force_vector = MX.zeros(6)
    force_vector[5] = 1000 * q[0] ** 2

    f_ext = biorbd.VecBiorbdSpatialVector()
    f_ext.append(biorbd.SpatialVector(force_vector))
    qddot = nlp.model.ForwardDynamics(q, qdot, tau, f_ext).to_mx()

    return DynamicsEvaluation(dxdt=vertcat(qdot, qddot), defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    The configuration of the dynamics (see custom_dynamics for more explanation)

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase of the ocp
    """
    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic)


def minimize_difference(all_pn: PenaltyNode):
    return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx


def prepare_ocp(biorbd_model_path: str = "Example_Simulation_1D.bioMod",
                biorbd_model_path_with_contact: str = "Example_Simulation_1D_with_impact.bioMod"):

    biorbd_model = (biorbd.Model(biorbd_model_path),
                    biorbd.Model(biorbd_model_path_with_contact),
                    biorbd.Model(biorbd_model_path)
                    )
    # arbitrary choices
    n_shooting = (20, 5, 20)

    # phase_appui = 0.16
    phase_time = (0.08, 0.02, 0.08)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", phase=0, weight=-1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", phase=1, weight=-1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", phase=2, weight=-1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=1)
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, phase=2)

    # Constraints
    min_bound = 50,
    max_bound = np.inf,

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=1
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=2,
        phase=1
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds = QAndQDotBounds(biorbd.Model(biorbd_model_path))
    x_bounds[:, 0] = [0] * biorbd_model[0].nbQ() + [0] * biorbd_model[0].nbQdot()
    x_bounds.min[:, 1] = [-0.1] * biorbd_model[0].nbQ() + [-1] * biorbd_model[0].nbQdot()
    x_bounds.max[:, 1] = [1] * biorbd_model[0].nbQ() + [1] * biorbd_model[0].nbQdot()
    x_bounds.min[:, 2] = [-0.1] * biorbd_model[0].nbQ() + [-1] * biorbd_model[0].nbQdot()
    x_bounds.max[:, 2] = [1] * biorbd_model[0].nbQ() + [1] * biorbd_model[0].nbQdot()

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([-100] * biorbd_model[0].nbGeneralizedTorque(), [0] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([-100] * biorbd_model[0].nbGeneralizedTorque(), [0] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([-100] * biorbd_model[0].nbGeneralizedTorque(), [0] * biorbd_model[0].nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([0] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([0] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([0] * biorbd_model[0].nbGeneralizedTorque())

    return OptimalControlProgram(
        biorbd_model=biorbd_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        phase_time=phase_time,
        u_bounds=u_bounds,
        objective_functions=objective_functions
    )


def main():
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()

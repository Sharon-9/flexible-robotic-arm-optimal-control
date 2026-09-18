#
# Direction and rollout utilities
#

import numpy as np

from task0_discretized_dynamics import dynamics, ns, ni


def compute_direction_trajectory(AA, BB, KK, sigma, cc_dyn, TT):
    """
    Compute the local optimal direction (Newton's direction produced by affine LQR problem)
    """
    dx = np.zeros((ns, TT))
    du = np.zeros((ni, TT - 1))

    for t in range(TT - 1):
        du[:, t] = KK[:, :, t] @ dx[:, t] + sigma[:, t]
        dx[:, t + 1] = AA[:, :, t] @ dx[:, t] + BB[:, :, t] @ du[:, t] + cc_dyn[:, t]

    return dx, du


def closed_loop_rollout(xx, uu, KK, sigma, gamma, TT): # gamma chosen by Armijo
    xx_new = np.zeros_like(xx)
    uu_new = np.zeros_like(uu)

    xx_new[:, 0] = xx[:, 0]

    for t in range(TT - 1):
        uu_new[:, t] = (uu[:, t] + KK[:, :, t] @ (xx_new[:, t] - xx[:, t]) + gamma * sigma[:, t] )

        xx_new[:, t + 1] = dynamics(xx_new[:, t], uu_new[:, t])

    uu_new[:, TT - 1] = uu_new[:, TT - 2] # inputs are defined up to TT-2 => fill in the last column to ease up plotting

    return xx_new, uu_new


def simulate_lqr_tracking(xx_gen, uu_gen, KK, dynamics, x0): # x0 = perturbed initial condition
    """
    Nonlinear simulation with LQR tracking:

        u_t = u_gen_t + K_t (x_t - x_gen_t)
    """
    # Obtain dimensions directly from generated trajectories
    ns, TT = xx_gen.shape
    ni = uu_gen.shape[0]

    xx = np.zeros((ns, TT))
    uu = np.zeros((ni, TT))

    xx[:, 0] = x0 

    for t in range(TT - 1):
        err = xx[:, t] - xx_gen[:, t]
        uu[:, t] = uu_gen[:, t] + KK[:, :, t] @ err
        xx[:, t + 1] = dynamics(xx[:, t], uu[:, t])

    uu[:, -1] = uu[:, -2] # copy last input to have same temporal length as states in plots

    return xx, uu


def simulate_mpc_tracking(xx_gen, uu_gen, AA, BB, QQ, RR, QQf, Np, x0, TT, umin, umax):
    """
    Nonlinear simulation with constrained linear MPC tracking.

    At each time instant:
      1. solve a constrained linear MPC problem in error coordinates;
      2. apply only the first input to the nonlinear system.
    """

    from mpc_solver import solver_linear_mpc_tracking

    xx = np.zeros((ns, TT))
    uu = np.zeros((ni, TT))

    xx[:, 0] = x0

    for tt in range(TT - 1):

        u_mpc, _, _ = solver_linear_mpc_tracking( AA, BB, QQ, RR, QQf, xx[:, tt], xx_gen, uu_gen, tt, umax=umax, umin=umin, T_pred=Np)

        if u_mpc is None:
            print(f"WARNING: MPC failed at time {tt}. Using nominal input.")
            uu[:, tt] = uu_gen[:, tt]
        else:
            uu[:, tt] = u_mpc

        xx[:, tt + 1] = dynamics(xx[:, tt], uu[:, tt])

    uu[:, TT - 1] = uu[:, TT - 2]

    return xx, uu
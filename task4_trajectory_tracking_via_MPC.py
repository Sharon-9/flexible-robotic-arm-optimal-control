#
# Task 4 - Trajectory tracking via MPC
#

import numpy as np
import matplotlib.pyplot as plt
import signal

from task0_discretized_dynamics import ns, dt
from task2_trajectory_generation import main as task2_main, TT, tt_hor
from rollout import simulate_mpc_tracking
from linearization import linearize_trajectory, dynamics_jacobian
from control import dare

signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({"font.size": 22})


Qmpc = np.diag([1, 1, 0.7, 0.7])
Rmpc = np.diag([0.1])

Np = 5 # predictive horizon


def main():

    xx_gen, uu_gen, _, _, _, _, _ = task2_main()

    Aeq, Beq = dynamics_jacobian(xx_gen[:, -1], uu_gen[:, -1])
    P_dare, _, _ = dare(Aeq, Beq, Qmpc, Rmpc)

    QmpcT = np.array(P_dare)
    QmpcT = 0.5 * (QmpcT + QmpcT.T)

    AA, BB = linearize_trajectory(xx_gen, uu_gen, TT)

    # Input constraints for constrained MPC
    u_margin = 0.01
    umax = float(np.max(np.abs(uu_gen[0, :])) + u_margin)
    umin = -umax

    print("MPC input constraints:")
    print("umin =", umin)
    print("umax =", umax)

    perturbations = [
        ("theta1 2 deg", np.array([np.deg2rad(2.0), 0.0, 0.0, 0.0])),
        ("theta2 2 deg", np.array([0.0, np.deg2rad(2.0), 0.0, 0.0])),
        ("theta1/theta2 2 deg", np.array([np.deg2rad(2.0), -np.deg2rad(2.0), 0.0, 0.0])),
        ("theta1 5 deg", np.array([np.deg2rad(5.0), 0.0, 0.0, 0.0])),
        ("theta1 10 deg", np.array([np.deg2rad(10.0), 0.0, 0.0, 0.0])),
    ]

    tracking_trajs = []

    for label, pp in perturbations:
        x0_mpc = xx_gen[:, 0] + pp # start from a perturbed state

        xx_mpc, uu_mpc = simulate_mpc_tracking(
            xx_gen, uu_gen, AA, BB, Qmpc, Rmpc, QmpcT, Np, x0_mpc, TT, umin, umax
        )

        tracking_trajs.append((label, pp, xx_mpc, uu_mpc))

    fig, axs = plt.subplots(ns + 1, 1, sharex=True, figsize=(11, 12))

    labels_x = [
        r'$\theta_1$',
        r'$\theta_2$',
        r'$\dot{\theta}_1$',
        r'$\dot{\theta}_2$',
    ]

    for i in range(ns):
        axs[i].plot(tt_hor, xx_gen[i, :], 'k--', linewidth=2, label='generated')

        for label, _, xx_i, _ in tracking_trajs:
            axs[i].plot(tt_hor, xx_i[i, :], linewidth=1.5, label=label)

        axs[i].set_ylabel(labels_x[i])
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc='best', fontsize=10)

    axs[ns].plot(tt_hor, uu_gen[0, :], 'k--', linewidth=2, label='generated')
    axs[ns].axhline(umax, linestyle='--', linewidth=2, label=r'$u_{\max}$')
    axs[ns].axhline(umin, linestyle='--', linewidth=2, label=r'$u_{\min}$')
    
    for label, _, _, uu_i in tracking_trajs:
        axs[ns].plot(tt_hor, uu_i[0, :], linewidth=1.5, label=label)

    axs[ns].set_ylabel(r'$u$')
    axs[ns].set_xlabel('time [s]')
    axs[ns].grid(True, alpha=0.3)
    axs[ns].legend(loc='best', fontsize=10)

    fig.suptitle('Task 4 - MPC tracking trajectories')
    plt.tight_layout()
    plt.show(block = False)
    

    plt.figure()

    for label, _, xx_i, _ in tracking_trajs:
        err = xx_i - xx_gen
        err_norm = np.linalg.norm(err, axis=0)

        plt.semilogy(tt_hor, err_norm, linewidth=2, label=label)

    plt.xlabel('time [s]')
    plt.ylabel(r'$\|x_t - x^{gen}_t\|$')
    plt.title('Task 4 - MPC tracking error norm')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    

    return tracking_trajs, xx_gen, uu_gen


if __name__ == "__main__":
    tracking_trajs, xx_gen, uu_gen = main()
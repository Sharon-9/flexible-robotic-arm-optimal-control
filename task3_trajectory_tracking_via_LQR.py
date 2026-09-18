#
# Task 3 - Trajectory tracking via LQR
#

import numpy as np
import matplotlib.pyplot as plt
import signal

from task0_discretized_dynamics import ns, ni, dt, dynamics
from task2_trajectory_generation import main as task2_main, TT, tt_hor
from lqr_solver import ltv_lqr_tracking_gains
from rollout import simulate_lqr_tracking
from control import dare
from linearization import dynamics_jacobian

signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({"font.size": 22})

Qreg = np.diag([1, 1, 0.7, 0.7])
Rreg = np.diag([0.1])


def main():

    xx_gen, uu_gen, _, _, _, _, _ = task2_main()

    Aeq, Beq = dynamics_jacobian(xx_gen[:, -1], uu_gen[:, -1])
    P_dare, _, _ = dare(Aeq, Beq, Qreg, Rreg)
    QregT = np.array(P_dare)
    QregT = 0.5 * (QregT + QregT.T)

    KK_track, _ = ltv_lqr_tracking_gains(
        xx_gen, uu_gen, Qreg, Rreg, QregT, TT
    )

    perturbations = [
        ("theta1 2 deg", np.array([np.deg2rad(2.0), 0.0, 0.0, 0.0])),
        ("theta2 2 deg", np.array([0.0, np.deg2rad(2.0), 0.0, 0.0])),
        ("theta1/theta2 2 deg", np.array([np.deg2rad(2.0), -np.deg2rad(2.0), 0.0, 0.0])),
        ("theta1 5 deg", np.array([np.deg2rad(5.0), 0.0, 0.0, 0.0])),
        ("theta1 10 deg", np.array([np.deg2rad(10.0), 0.0, 0.0, 0.0])),
    ]

    tracking_trajs = []

    for label, pp in perturbations:
        x0_track = xx_gen[:, 0] + pp # nominal initial state + perturbation

        xx_track, uu_track = simulate_lqr_tracking(
            xx_gen, uu_gen, KK_track, dynamics, x0_track
        )

        tracking_trajs.append((label, pp, xx_track, uu_track))

    # --------------------------------------------------------
    # Plot 1: generated trajectory and tracking trajectories
    # --------------------------------------------------------
    fig, axs = plt.subplots(ns + 1, 1, sharex=True, figsize=(11, 12))

    labels_x = [r'$\theta_1$', r'$\theta_2$', r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
    for i in range(ns):
        axs[i].plot(tt_hor, xx_gen[i, :], 'k--', linewidth=2, label='generated')

        for kk_i, (label, pp, xx_i, uu_i) in enumerate(tracking_trajs):
            axs[i].plot(tt_hor, xx_i[i, :], linewidth=1.5, label=label)

        axs[i].set_ylabel(labels_x[i])
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc='best', fontsize=10)

    axs[ns].plot(tt_hor, uu_gen[0, :], 'k--', linewidth=2, label='generated')

    for kk_i, (label, pp, xx_i, uu_i) in enumerate(tracking_trajs):
        axs[ns].plot(tt_hor, uu_i[0, :], linewidth=1.5, label=label)

    axs[ns].set_ylabel(r'$u$')
    axs[ns].set_xlabel('time [s]')
    axs[ns].grid(True, alpha=0.3)
    axs[ns].legend(loc='best', fontsize=10)

    fig.suptitle('Task 3 - Generated and tracking trajectories')
    plt.tight_layout()
    plt.show(block = False)
    

    # --------------------------------------------------------
    # Plot 2: tracking error norm
    # --------------------------------------------------------
    plt.figure()

    for kk_i, (label, pp, xx_i, uu_i) in enumerate(tracking_trajs):
        tracking_error = xx_i - xx_gen
        tracking_error_norm = np.linalg.norm(tracking_error, axis=0)

        plt.semilogy(
            tt_hor,
            tracking_error_norm,
            linewidth=2,
            label=label
        )

    plt.xlabel('time [s]')
    plt.ylabel(r'$\|x_t - x^{gen}_t\|$')
    plt.title('Task 3 - Tracking error norm')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    

    return KK_track, tracking_trajs, xx_gen, uu_gen


if __name__ == "__main__":
    KK_track, tracking_trajs, xx_gen, uu_gen = main()
#
# Task 2 - Trajectory generation (II)
#

import numpy as np
import matplotlib.pyplot as plt
import signal

from task0_discretized_dynamics import ns, ni, dt
from reference_curves import compute_quasi_static_trajectory
from costs import total_cost
from linearization import build_local_model, dynamics_jacobian
from lqr_solver import solve_affine_lqr, ltv_lqr_tracking_gains
from rollout import compute_direction_trajectory, closed_loop_rollout
from line_search import armijo_line_search
from initial_guess import initial_guess_from_quasi_static
from costates import compute_reduced_gradient
from control import dare

signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({"font.size": 22})

tf = 8.5
TT = int(tf / dt) + 1
tt_hor = np.linspace(0, tf, TT)

max_iters = 30
term_cond = 1e-4

stepsize_0 = 1.0
beta = 0.7
cc = 0.5
armijo_maxiters = 20

plot_descent_iterations = [0, 1, 9, 10]

theta_start = np.deg2rad(25.0)
theta_end = np.deg2rad(-25.0)

# Cost matrices for Newton optimization
QQ = np.diag([1, 1, 0.7, 0.7])
RR = np.diag([0.1])

# Cost matrices for first guess through LQR
Qreg = np.diag([1, 1, 0.7, 0.7])
Rreg = np.diag([0.1])


def plot_armijo_descent(xx, uu, xx_ref, uu_ref, KK, sigma, dx, gamma_tests, JJ_tests, descent, kk):
    JJ = total_cost(xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT)

    gamma_domain = np.linspace(0.0, 1.05 * max(stepsize_0, gamma_tests[0]), 50)
    JJ_domain = np.zeros_like(gamma_domain)

    for i, gg in enumerate(gamma_domain):
        xx_temp, uu_temp = closed_loop_rollout(xx, uu, KK, sigma, gg, TT)
        JJ_domain[i] = total_cost(xx_temp, uu_temp, xx_ref, uu_ref, QQ, RR, QQf, TT)

    JJ_lin = JJ + descent * gamma_domain
    JJ_arm = JJ + cc * descent * gamma_domain

    plt.figure()
    plt.plot(gamma_domain, JJ_domain, linewidth=2, label='J(u^k + closed-loop step)')
    plt.plot(gamma_domain, JJ_lin, linewidth=2, label='first-order model')
    plt.plot(gamma_domain, JJ_arm, '--', linewidth=2, label='Armijo line')
    plt.scatter(gamma_tests, JJ_tests, s=60, zorder=5, label='tested stepsizes')
    plt.xlabel('stepsize')
    plt.ylabel('cost')
    plt.title(f'Armijo descent plot - iteration {kk}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block = False)
    


def main():

    # Construction of quasi static trajectory (not dynamically feasible)
    xx_qs, uu_qs = compute_quasi_static_trajectory(theta_start, theta_end, TT)

    xx_ref = xx_qs.copy()
    uu_ref = uu_qs.copy()

    # Replace null velocities with derivative of theta1 and theta2
    xx_ref[2, :] = np.gradient(xx_ref[0, :], dt)
    xx_ref[3, :] = np.gradient(xx_ref[1, :], dt)

    print("Quasi-static trajectory:")
    print("x_qs initial =", xx_qs[:, 0])
    print("u_qs initial =", uu_qs[:, 0])
    print()

    print("Final quasi-static point:")
    print("x_qs final =", xx_qs[:, -1])
    print("u_qs final =", uu_qs[:, -1])
    print()

    global QQf, QregT

    # Linearize system with CasADi and compute terminal cost with DARE
    Aeq, Beq = dynamics_jacobian(xx_qs[:, -1], uu_qs[:, -1])
    P_dare, _, _ = dare(Aeq, Beq, QQ, RR)

    P_dare = np.array(P_dare)
    P_dare = 0.5 * (P_dare + P_dare.T)

    QQf = np.array(P_dare)
    QregT = np.array(P_dare)

    # First guess with LQR tracking of quasi static trajectory
    KK_track, PP_track = ltv_lqr_tracking_gains(xx_qs, uu_qs, Qreg, Rreg, QregT, TT)
    

    # Initial guess is simulated through real dynamics => feasible
    x0 = xx_ref[:, 0].copy()
    xx, uu = initial_guess_from_quasi_static(xx_qs, uu_qs, KK_track, x0, TT)

    JJ_hist = []
    descent_norm_hist = []
    grad_norm_hist = []

    intermediate_trajs = []
    intermediate_indices = [0, 1, 2, 5, max_iters - 1]


    # Newton optimization
    for kk in range(max_iters):
        JJ = total_cost(xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT)
        JJ_hist.append(JJ)

        grad_u, _ = compute_reduced_gradient(
            xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT
        )
        grad_norm = np.linalg.norm(grad_u)
        grad_norm_hist.append(grad_norm)

        AA, BB, qq, rr, QQseq, RRseq, SSseq, qqT, QQT, cc_dyn = \
            build_local_model(
                xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT
            )
        
        KK, sigma, PP, pp = solve_affine_lqr(
            AA, BB, qq, rr, QQseq, RRseq, SSseq, qqT, QQT, cc_dyn, TT
        )
                
      
        dx, du = compute_direction_trajectory(AA, BB, KK, sigma, cc_dyn, TT)

        descent_norm = np.linalg.norm(du)
        descent_norm_hist.append(descent_norm)

        print(
            f"iter {kk:02d} | cost = {JJ:.6e} | "
            f"||grad|| = {grad_norm:.6e} | "
            f"||du|| = {descent_norm:.6e}"
        )

        if descent_norm <= term_cond:
            print("\nConvergence reached based on descent direction norm.")
            break

        if kk in intermediate_indices:
            intermediate_trajs.append((kk, xx.copy(), uu.copy()))


        # Newton direction applied with rollout in closed loop
        gamma, xx_new, uu_new, JJ_new, descent, gamma_tests, JJ_tests = armijo_line_search(
            xx, uu, xx_ref, uu_ref, KK, sigma, dx, du,
            QQ, RR, QQf, TT,
            stepsize_0, beta, cc, armijo_maxiters
        )

        if kk in plot_descent_iterations:
            plot_armijo_descent(xx, uu, xx_ref, uu_ref, KK, sigma, dx, gamma_tests, JJ_tests, descent, kk)

        print(f"         Armijo stepsize = {gamma:.6e} | new cost = {JJ_new:.6e}")


        xx = xx_new
        uu = uu_new


    xx_opt = xx
    uu_opt = uu


    # --------------------------------------------------------
    # Plot 1: optimal trajectory and desired curve
    # --------------------------------------------------------
    fig, axs = plt.subplots(ns + 1, 1, sharex=True, figsize=(11, 12))

    labels_x = [r'$\theta_1$', r'$\theta_2$', r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']
    for i in range(ns):
        axs[i].plot(tt_hor, xx_ref[i, :], '--', linewidth=2, label='desired')
        axs[i].plot(tt_hor, xx_opt[i, :], linewidth=2, label='optimal')
        axs[i].set_ylabel(labels_x[i])
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc='best')

    axs[ns].plot(tt_hor, uu_ref[0, :], '--', linewidth=2, label='desired')
    axs[ns].plot(tt_hor, uu_opt[0, :], linewidth=2, label='optimal')
    axs[ns].set_ylabel(r'$u$')
    axs[ns].set_xlabel('time [s]')
    axs[ns].grid(True, alpha=0.3)
    axs[ns].legend(loc='best')

    fig.suptitle('Task 2 - Optimal trajectory and desired curve')
    plt.tight_layout()
    plt.show(block = False)
    

    # --------------------------------------------------------
    # Plot 2: optimal trajectory, desired curve and intermediate trajectories
    # --------------------------------------------------------
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(11, 10))

    axs[0].plot(tt_hor, xx_ref[0, :], 'k--', linewidth=2, label='desired')
    axs[0].plot(tt_hor, xx_opt[0, :], linewidth=2.5, label='optimal')
    for kk_i, xx_i, _ in intermediate_trajs:
        axs[0].plot(tt_hor, xx_i[0, :], alpha=0.5, linewidth=1.5, label=f'iter {kk_i}')
    axs[0].set_ylabel(r'$\theta_1$')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='best', fontsize=10)

    axs[1].plot(tt_hor, xx_ref[1, :], 'k--', linewidth=2, label='desired')
    axs[1].plot(tt_hor, xx_opt[1, :], linewidth=2.5, label='optimal')
    for kk_i, xx_i, _ in intermediate_trajs:
        axs[1].plot(tt_hor, xx_i[1, :], alpha=0.5, linewidth=1.5)
    axs[1].set_ylabel(r'$\theta_2$')
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(tt_hor, uu_ref[0, :], 'k--', linewidth=2, label='desired')
    axs[2].plot(tt_hor, uu_opt[0, :], linewidth=2.5, label='optimal')
    for kk_i, _, uu_i in intermediate_trajs:
        axs[2].plot(tt_hor, uu_i[0, :], alpha=0.5, linewidth=1.5)
    axs[2].set_ylabel(r'$u$')
    axs[2].set_xlabel('time [s]')
    axs[2].grid(True, alpha=0.3)

    fig.suptitle('Task 2 - Optimal, desired and intermediate trajectories')
    plt.tight_layout()
    plt.show(block = False)
    

    # --------------------------------------------------------
    # Plot 3: norm of descent direction
    # --------------------------------------------------------
    plt.figure()
    plt.semilogy(np.arange(len(descent_norm_hist)), descent_norm_hist, marker='o', linewidth=2)
    plt.xlabel('iteration')
    plt.ylabel(r'$\|\Delta u\|$')
    plt.title('Task 2 - Norm of the descent direction')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show(block = False)
    

    # --------------------------------------------------------
    # Plot 4: norm of the reduced gradient
    # --------------------------------------------------------
    plt.figure()
    plt.semilogy(np.arange(len(grad_norm_hist)), grad_norm_hist, marker='o', linewidth=2)
    plt.xlabel('iteration')
    plt.ylabel(r'$\|\nabla J(u)\|$')
    plt.title('Task 2 - Norm of the reduced gradient')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show(block = False)
    

    # --------------------------------------------------------
    # Plot 5: cost along iterations
    # --------------------------------------------------------
    plt.figure()
    plt.semilogy(np.arange(len(JJ_hist)), JJ_hist, marker='o', linewidth=2)
    plt.xlabel('iteration')
    plt.ylabel(r'$J^k$')
    plt.title('Task 2 - Cost along iterations')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()
    

    return xx_opt, uu_opt, xx_ref, uu_ref, JJ_hist, descent_norm_hist, grad_norm_hist


if __name__ == "__main__":
    xx_opt, uu_opt, xx_ref, uu_ref, JJ_hist, descent_norm_hist, grad_norm_hist = main()
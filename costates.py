#
# Costate and reduced gradient computation
#

import numpy as np

from task0_discretized_dynamics import ns, ni
from linearization import dynamics_jacobian
from costs import cost_derivatives, terminal_cost_derivatives


def compute_costates(xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT):
    """
    Compute the costate sequence lambda_t for the reduced cost J(u),
    along a feasible trajectory (xx, uu).

    Dynamics:
        x_{t+1} = f(x_t, u_t)

    Cost:
        J = sum_{t=0}^{T-1} l(x_t, u_t) + l_T(x_T)

    Backward equations:
        lambda_T = d l_T / d x
        lambda_t = d l_t / d x + A_t^T lambda_{t+1}
    """
    AA = np.zeros((ns, ns, TT - 1))
    BB = np.zeros((ns, ni, TT - 1))

    qq = np.zeros((ns, TT - 1))
    rr = np.zeros((ni, TT - 1))

    for t in range(TT - 1):
        A_t, B_t = dynamics_jacobian(xx[:, t], uu[:, t])
        q_t, r_t, _, _, _ = cost_derivatives(
            xx[:, t], uu[:, t], xx_ref[:, t], uu_ref[:, t], QQ, RR, ni, ns
        )

        AA[:, :, t] = A_t
        BB[:, :, t] = B_t
        qq[:, t] = q_t
        rr[:, t] = r_t

    qT, _ = terminal_cost_derivatives(xx[:, TT - 1], xx_ref[:, TT - 1], QQf)

    lambdas = np.zeros((ns, TT))
    lambdas[:, TT - 1] = qT # terminal condition for lambda_T

    for t in range(TT - 2, -1, -1):
        A_t = AA[:, :, t]
        lambdas[:, t] = qq[:, t] + A_t.T @ lambdas[:, t + 1] # costates recursion

    return lambdas, AA, BB, qq, rr, qT


def compute_reduced_gradient(xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT):
    """
    Compute the reduced gradient dJ/du_t explicitly using the costates:

        g_t = d l_t / d u + B_t^T lambda_{t+1}

    Returns
    -------
    grad_u : shape (ni, TT-1)
    lambdas : shape (ns, TT)
    """
    lambdas, AA, BB, qq, rr, qT = compute_costates(
        xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT
    )

    grad_u = np.zeros((ni, TT - 1))

    for t in range(TT - 1):
        B_t = BB[:, :, t]
        grad_u[:, t] = rr[:, t] + B_t.T @ lambdas[:, t + 1] # reduced gradient

    return grad_u, lambdas
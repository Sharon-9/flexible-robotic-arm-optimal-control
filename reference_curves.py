#
# Reference curves for Task 1 and Task 2
#

import numpy as np

from task0_discretized_dynamics import ns, ni, dt
from equilibria import compute_equilibrium


def sigmoid_transition_profile(TT, t1_ratio=0.25, t2_ratio=0.75, sharpness=10.0):
    """
    Smooth sigmoid-like transition profile s_t in [0,1]:
      - s = 0 in the initial constant part
      - sigmoid transition in the middle
      - s = 1 in the final constant part
    """
    s = np.zeros(TT)

    t1 = int(t1_ratio * TT)
    t2 = int(t2_ratio * TT)

    s[:t1] = 0.0
    s[t2:] = 1.0

    if t2 > t1:
        tau_vec = np.linspace(0.0, 1.0, t2 - t1)

        sig = 1.0 / (1.0 + np.exp(-sharpness * (tau_vec - 0.5)))

        sig = (sig - sig[0]) / (sig[-1] - sig[0])

        s[t1:t2] = sig

    return s


def build_reference(xx_eq1, uu_eq1, xx_eq2, uu_eq2, TT):
    """
    Task 1 reference curve:
      - smooth sigmoid interpolation on the angles
      - desired angular velocities obtained by numerical differentiation
      - smooth interpolation on the input
    """
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    s = sigmoid_transition_profile(TT)

    # Angles interpolation between the two equilibria
    theta1_ref = (1.0 - s) * xx_eq1[0] + s * xx_eq2[0]
    theta2_ref = (1.0 - s) * xx_eq1[1] + s * xx_eq2[1]

    # Desired velocities from numerical differentiation
    dtheta1_ref = np.gradient(theta1_ref, dt)
    dtheta2_ref = np.gradient(theta2_ref, dt)

    # Input interpolation
    u_ref = (1.0 - s) * uu_eq1[0] + s * uu_eq2[0]

    xx_ref[0, :] = theta1_ref
    xx_ref[1, :] = theta2_ref
    xx_ref[2, :] = dtheta1_ref
    xx_ref[3, :] = dtheta2_ref

    uu_ref[0, :] = u_ref

    return xx_ref, uu_ref


def compute_quasi_static_trajectory(theta1_start, theta1_end, TT):
    """
    Task 2 quasi-static trajectory = collection of equilibria.
    """
    xx_qs = np.zeros((ns, TT))
    uu_qs = np.zeros((ni, TT))

    s = sigmoid_transition_profile(TT)
    theta2_guess = 0.0 

    for t in range(TT):
        # For each intermediate point compute an equilibrium
        theta1_t = (1.0 - s[t]) * theta1_start + s[t] * theta1_end
        xx_eq_t, uu_eq_t = compute_equilibrium(theta1_t, guess_theta2=theta2_guess) 

        xx_qs[:, t] = xx_eq_t
        uu_qs[:, t] = uu_eq_t

        theta2_guess = xx_eq_t[1]

    return xx_qs, uu_qs
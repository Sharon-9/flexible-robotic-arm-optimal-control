#
# Equilibria utilities for the flexible robotic arm
#

import numpy as np
from scipy.optimize import fsolve

from task0_discretized_dynamics import m1, m2, l1, r1, r2, g


def gravity_terms(theta1, theta2):
    """
    Gravity vector G(theta1, theta2).
    """
    G1 = g * (m1 * r1 + m2 * l1) * np.sin(theta1) + g * m2 * r2 * np.sin(theta1 + theta2)
    G2 = g * m2 * r2 * np.sin(theta1 + theta2)
    return np.array([G1, G2])


def compute_equilibrium(theta1_target, guess_theta2=0.0):
    """
    Compute one equilibrium with:
      - dtheta1 = dtheta2 = 0
      - ddtheta1 = ddtheta2 = 0
      - theta1 fixed (desired value)
      - theta2 obtained from G2(theta1, theta2) = 0 [find theta2 that makes the passive joint balanced]
      - u_eq = G1(theta1, theta2) [compute torque needed at joint 1 to hold the arm still, to compensate gravity]
    """
    def residual(z):
        theta1, theta2 = z
        GG = gravity_terms(theta1, theta2)
        return np.array([ # fsolve impose them to be zero => gravity torque at joint 2 = 0 and theta1=desired angle
            GG[1],
            theta1 - theta1_target
        ])

    sol = fsolve(residual, np.array([theta1_target, guess_theta2])) 
    theta1_eq = sol[0]
    theta2_eq = sol[1]

    u_eq = gravity_terms(theta1_eq, theta2_eq)[0]

    xx_eq = np.array([theta1_eq, theta2_eq, 0.0, 0.0])
    uu_eq = np.array([u_eq])

    return xx_eq, uu_eq
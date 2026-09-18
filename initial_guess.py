#
# Initial guess utilities
#

import numpy as np

from task0_discretized_dynamics import dynamics, ns, ni


def initial_guess_simple(x0, uu_ref, TT):
    """
    Task 1 initial guess: use the reference input and propagate the true dynamics.
    """
    xx_init = np.zeros((ns, TT))
    uu_init = uu_ref.copy()

    xx_init[:, 0] = x0
    for t in range(TT - 1):
        xx_init[:, t + 1] = dynamics(xx_init[:, t], uu_init[:, t])

    return xx_init, uu_init


def initial_guess_from_quasi_static(xx_qs, uu_qs, KK, x0, TT):
    """
    Task 2 initial guess from quasi-static tracking.
    """
    xx_init = np.zeros((ns, TT))
    uu_init = np.zeros((ni, TT))

    xx_init[:, 0] = x0

    for t in range(TT - 1):
        uu_init[:, t] = uu_qs[:, t] + KK[:, :, t] @ (xx_init[:, t] - xx_qs[:, t]) #feedback gains K are obtained from Riccati in ltv_lqr_tracking_gains
                                                                                    
        xx_init[:, t + 1] = dynamics(xx_init[:, t], uu_init[:, t])

    uu_init[:, TT - 1] = uu_init[:, TT - 2]

    return xx_init, uu_init
#
# LQR solvers
#

import numpy as np

from task0_discretized_dynamics import ns, ni
from linearization import dynamics_jacobian


def solve_affine_lqr(AA, BB, qq, rr, QQseq, RRseq, SSseq, qqT, QQT, cc_dyn, TT):
    """
    Solve the affine LQR subproblem for Newton step.
    """
    KK = np.zeros((ni, ns, TT - 1))
    sigma = np.zeros((ni, TT - 1))

    PP = np.zeros((ns, ns, TT))
    pp = np.zeros((ns, TT))

    # Terminal condition of the recursion
    PP[:, :, TT - 1] = QQT
    pp[:, TT - 1] = qqT

    for t in range(TT - 2, -1, -1):
        AAt = AA[:, :, t]
        BBt = BB[:, :, t]
        QQt = QQseq[:, :, t]
        RRt = RRseq[:, :, t]
        SSt = SSseq[:, :, t]
        qqt = qq[:, t]
        rrt = rr[:, t]
        cct = cc_dyn[:, t]

        PPtp = PP[:, :, t + 1]
        pptp = pp[:, t + 1]

        MMt = RRt + BBt.T @ PPtp @ BBt
        MMt = 0.5 * (MMt + MMt.T) # force the matrix to be symmetric (avoid numerical errors)

        rhsK = SSt + BBt.T @ PPtp @ AAt
        rhsSigma = rrt + BBt.T @ pptp + BBt.T @ PPtp @ cct

        KKt = -np.linalg.solve(MMt, rhsK)
        sigt = -np.linalg.solve(MMt, rhsSigma)

        PPt = QQt + AAt.T @ PPtp @ AAt - KKt.T @ MMt @ KKt
        ppt = qqt + AAt.T @ pptp + AAt.T @ PPtp @ cct - KKt.T @ MMt @ sigt

        PPt = 0.5 * (PPt + PPt.T)

        # Save results for time t
        KK[:, :, t] = KKt
        sigma[:, t] = sigt
        PP[:, :, t] = PPt
        pp[:, t] = ppt

    return KK, sigma, PP, pp


def ltv_lqr_tracking_gains(xx_nom, uu_nom, Qreg, Rreg, QregT, TT):
    """
    Compute LTV LQR gains along a nominal curve.
    """
    AA = np.zeros((ns, ns, TT - 1))
    BB = np.zeros((ns, ni, TT - 1))

    for t in range(TT - 1):
        # Linearize along nominal trajectory (xx_qs, uu_qs)
        A_t, B_t = dynamics_jacobian(xx_nom[:, t], uu_nom[:, t])
        AA[:, :, t] = A_t
        BB[:, :, t] = B_t

    PP = np.zeros((ns, ns, TT))
    KK = np.zeros((ni, ns, TT - 1))

    # Terminal condition
    PP[:, :, TT - 1] = QregT
    
    # Solve Riccati backward recursion
    for t in range(TT - 2, -1, -1):
        A_t = AA[:, :, t]
        B_t = BB[:, :, t]
        P_tp = PP[:, :, t + 1]

        MM = Rreg + B_t.T @ P_tp @ B_t
        MM = 0.5 * (MM + MM.T)

        # Compute gain
        KK_t = -np.linalg.solve(MM, B_t.T @ P_tp @ A_t)

        PP_t = Qreg + A_t.T @ P_tp @ A_t - KK_t.T @ MM @ KK_t
        PP_t = 0.5 * (PP_t + PP_t.T)

        # Save gain and Riccati matrix at time t
        KK[:, :, t] = KK_t
        PP[:, :, t] = PP_t

    return KK, PP
import numpy as np
import casadi as ca


def solver_linear_mpc_tracking(AA, BB, QQ, RR, QQf, xxt, xx_gen, uu_gen, kk, umax=1, umin=-1, T_pred=5, ipopt_opts=None):
    """
    Linear constrained MPC for trajectory tracking using CasADi Opti.

    The optimization is written in error coordinates:

        dx_t = x_t - x_gen_t
        du_t = u_t - u_gen_t

    Dynamics:

        dx_{t+1} = A_t dx_t + B_t du_t

    Input constraint on the real input:

        umin <= u_gen_t + du_t <= umax

    Returns:
        u_t, dx_pred, du_pred
    """

    QQ = ca.DM(QQ)
    RR = ca.DM(RR)
    QQf = ca.DM(QQf)

    xxt = np.array(xxt).squeeze()

    ns = AA.shape[0]
    ni = BB.shape[1]
    TT = AA.shape[2] + 1

    H = min(T_pred, TT - kk)

    dx0 = xxt - xx_gen[:, kk]

    opti = ca.Opti()

    DX = opti.variable(ns, H)
    DU = opti.variable(ni, H - 1)

    cost = 0

    opti.subject_to(DX[:, 0] == ca.DM(dx0))

    for tt in range(H - 1):

        AAt = ca.DM(AA[:, :, kk + tt])
        BBt = ca.DM(BB[:, :, kk + tt])

        dxt = DX[:, tt]
        dut = DU[:, tt]

        cost += ca.mtimes([dxt.T, QQ, dxt])
        cost += ca.mtimes([dut.T, RR, dut])

        opti.subject_to(DX[:, tt + 1] == AAt @ dxt + BBt @ dut)

        # actual input = nominal input + input correction
        ut_real = ca.DM(uu_gen[:, kk + tt]) + dut

        opti.subject_to(ut_real <= umax)
        opti.subject_to(ut_real >= umin)

    dxT = DX[:, H - 1]
    cost += ca.mtimes([dxT.T, QQf, dxT])

    opti.minimize(cost)

    if ipopt_opts is None:
        ipopt_opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 1000
        }

    opti.solver("ipopt", ipopt_opts)

    try:
        sol = opti.solve()
    except RuntimeError as e:
        opti.set_initial(DX, np.zeros((ns, H)))
        opti.set_initial(DU, np.zeros((ni, H - 1)))

        try:
            sol = opti.solve()
        except Exception as e2:
            print("solver_linear_mpc_tracking: solver failed:", e2)
            return None, None, None

    du0 = sol.value(DU[:, 0])
    u0 = uu_gen[:, kk] + du0

    dx_pred = sol.value(DX)
    du_pred = sol.value(DU)

    return np.asarray(u0).squeeze(), np.asarray(dx_pred), np.asarray(du_pred)
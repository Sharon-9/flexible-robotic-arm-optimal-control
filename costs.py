import numpy as np
from task0_discretized_dynamics import dt


def stage_cost(xx, uu, xx_ref, uu_ref, QQ, RR):
    dx = xx - xx_ref
    du = uu - uu_ref
    ll = (0.5 * dx.T @ QQ @ dx + 0.5 * du.T @ RR @ du)
    return float(ll)


def terminal_cost(xx, xx_ref, QQf):
    dx = xx - xx_ref
    llT = 0.5 * dx.T @ QQf @ dx
    return float(llT)


def total_cost(xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT):
    JJ = 0.0
    for t in range(TT - 1):
        JJ += stage_cost(xx[:, t], uu[:, t], xx_ref[:, t], uu_ref[:, t], QQ, RR)
    JJ += terminal_cost(xx[:, TT - 1], xx_ref[:, TT - 1], QQf)
    return JJ


def cost_derivatives(xx, uu, xx_ref, uu_ref, QQ, RR, ni, ns):
    qq = QQ @ (xx - xx_ref)
    rr = RR @ (uu - uu_ref)
    QQt = QQ.copy()
    RRt =  RR.copy()
    SSt = np.zeros((ni, ns))
    return qq, rr, QQt, RRt, SSt


def terminal_cost_derivatives(xx, xx_ref, QQf):
    qqT = QQf @ (xx - xx_ref)
    QQT = QQf.copy()
    return qqT, QQT
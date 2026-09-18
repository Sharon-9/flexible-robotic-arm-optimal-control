#
# Numerical linearization and local Newton model construction
#

import numpy as np
import casadi as ca

from task0_discretized_dynamics import (
    ns, ni, dt,
    m1, m2, l1, r1, r2, I1, I2, g, f1, f2
)
from costs import cost_derivatives, terminal_cost_derivatives


def build_casadi_dynamics(): # symbolically build discrete dynamics 
    # Define symbolic variables
    xx = ca.SX.sym('xx', ns)
    uu = ca.SX.sym('uu', ni)

    theta1  = xx[0]
    theta2  = xx[1]
    dtheta1 = xx[2]
    dtheta2 = xx[3]

    MM11 = I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) + 2*m2*l1*r2*ca.cos(theta2)
    MM12 = I2 + m2*r2**2 + m2*l1*r2*ca.cos(theta2)
    MM21 = MM12
    MM22 = I2 + m2*r2**2

    # Build the matrix
    MM = ca.vertcat(
        ca.horzcat(MM11, MM12),
        ca.horzcat(MM21, MM22)
    )

    CC1 = -m2*l1*r2*dtheta2*ca.sin(theta2)*(dtheta2 + 2*dtheta1)
    CC2 =  m2*l1*r2*ca.sin(theta2)*dtheta1**2
    CC = ca.vertcat(CC1, CC2)

    GG1 = g*(m1*r1 + m2*l1)*ca.sin(theta1) + g*m2*r2*ca.sin(theta1 + theta2)
    GG2 = g*m2*r2*ca.sin(theta1 + theta2)
    GG = ca.vertcat(GG1, GG2)

    FF = ca.vertcat(
        ca.horzcat(f1, 0.0),
        ca.horzcat(0.0, f2)
    )

    dd = ca.vertcat(dtheta1, dtheta2)
    tau = ca.vertcat(uu[0], 0.0)

    ddtheta = ca.solve(MM, tau - CC - FF @ dd - GG) 

    # Forward Euler discretization (xxp is symbolic discrete dynamics)
    xxp = ca.vertcat(
        theta1  + dt * dtheta1,
        theta2  + dt * dtheta2,
        dtheta1 + dt * ddtheta[0],
        dtheta2 + dt * ddtheta[1]
    )

    # Symbolic Jacobians
    dfx = ca.jacobian(xxp, xx)
    dfu = ca.jacobian(xxp, uu) 

    # Create CasADi functions numerically evaluable
    f_fun = ca.Function('f_fun', [xx, uu], [xxp])
    jac_fun = ca.Function('jac_fun', [xx, uu], [dfx, dfu]) # computes A,B

    return f_fun, jac_fun

f_fun, jac_fun = build_casadi_dynamics()


def dynamics_jacobian(xx, uu): # to obtain the numerical Jacobians (numpy arrays)
    """
    Jacobian of the discrete-time dynamics computed with CasADi.
    """

    xx = np.asarray(xx).reshape(ns,)
    uu = np.asarray(uu).reshape(ni,)

    # Evaluate Jacobians at the given point
    dfx, dfu = jac_fun(xx, uu)

    # Convert CasADi result into a NumPy array
    dfx = np.array(dfx) # A
    dfu = np.array(dfu) # B

    return dfx, dfu


def build_local_model(xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT):  # xx,uu = current feasible trajectory
    """
    Build the local LQ model used by the regularized Newton-like method.
    The quadratic terms are based on the Hessians of the cost only,
    as suggested in the project guidelines.
    """
    AA = np.zeros((ns, ns, TT - 1)) 
    BB = np.zeros((ns, ni, TT - 1)) 

    # Linear terms of the cost
    qq = np.zeros((ns, TT - 1)) 
    rr = np.zeros((ni, TT - 1)) 

    # Quadratic matrices of local cost Q_t, R_t, S_t (S_t = 0, no cross terms)
    QQseq = np.zeros((ns, ns, TT - 1))
    RRseq = np.zeros((ni, ni, TT - 1))
    SSseq = np.zeros((ni, ns, TT - 1))

    cc_dyn = np.zeros((ns, TT - 1)) # = 0 

    for t in range(TT - 1):
        # Linearize at the current point
        dfx, dfu = dynamics_jacobian(xx[:, t], uu[:, t])

        # Store Jacobians
        AA[:, :, t] = dfx
        BB[:, :, t] = dfu

        q_t, r_t, Q_t, R_t, S_t = cost_derivatives(
            xx[:, t], uu[:, t], xx_ref[:, t], uu_ref[:, t], QQ, RR, ni, ns
        )

        qq[:, t] = q_t
        rr[:, t] = r_t
        QQseq[:, :, t] = Q_t
        RRseq[:, :, t] = R_t 
        SSseq[:, :, t] = S_t

    qqT, QQT = terminal_cost_derivatives(xx[:, TT - 1], xx_ref[:, TT - 1], QQf)

    return AA, BB, qq, rr, QQseq, RRseq, SSseq, qqT, QQT, cc_dyn


def linearize_trajectory(xx_gen, uu_gen, TT):
    AA = np.zeros((ns, ns, TT - 1))
    BB = np.zeros((ns, ni, TT - 1))

    for t in range(TT - 1):
        AA[:, :, t], BB[:, :, t] = dynamics_jacobian(xx_gen[:, t], uu_gen[:, t])

    return AA, BB
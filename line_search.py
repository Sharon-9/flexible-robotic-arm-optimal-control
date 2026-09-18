#
# Armijo line search
#

from costs import total_cost
from rollout import closed_loop_rollout
from costates import compute_reduced_gradient


def armijo_line_search(
    xx, uu, xx_ref, uu_ref, KK, sigma, dx, du,
    QQ, RR, QQf, TT,
    stepsize_0, beta, cc, armijo_maxiters
):
    """
    Armijo line search on the closed-loop update -> how much of
    the (Δx,Δu) direction to apply
    """

    # Compute current cost
    JJ = total_cost(xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT)

    # Compute reduced gradient
    grad_u, _ = compute_reduced_gradient(xx, uu, xx_ref, uu_ref, QQ, RR, QQf, TT)

    descent = 0.0
    for t in range(TT - 1):
        descent += grad_u[:, t].T @ du[:, t] 

    gamma = stepsize_0 # initial stepsize

    # Initially the best trajectory is the current one
    xx_best = xx.copy()
    uu_best = uu.copy()
    JJ_best = JJ

    # Store tested values of gamma and corresponding costs
    stepsizes = [] 
    costs = []

    accepted = False # set to true when Armijo finds a valid step

    for _ in range(armijo_maxiters): # try at most armijo_maxiters different stepsizes
        # Build a candidate new trajectory using the current gamma
        xx_temp, uu_temp = closed_loop_rollout(xx, uu, KK, sigma, gamma, TT)
        JJ_temp = total_cost(xx_temp, uu_temp, xx_ref, uu_ref, QQ, RR, QQf, TT) 

        stepsizes.append(gamma)
        costs.append(JJ_temp)

        # Armijo condition: checks if cost has decreased enough
        if JJ_temp <= JJ + cc * gamma * descent:
            # If condition is satisfied, save candidate as accepted new trajectory
            xx_best = xx_temp
            uu_best = uu_temp
            JJ_best = JJ_temp
            accepted = True
            break

        gamma *= beta # if condition is not satisfied, reduce stepsize gamma

    if not accepted:
        print("WARNING: Armijo did not find an acceptable stepsize.")
        print("grad^T du =", descent)
        print("last gamma =", gamma)
        print("old cost =", JJ)
        print("best returned cost =", JJ_best)


    return gamma, xx_best, uu_best, JJ_best, descent, stepsizes, costs
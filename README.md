================================================================
Project - Optimal Control of a Flexible Robotic Arm
Group #39 - Martina Raffaele, Sharon Patrone, Sarah Mercolino
================================================================

FILES DESCRIPTION
-----------------
task0_discretized_dynamics.py
    Defines the system parameters, the discrete-time nonlinear
    dynamics via Forward Euler, and the state/input dimensions.

task1_trajectory_generation.py
    Task 1: computes two equilibria, builds a sigmoid-interpolated
    reference curve, and runs the Newton-like algorithm to generate
    the optimal trajectory.

task2_trajectory_generation.py
    Task 2: computes a quasi-static reference trajectory, obtains
    an LQR-based initial guess, and runs the Newton-like algorithm
    on the new reference.

task3_trajectory_tracking_via_LQR.py
    Task 3: linearizes the dynamics along the generated trajectory
    and computes LTV-LQR tracking gains via Riccati recursion.
    Simulates tracking under five initial perturbations.

task4_trajectory_tracking_via_MPC.py
    Task 4: tracks the generated trajectory using a constrained
    receding-horizon linear MPC with prediction horizon Np=5.
    The controller operates in error coordinates and enforces
    bounds on the actual control input. Simulates the same five
    perturbation scenarios as Task 3.

task5_animation.py
    Task 5: produces an animation of the two-link arm executing
    the LQR tracking from Task 3. Saves the result as a GIF file.

equilibria.py
    Computes equilibrium configurations by solving the static
    gravity balance condition using scipy fsolve.

reference_curves.py
    Builds the sigmoid-interpolated reference (Task 1) and the
    quasi-static trajectory (Task 2).

linearization.py
    Computes the Jacobians of the discrete-time dynamics using
    CasADi algorithmic differentiation. Builds the local LQ model
    for the Newton algorithm.

costs.py
    Defines stage cost, terminal cost, total cost, and their
    derivatives.

lqr_solver.py
    Solves the affine LQR subproblem (Newton step) and the
    LTV-LQR tracking problem via backward Riccati recursion.

mpc_solver.py
    Solves the constrained finite-horizon linear MPC tracking
    problem in error coordinates using CasADi Opti and IPOPT.
    Enforces bounds on the actual control input and returns
    the first control action of the optimized sequence.

rollout.py
    Implements the closed-loop rollout, the Newton direction
    computation, the LQR tracking simulation, and the constrained
    MPC tracking simulation on the nonlinear system.

costates.py
    Computes the costate sequence and the reduced gradient
    of the cost function.

line_search.py
    Implements the Armijo backtracking line search.

initial_guess.py
    Provides the initial feasible trajectories for Task 1
    (reference input rollout) and Task 2 (LQR-tracked
    quasi-static trajectory).

HOW TO RUN
----------
Each task can be run independently. Tasks 3, 4 and 5 internally call task2_trajectory_generation.py
to obtain the generated trajectory, so running them will also
execute Task 2 first.

The animation produced by Task 5 can be saved as task5_animation.gif in the working directory by
setting SAVE_ANIMATION = True (Task 5).By default, the animation shows the trajectory with a 10 deg
    perturbation on theta1. To animate a different perturbation, change the index in the line:
        label, pp, xx_track, uu_track = tracking_trajs[4]
    where the index corresponds to:
        0 -> theta1 2 deg
        1 -> theta2 2 deg
        2 -> theta1/theta2 2 deg
        3 -> theta1 5 deg
        4 -> theta1 10 deg

## Report

A detailed description of the project, methodology, and results is available [here](report/Optimal_Control_Flexible_Robotic_Arm_Report.pdf).
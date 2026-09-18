# Optimal Control of a Flexible Robotic Arm

Optimal trajectory generation and trajectory tracking for an underactuated two-link robotic arm using nonlinear optimal control, LTV-LQR and Model Predictive Control.

Course project developed for **Optimal Control and Reinforcement Learning**  
MSc in Automation Engineering — University of Bologna

**Supervisor:** Prof. Giuseppe Notarstefano

## Overview

The project considers a planar two-link robotic arm actuated only at the first joint.

The nonlinear dynamics are modeled and discretized using Forward Euler. Optimal trajectories are generated through a Newton-like optimization method, while trajectory tracking is performed using both LTV-LQR and constrained Model Predictive Control.

The implementation includes:

- nonlinear dynamic modelling
- equilibrium computation
- trajectory generation
- Newton-like optimal control
- Armijo backtracking line search
- LTV-LQR trajectory tracking
- constrained MPC in error coordinates
- simulation under different initial perturbations
- animation of the controlled robotic arm

## Technologies

Python · NumPy · SciPy · CasADi · Matplotlib · Python Control

## Control Methods

### Optimal Trajectory Generation

Two trajectory generation approaches are implemented:

- sigmoid-interpolated reference between equilibrium configurations
- quasi-static reference trajectory with an LQR-based initial guess

The optimization is performed using a Newton-like method based on local affine LQR subproblems and backward Riccati recursion.

### LTV-LQR Tracking

The nonlinear system is linearized along the generated trajectory and an LTV-LQR controller is used to track it under different initial perturbations.

### Model Predictive Control

A constrained receding-horizon linear MPC is implemented in error coordinates.

The controller uses a prediction horizon of `Np = 5` and explicitly enforces bounds on the actual control input. The optimization problem is solved using CasADi Opti and IPOPT.

## Project Structure

```text
task0_discretized_dynamics.py         Nonlinear dynamics and discretization
task1_trajectory_generation.py        Trajectory generation — Task 1
task2_trajectory_generation.py        Trajectory generation — Task 2
task3_trajectory_tracking_via_LQR.py  LTV-LQR tracking
task4_trajectory_tracking_via_MPC.py  Constrained MPC tracking
task5_animation.py                    Robotic arm animation
```

Additional modules contain the equilibrium solver, linearization, cost functions, LQR and MPC solvers, rollout, costate computation and line search.

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run any task, for example:

```bash
python task3_trajectory_tracking_via_LQR.py
```

Tasks 3, 4 and 5 automatically generate the trajectory required from Task 2.

## Report

A detailed description of the methodology and results is available in the [project report](report/Optimal_Control_Flexible_Robotic_Arm_Report.pdf).

## Authors

- Sharon Patrone
- Martina Raffaele
- Sarah Mercolino

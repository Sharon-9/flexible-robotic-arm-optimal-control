import numpy as np

ns = 4
ni = 1

dt = 1e-2   # discretization stepsize - Forward Euler

# Model parameters - Set 1
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
r1 = 0.5
r2 = 0.5
I1 = 0.33
I2 = 0.33
g  = 9.81
f1 = 0.1
f2 = 0.1


def dynamics(xx, uu):
    """
    Discrete-time nonlinear dynamics of the flexible robotic arm

    Args
      - xx in R^4 : state at time t
      - uu in R^1 : input at time t

    Return
      - next state xxp = x_{t+1}
    """

    xx = xx[:, None]
    uu = uu[:, None]

    theta1  = xx[0, 0]
    theta2  = xx[1, 0]
    dtheta1 = xx[2, 0]
    dtheta2 = xx[3, 0]

    # Inertia matrix
    MM11 = I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2) + 2*m2*l1*r2*np.cos(theta2)
    MM12 = I2 + m2*r2**2 + m2*l1*r2*np.cos(theta2)
    MM21 = MM12
    MM22 = I2 + m2*r2**2
    MM = np.array([[MM11, MM12],
                   [MM21, MM22]])

    # Coriolis / centrifugal terms
    CC1 = -m2*l1*r2*dtheta2*np.sin(theta2)*(dtheta2 + 2*dtheta1)
    CC2 =  m2*l1*r2*np.sin(theta2)*dtheta1**2
    CC = np.array([[CC1],
                   [CC2]])

    # Gravity terms
    GG1 = g*(m1*r1 + m2*l1)*np.sin(theta1) + g*m2*r2*np.sin(theta1 + theta2)
    GG2 = g*m2*r2*np.sin(theta1 + theta2)
    GG = np.array([[GG1],
                   [GG2]])

    # Friction
    FF = np.array([[f1, 0.0],
                   [0.0, f2]])
    dd = np.array([[dtheta1],
                   [dtheta2]])

    # Input torque
    tau = np.array([[uu[0,0]],
                    [0.0]])

    # Accelerations
    ddtheta = np.linalg.solve(MM, tau - CC - FF @ dd - GG)

    # Forward Euler discretization
    xxp = np.zeros((ns, 1))
    xxp[0, 0] = theta1  + dt * dtheta1
    xxp[1, 0] = theta2  + dt * dtheta2
    xxp[2, 0] = dtheta1 + dt * ddtheta[0, 0]
    xxp[3, 0] = dtheta2 + dt * ddtheta[1, 0]

    xxp = xxp.squeeze()

    return xxp

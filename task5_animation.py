#
# Task 5 - Animation of the robot executing Task 3
#

import numpy as np
import matplotlib.pyplot as plt
import signal

from matplotlib.animation import FuncAnimation, PillowWriter

from task0_discretized_dynamics import l1, l2
from task3_trajectory_tracking_via_LQR import main as task3_main, tt_hor

signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({"font.size": 22})

SAVE_ANIMATION = True # toggle to save or not the created animation
FRAME_STEP = 20 # take a frame every 20 instants


def robot_points(xx):
    """
    Compute the Cartesian coordinates of the two-link robot.

    State:
        x = [theta1, theta2, dtheta1, dtheta2]

    Angles:
        theta1 is the angle of the first link with respect to the vertical direction.
        theta2 is the relative angle of the second link with respect to the first link.
    """
    theta1 = xx[0]
    theta2 = xx[1]

    x0 = 0.0
    y0 = 0.0

    # First link end
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    # Second link end
    x2 = x1 + l2 * np.sin(theta1 + theta2)
    y2 = y1 - l2 * np.cos(theta1 + theta2)

    return np.array([x0, x1, x2]), np.array([y0, y1, y2])


def main():

    KK_track, tracking_trajs, xx_gen, uu_gen = task3_main() # LQR gains, simulated trajectories with perturbations

    label, pp, xx_track, uu_track = tracking_trajs[4] # choose which perturbated trajectory to animate

    frames = np.arange(0, xx_gen.shape[1], FRAME_STEP)

    # Build Matplotlib figure and 3 objects
    fig, ax = plt.subplots()
    arm_gen, = ax.plot([], [], 'k--o', linewidth=2, markersize=6, label='generated')
    arm_track, = ax.plot([], [], 'o-', linewidth=3, markersize=7, label='LQR tracking')
    trace_track, = ax.plot([], [], linewidth=1.5, alpha=0.7, label='end-effector trace')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # To ensure the robot stays visible
    axis_lim = 1.2 * (l1 + l2)
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Task 5 - Animation of Task 3 ({label})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    x_trace = []
    y_trace = []

    def init():
        arm_gen.set_data([], [])
        arm_track.set_data([], [])
        trace_track.set_data([], [])
        time_text.set_text('')
        return arm_gen, arm_track, trace_track, time_text # needed by blit=True to know which objects change in time

    def update(frame):
        # For each time instant, compute cartesian coordinates of nominal robot
        # (optimal trajectory) and controlled robot (trajectory obtained under LQR)
        x_gen, y_gen = robot_points(xx_gen[:, frame])
        x_track, y_track = robot_points(xx_track[:, frame])

        # Update links
        arm_gen.set_data(x_gen, y_gen)
        arm_track.set_data(x_track, y_track)

        # Store final position of end effector
        x_trace.append(x_track[-1])
        y_trace.append(y_track[-1])
        trace_track.set_data(x_trace, y_trace)

        # Update text with current time 
        time_text.set_text(f't = {tt_hor[frame]:.2f} s')

        return arm_gen, arm_track, trace_track, time_text

    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=30, # 30 ms between each frame
        blit=True # update only changing objects for efficiency
    )

    # To save the created animation
    if SAVE_ANIMATION:
        ani.save('task5_animation.gif', writer=PillowWriter(fps=30))

    plt.tight_layout()
    plt.show()
    

    return ani


if __name__ == "__main__":
    ani = main()
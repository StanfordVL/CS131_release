import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, logm


def plot_frame(ax, T_local_from_global, label):
    assert T_local_from_global.shape == (4, 4)

    # Get rotation/translation of local origin wrt global frame
    R = T_local_from_global[:3, :3].T
    origin = -R @ T_local_from_global[:3, 3]

    # Draw line for each basis
    for direction, color in zip(R.T, "rgb"):
        ax.quiver(*origin, *direction, color=color, length=0.3, arrow_length_ratio=0.05)

    # Label
    ax.text(origin[0] - 0.1, origin[1], origin[2] + 0.0, "↙" + label, color="black")


def plot_square(ax, vertices):
    return ax.plot3D(vertices[0], vertices[1], vertices[2], "orange",)


def configure_ax(ax):
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(elev=20.0, azim=25)


def animate_transformation(
    filename, vertices_wrt_world, camera_from_world_transform, apply_transform,
):
    # Transformation parameters
    d = 1.0

    # Animation parameters
    start_pause = 20
    end_pause = 20

    num_rotation_frames = 20
    num_translation_frames = 20

    # First set up the figure and axes
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection="3d")
    configure_ax(ax)

    # Initial elements
    T_camera_from_world = camera_from_world_transform(d)
    plot_square(ax, vertices=vertices_wrt_world)
    plot_frame(
        ax, T_camera_from_world, label="Camera Frame",
    )
    plot_frame(
        ax, np.eye(4), label="World Frame",
    )

    # Animation function which updates figure data.  This is called sequentially
    def animate(i):
        print(".", end="")
        if i < start_pause:
            return (fig,)
        elif i >= start_pause + num_rotation_frames + num_translation_frames:
            return (fig,)
        else:
            i -= start_pause

        # Disclaimer: this is really inefficient!
        ax.clear()
        configure_ax(ax)
        if i < num_rotation_frames:
            R = expm(logm(T_camera_from_world[:3, :3]) * i / (num_rotation_frames - 1))
            t = np.zeros(3)
        else:
            i -= num_rotation_frames
            R = T_camera_from_world[:3, :3]
            t = i / (num_translation_frames - 1) * T_camera_from_world[:3, 3]

        T_camera_from_world_interp = np.eye(4)
        T_camera_from_world_interp[:3, :3] = R
        T_camera_from_world_interp[:3, 3] = t

        plot_square(
            ax, vertices=apply_transform(T_camera_from_world_interp, vertices_wrt_world)
        )
        plot_frame(
            ax,
            T_camera_from_world @ np.linalg.inv(T_camera_from_world_interp),
            label="Camera Frame",
        )
        plot_frame(
            ax, np.linalg.inv(T_camera_from_world_interp), label="World Frame",
        )

        return (fig,)

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=start_pause + num_rotation_frames + num_translation_frames + end_pause,
        interval=100,
        blit=True,
    )

    anim.save(filename, writer="pillow")
    plt.close()

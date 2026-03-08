from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.colors import Normalize, to_rgba

from hand.data.utils import load_data_compressed


def _traj_path(p):
    """Resolve path to traj data file."""
    p = Path(p)
    return p / "traj_data.dat" if p.is_dir() else p


def visualize_first_step_retrieved_trajs(cfg, retrieved_trajs, num_vis=100):
    all_videos = []
    all_frames = []

    # First collect all videos and frames
    for i, path in enumerate(retrieved_trajs[:num_vis]):
        path = Path(path)
        video_path = path / "external_images.dat"
        video = load_data_compressed(video_path)
        if "2d" in cfg.method or "hand" in cfg.method:
            flow_path = path / "2d_flow_query.dat"
            if flow_path.exists():
                flow = load_data_compressed(flow_path)
                if isinstance(flow, dict):
                    flow = flow["points_normalized"][:, 0]
                    flow[:, 0] *= video.shape[2]
                    flow[:, 1] *= video.shape[1]
                video = add_flow_to_video(video, flow)

        # Get keyframes
        f_i = video[0]
        f_m = video[len(video) // 2]
        f_f = video[-1]

        # Prepare video for wandb
        video = video.transpose(0, 3, 1, 2)
        all_videos.append(
            wandb.Video(
                video,
                caption=f"retrieved_subtrajectory {i}",
                fps=30,
                format="mp4",
            )
        )

        # Prepare frames for wandb
        full_image = np.concatenate((f_i, f_m, f_f), axis=1)
        all_frames.append(wandb.Image(full_image))

    return all_videos, all_frames


def visualize_retrieved_trajs(cfg, retrieved_trajs, num_vis: int = 100):
    all_videos = []
    all_frames = []

    num_vis = min(num_vis, len(retrieved_trajs))

    # Visualize first + last half
    retrieved_trajs = retrieved_trajs[: num_vis // 2] + retrieved_trajs[-num_vis // 2 :]

    # First collect all videos and frames
    for i, (path, info) in enumerate(retrieved_trajs):
        cost, start, end = info[:3]

        video_path = path / "external_images.dat"
        video = load_data_compressed(video_path)
        if "2d" in cfg.method:
            flow_path = path / "2d_flow_query.dat"
            other_flow_path = path / "2d_flow.dat"

            if flow_path.exists(): # robot
                flow = load_data_compressed(flow_path)["points_normalized"][:, 0]
                flow[start : end + 1, 0] *= video.shape[2]
                flow[start : end + 1, 1] *= video.shape[1]
            elif other_flow_path.exists(): # calvin
                flow = load_data_compressed(other_flow_path)

            video = add_flow_to_video(video, flow)

        # Get keyframes
        f_i = video[start]
        f_m = video[(start + end) // 2]
        f_f = video[-1]

        # Prepare video for wandb
        video = video[start:end]

        video = video.transpose(0, 3, 1, 2)
        all_videos.append(
            wandb.Video(
                video,
                caption=f"retrieved_subtrajectory {i} | pre-normalized cost: {cost}",
                fps=30,
                format="mp4",
            )
        )

        # Prepare frames for wandb
        full_image = np.concatenate((f_i, f_m, f_f), axis=1)
        all_frames.append(wandb.Image(full_image))

    return all_videos, all_frames


def add_flow_to_video(video, flow):
    """
    Video: [T, H, W, 3]
    Flow: [T, 2]

    Add flow as points on the video with temporal color gradient trail.
    """
    # make circle size adaptive to video size
    H, W = video.shape[1], video.shape[2]
    circle_size = max(H, W) // 100

    # Store all previous points for each frame
    for i in range(video.shape[0]):
        frame = video[i]

        # Draw all points up to current timestep
        for t in range(i + 1):
            flow_vec = flow[t]
            x, y = flow_vec[0], flow_vec[1]

            # Calculate color based on temporal distance from current frame
            # Newer points are greener, older points are more faded
            alpha = (t / i) if i > 0 else 1.0  # avoid division by zero
            green = int(255 * alpha)

            cv2.circle(
                frame,
                center=(int(x), int(y)),
                radius=circle_size,
                color=(0, green, 0),  # fading green
                thickness=-1,  # filled circle
            )

        video[i] = frame
    return video


def visualize_query(query_data, cfg):
    rgb_static = query_data["external_images"] if isinstance(query_data, dict) else query_data[1]
    flow = query_data.get("flow", query_data[7] if not isinstance(query_data, dict) else None)
    points = flow["points"].reshape(-1, 2) if isinstance(flow, dict) else flow.reshape(-1, 2)

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))

    for a in ax:
        a.axis("off")

    ax[0].imshow(rgb_static[0])
    ax[1].imshow(rgb_static[-1])

    if cfg.method == "2d":
        for a in ax:
            a.plot(
                points[:, 0],
                points[:, 1],
                c="orange",
                linewidth=6,
            )

        ax[0].scatter(points[0, 0], points[0, 1], c="red", s=40)
        ax[1].scatter(points[-1, 0], points[-1, 1], c="red", s=40)

    plt.tight_layout()

    return fig


def robust_normalize_costs(costs, min_alpha=0.1, max_alpha=1.0):
    costs = np.array(costs, dtype=np.float32)
    ranks = np.argsort(np.argsort(costs))
    normalized_ranks = ranks / (len(costs) - 1 + 1e-8)
    alphas = max_alpha - (normalized_ranks * (max_alpha - min_alpha))
    return alphas


def visualize_paths(retrieved_trajs):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    costs = [info[0] for _, info in retrieved_trajs]
    alphas = robust_normalize_costs(costs)

    for i, ((path, info), alpha) in enumerate(zip(retrieved_trajs, alphas)):
        data = load_data_compressed(_traj_path(path))
        ee_pos = data["states"][:, :3]

        color = "red" if i == 0 else "blue"
        linewidth = 3 if i == 0 else 1.5
        alpha = 1.0 if i == 0 else alpha  # ensure query is fully opaque

        ax.plot(
            ee_pos[:, 0],
            ee_pos[:, 1],
            ee_pos[:, 2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label="Query" if i == 0 else f"Traj {i}",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.grid(True)

    plt.tight_layout()

    return fig


from matplotlib.lines import Line2D


def visualize_method_paths(methods_trajs):
    idx_to_method = ["2d", "3d", "strap"]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # More distinct colors for methods
    method_colors = ["#00CC00", "#00008B", "#FF00FF"]  # Green, Dark Navy Blue, Purple
    query_color = "#FF0000"  # Standard Red

    handles = []  # For custom legend
    labels = []

    # Add query trajectory to legend first
    handles.append(Line2D([0], [0], color=query_color, lw=3))
    labels.append("Query")

    # Add lines for each method
    for method_idx, retrieved_trajs in enumerate(methods_trajs):
        method_color = method_colors[method_idx % len(method_colors)]
        method_label = idx_to_method[method_idx]

        # Add method to legend
        handles.append(Line2D([0], [0], color=method_color, lw=2))
        labels.append(method_label)

        # Limit to first 25 trajectories (or fewer if there aren't that many)
        limited_trajs = retrieved_trajs[
            : min(11, len(retrieved_trajs))
        ]  # +1 for the query

        # Normalize costs per method
        method_costs = [info[0] for _, info in limited_trajs[2:]]

        if method_costs:  # Check if there are costs to normalize
            norm = Normalize(vmin=min(method_costs), vmax=max(method_costs))

        for traj_idx, (path, info) in enumerate(limited_trajs):
            if traj_idx >= 11:  # Limit to first 25 (plus query)
                break

            data = load_data_compressed(_traj_path(path))
            ee_pos = data["states"][:, :3]

            if traj_idx == 0:
                # Query trajectory (dark red)
                color = query_color
                linewidth = 3  # Increased to 3
                traj_alpha = 1.0
            else:
                color = method_color
                linewidth = 2  # Increased to 2
                # Normalize within method with wider opacity range
                traj_alpha = max(
                    0.15, min(0.9, 1.0 - norm(info[0]))
                )  # Higher cost = lower opacity, bigger range

            num_points = len(ee_pos)

            # Plot entire trajectory as a single line
            full_trajectory = ax.plot(
                ee_pos[:, 0],
                ee_pos[:, 1],
                ee_pos[:, 2],
                color=to_rgba(color, alpha=traj_alpha),
                linewidth=linewidth,
            )[0]

            # Add a single directional arrow in the middle of the trajectory
            if num_points > 2:
                # Find middle point
                mid_idx = num_points // 2

                # Get points to define direction
                p1 = ee_pos[mid_idx - 1]
                p2 = ee_pos[mid_idx + 1]

                # Direction vector
                direction = p2 - p1
                direction = (
                    direction / np.linalg.norm(direction) * 0.1
                )  # Normalize and scale

                # Plot arrow using quiver at the middle point
                middle_point = ee_pos[mid_idx]
                ax.quiver(
                    middle_point[0],
                    middle_point[1],
                    middle_point[2],
                    direction[0],
                    direction[1],
                    direction[2],
                    color=to_rgba(color, alpha=traj_alpha),
                    arrow_length_ratio=0.3,  # Larger arrow head
                    linewidth=linewidth,
                )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.grid(True)
    ax.legend(handles, labels, loc="best")

    plt.tight_layout()

    return fig


def log_query_distribution(retrieved_trajs):
    """Log distribution of query indices to wandb."""
    query_indices = [info[3] for _, info in retrieved_trajs]
    unique_indices, counts = np.unique(query_indices, return_counts=True)

    plt.figure()
    plt.bar(unique_indices, counts)
    plt.xlabel("Query Index")
    plt.ylabel("Frequency")
    plt.title("Distribution of Retrieved Trajectories by Query Index")

    return plt

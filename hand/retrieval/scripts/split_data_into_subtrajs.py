"""
Script for converting robot data from WidowX to dat format
for retrieval experiments.

uv run hand/retrieval/scripts/split_data_into_subtrajs.py \
    env=robot \
    task=play
"""

from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from hand.data.utils import load_data_compressed, save_data_compressed
from hand.utils.logger import log


def split_into_subtrajs(qvel, accel_threshold=0.03, min_length=20, traj_dir=None):
    """
    qvel: np.array of shape [T, 6]
    accel_threshold: float — threshold on acceleration magnitude to trigger split
    min_length: int — minimum length of a subtrajectory
    """
    # Compute acceleration: [T-1, 6]
    accel = qvel[1:] - qvel[:-1]

    # Norm of acceleration per timestep: [T-1]
    accel_mag = np.linalg.norm(accel, axis=1)

    if traj_dir is not None:
        # create plot of accel
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(accel)
        ax.set_title("Acceleration")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Acceleration")
        # set ylim to [-1, 1]
        ax.set_ylim(-0.1, 0.1)
        fig.savefig(traj_dir / "accel_plot.png")

        fig = plt.figure(figsize=(10, 5))
        # create two subplots side by side
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(accel_mag)
        accel_mag_smoothed = np.convolve(accel_mag, np.ones(10) / 10, mode="valid")
        ax2.plot(accel_mag_smoothed)
        ax1.set_title("Acceleration Magnitude")
        ax2.set_title("Smoothed Acceleration Magnitude")
        ax1.set_xlabel("Timestep")
        ax1.set_ylim(-0.05, 0.5)
        ax2.set_ylim(-0.05, 0.5)
        # draw horizontal line at accel_threshold
        ax1.axhline(accel_threshold, color="r", linestyle="--")
        ax2.axhline(accel_threshold, color="r", linestyle="--")
        fig.savefig(traj_dir / "accel_mag_plots.png")

    # Find points with high acceleration
    split_indices = (
        np.where(accel_mag < accel_threshold)[0] + 1
    )  # +1 to point to the next frame

    # Optional: ensure minimum distance between splits
    final_splits = []
    last = 0
    for idx in split_indices:
        if idx - last >= min_length:
            final_splits.append(idx)
            last = idx

    # Add final segment
    if len(qvel) - last >= min_length:
        final_splits.append(len(qvel))

    # Split into subtrajectories
    subtrajs = []
    start = 0
    for end in final_splits:
        subtrajs.append((start, end))
        start = end

    return subtrajs, fig


def create_text_overlay(height, width, segment_num, total_segments, border_color):
    """Pre-create text overlay with colored border"""
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # Add border (15 pixels thick)
    border_thickness = 15
    overlay[:border_thickness, :] = border_color  # top
    overlay[-border_thickness:, :] = border_color  # bottom
    overlay[:, :border_thickness] = border_color  # left
    overlay[:, -border_thickness:] = border_color  # right

    # Add text showing segment number
    text = f"Segment {segment_num}/{total_segments}"
    cv2.putText(
        overlay,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),  # Green text
        2,
    )

    # Add progress bar
    bar_height = 5
    bar_width = int((width * segment_num) / total_segments)
    overlay[-bar_height - border_thickness :, :bar_width] = (0, 255, 0)

    return overlay


def visualize_segments(video_path, subtraj_segments):
    video = load_data_compressed(video_path)
    video = video.transpose(0, 3, 1, 2)
    if video.dtype != np.uint8:
        video = (video * 255).astype(np.uint8)

    height, width = video.shape[2:4]
    total_segments = len(subtraj_segments)
    marked_video = []

    # Define different border colors for each segment
    border_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    # Pre-create overlays
    segment_overlays = [
        create_text_overlay(
            height, width, i, total_segments, border_colors[i % len(border_colors)]
        )
        for i in range(1, total_segments + 1)
    ]

    # Process each segment
    for i, (start, end) in enumerate(subtraj_segments):
        # Get segment frames and add overlay
        segment_frames = video[start:end].transpose(0, 2, 3, 1)
        overlay = segment_overlays[i]
        marked_frames = np.where(overlay > 0, overlay, segment_frames)
        marked_video.extend(marked_frames)

        # Duplicate last frame 10 times
        if i < len(subtraj_segments) - 1:  # Don't add pause after final segment
            last_frame = marked_frames[-1]
            marked_video.extend([last_frame] * 30)

    return np.array(marked_video)


@hydra.main(config_path="../../cfg", config_name="retrieval")
def main(cfg: DictConfig):
    data_dir = Path(cfg.paths.data_dir) / "datasets" / cfg.dataset_name / cfg.task
    trajs = sorted((data_dir / "processed_trajs").glob("traj_*"))

    log(f"Data directory: {data_dir}", "green")
    log(f"Found {len(trajs)} trajectories", "green")
    log("Splitting trajectories into subtrajectories", "green")

    subtraj_idx = 0
    for i, traj_dir in tqdm(enumerate(trajs), total=len(trajs)):
        data_folders = sorted(traj_dir.glob("*.dat"))

        traj_data_file = traj_dir / "traj_data.dat"
        if traj_data_file.exists():
            traj_data = load_data_compressed(traj_data_file)
        else:
            pass

        if "hand" not in str(data_dir.name):
            qvel = traj_data["qvel"][:, :6]
            subtraj_segments, fig = split_into_subtrajs(qvel, traj_dir=traj_dir)
        else:
            image_data_file = traj_dir / "external_images.dat"
            image_data = load_data_compressed(image_data_file)

            T = image_data.shape[0]
            subtraj_segments = [
                (0, T // 3),
                (T // 3, 2 * T // 3),
                (2 * T // 3, T),
            ]

        log(
            f"Splitting {traj_dir} into {len(subtraj_segments)} subtrajectories",
            "green",
        )
        for j, (start, end) in tqdm(
            enumerate(subtraj_segments), total=len(subtraj_segments)
        ):
            subtraj_dir = data_dir / "subtraj_data" / f"subtraj_{subtraj_idx:06d}"
            subtraj_dir.mkdir(parents=True, exist_ok=True)

            for dat_file in data_folders:
                new_dat_file = subtraj_dir / dat_file.name
                if new_dat_file.exists():
                    continue

                try:
                    data = load_data_compressed(dat_file)
                except Exception as e:
                    log(f"Error loading {dat_file}: {e}", "red")
                    continue
                if data is None:
                    continue

                if isinstance(data, dict):
                    subtraj = {k: v[start:end] for k, v in data.items()}
                else:
                    subtraj = data[start:end]

                save_data_compressed(new_dat_file, subtraj)

            subtraj_idx += 1
    log(f"Saved a total of {subtraj_idx} subtrajectories", "green")


if __name__ == "__main__":
    main()

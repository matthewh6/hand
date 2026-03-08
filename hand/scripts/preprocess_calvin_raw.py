"""
Convert raw CALVIN .npz episodes into processed_trajs/ format for retrieval.

This reads the raw CALVIN dataset (individual .npz per timestep) and:
1. Segments trajectories by task (using lang annotations) and play (using episode boundaries)
2. Saves each trajectory in processed_trajs/ format with separate .dat files
3. Computes DINOv2 embeddings for visual retrieval (STRAP / visual filtering)
4. Computes 2D flow via Molmo + CoTracker for HAND

Output structure:
    data_dir/datasets/calvin/{env}/{task}/processed_trajs/traj_000000/
        traj_data.dat           # dict: {states, actions, rewards}
        external_images.dat     # (T, 200, 200, 3) uint8
        wrist_images.dat        # (T, 84, 84, 3) uint8
        external_img_embeds_dinov2_vitb14.dat  # (T, D) float32
        2d_flow_query.dat       # dict: {points_normalized: (T, 1, 2)}

Usage:
    # Process all tasks + play (embeddings, 2D)
    uv run hand/scripts/preprocess_calvin_raw.py \
        calvin_data_dir=calvin/dataset/task_D_D/training \
        task_name=all
"""

import random
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig

from hand.data.preprocess import compute_image_embeddings
from hand.data.utils import save_data_compressed, load_data_compressed
from hand.models.image_embedder import ImageEmbedder
from hand.utils.logger import log


def load_calvin_frames(calvin_data_dir: Path, start_idx: int, end_idx: int):
    """Load a sequence of CALVIN .npz frames and stack them into arrays."""
    frames = []
    for idx in range(start_idx, end_idx + 1):
        npz_file = calvin_data_dir / f"episode_{idx:07d}.npz"
        if not npz_file.exists():
            break
        frames.append(np.load(npz_file))

    if not frames:
        return None

    return {
        "rgb_static": np.stack([f["rgb_static"] for f in frames]),
        "rgb_gripper": np.stack([f["rgb_gripper"] for f in frames]),
        "actions": np.stack([f["actions"] for f in frames]),
        "rel_actions": np.stack([f["rel_actions"] for f in frames]),
        "robot_obs": np.stack([f["robot_obs"] for f in frames]),
        "scene_obs": np.stack([f["scene_obs"] for f in frames]),
    }


def get_task_segments(calvin_data_dir: Path, task_name: str):
    """Get (start, end) frame indices for a specific task from annotations."""
    ann_file = calvin_data_dir / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_file, allow_pickle=True).item()

    tasks = ann["language"]["task"]
    indices = ann["info"]["indx"]

    segments = []
    for task, (start, end) in zip(tasks, indices):
        if task == task_name:
            segments.append((start, end))

    return segments


def get_play_segments(calvin_data_dir: Path, chunk_size: int = 64):
    """Split episode data into fixed-length chunks for play data."""
    ep_ids = np.load(calvin_data_dir / "ep_start_end_ids.npy")
    segments = []
    for ep_start, ep_end in ep_ids:
        for chunk_start in range(ep_start, ep_end + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size - 1, ep_end)
            if chunk_end - chunk_start >= chunk_size // 2:
                segments.append((chunk_start, chunk_end))
    return segments


def save_trajectory(traj_dir: Path, data: dict, embedder=None):
    """Save a single trajectory in processed_trajs format."""
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Save traj_data.dat (dict format)
    traj_data_file = traj_dir / "traj_data.dat"
    if not traj_data_file.exists():
        traj_data = {
            "states": data["robot_obs"],
            "actions": data["rel_actions"],
            "rewards": np.zeros(len(data["actions"])),
        }
        save_data_compressed(traj_data_file, traj_data)

    # Save images
    img_file = traj_dir / "external_images.dat"
    if not img_file.exists():
        save_data_compressed(img_file, data["rgb_static"])

    wrist_file = traj_dir / "wrist_images.dat"
    if not wrist_file.exists():
        save_data_compressed(wrist_file, data["rgb_gripper"])

    # Compute and save DINOv2 embeddings
    embed_file = traj_dir / "external_img_embeds_dinov2_vitb14.dat"
    if not embed_file.exists() and embedder is not None:
        embeds = compute_image_embeddings(embedder, [data["rgb_static"]])[0]
        save_data_compressed(embed_file, embeds)


def get_molmo_points_for_dir(traj_base_dir: Path, molmo_model, molmo_processor, device, debug=False):
    """Phase 1: Use Molmo to find robot end-effector in first frame of each trajectory.

    Returns dict mapping traj_dir -> (x, y) pixel coordinates.
    """
    from hand.models.molmo import get_point_from_molmo

    traj_dirs = sorted(traj_base_dir.glob("traj_*"))
    query_points = {}

    for i, traj_dir in enumerate(tqdm.tqdm(traj_dirs, desc="Molmo pointing")):
        if debug and i > 5:
            break

        flow_file = traj_dir / "2d_flow_query.dat"
        if flow_file.exists():
            continue

        img_file = traj_dir / "external_images.dat"
        if not img_file.exists():
            continue

        video = load_data_compressed(img_file)  # (T, H, W, 3)
        point = get_point_from_molmo(
            molmo_model, molmo_processor, video[0],
            prompt="Point to the robot end effector",
            device=device,
        )

        if point is None:
            log(f"Molmo failed to find robot in {traj_dir}, skipping", "yellow")
            continue

        query_points[traj_dir] = point

    return query_points


def track_points_for_dir(traj_base_dir: Path, query_points: dict, cotracker, device, debug=False):
    """Phase 2: Use CoTracker to track Molmo query points through each video."""
    from hand.data.optical_flow import generate_point_tracks
    from hand.utils.general_utils import to_numpy

    for traj_dir, (x, y) in tqdm.tqdm(query_points.items(), desc="CoTracker tracking"):
        flow_file = traj_dir / "2d_flow_query.dat"
        if flow_file.exists():
            continue

        video = load_data_compressed(traj_dir / "external_images.dat")
        H, W = video.shape[1], video.shape[2]

        queries = np.array([[0, x, y]])
        pred_tracks, _ = generate_point_tracks(
            cotracker=cotracker, video=video, queries=queries, device=device,
        )

        points = to_numpy(pred_tracks[0])  # (T, N_points, 2)
        points_normalized = points.copy()
        points_normalized[..., 0] /= W
        points_normalized[..., 1] /= H

        save_data_compressed(flow_file, {"points_normalized": points_normalized})


@hydra.main(
    version_base=None, config_name="preprocess_calvin_raw", config_path="../cfg"
)
def main(cfg: DictConfig):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    calvin_data_dir = Path(cfg.calvin_data_dir)
    output_base = Path(cfg.paths.data_dir) / "datasets" / "calvin" / cfg.calvin_env

    log(f"CALVIN data dir: {calvin_data_dir}", "yellow")
    log(f"Output base: {output_base}", "yellow")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize DINOv2 embedder
    embedder = ImageEmbedder(model_name="dinov2_vitb14", device=device)
    embedder = embedder.to(device)

    # Process trajectories (images, states, embeddings)
    if cfg.task_name == "all":
        ann_file = calvin_data_dir / "lang_annotations" / "auto_lang_ann.npy"
        ann = np.load(ann_file, allow_pickle=True).item()
        all_tasks = sorted(set(ann["language"]["task"]))
        log(f"Processing {len(all_tasks)} tasks + play data", "yellow")

        for task in all_tasks:
            process_task(calvin_data_dir, output_base, task, embedder, cfg)

        process_play(calvin_data_dir, output_base, embedder, cfg)

    elif cfg.task_name == "play":
        process_play(calvin_data_dir, output_base, embedder, cfg)

    else:
        process_task(calvin_data_dir, output_base, cfg.task_name, embedder, cfg)

    # Compute 2D flow (HAND)
    log("Computing 2D flow with Molmo + CoTracker", "yellow")

    # Free embedder memory before loading Molmo + CoTracker
    del embedder
    torch.cuda.empty_cache()

    # Collect all traj directories to process
    traj_dirs_to_process = []
    if cfg.task_name == "all":
        ann_file = calvin_data_dir / "lang_annotations" / "auto_lang_ann.npy"
        ann = np.load(ann_file, allow_pickle=True).item()
        all_tasks = sorted(set(ann["language"]["task"]))
        for task in all_tasks:
            task_dir = output_base / task / "processed_trajs"
            if task_dir.exists():
                traj_dirs_to_process.append((task, task_dir))
        play_dir = output_base / "processed_trajs"
        if play_dir.exists():
            traj_dirs_to_process.append(("play", play_dir))
    elif cfg.task_name == "play":
        play_dir = output_base / "processed_trajs"
        if play_dir.exists():
            traj_dirs_to_process.append(("play", play_dir))
    else:
        task_dir = output_base / cfg.task_name / "processed_trajs"
        if task_dir.exists():
            traj_dirs_to_process.append((cfg.task_name, task_dir))

    # Phase 1: Molmo pointing (find robot end-effector in first frames)
    log("Phase 1: Molmo pointing", "yellow")
    from hand.models.molmo import load_molmo
    molmo_model, molmo_processor = load_molmo(device=device)

    all_query_points = {}
    for name, traj_base_dir in traj_dirs_to_process:
        log(f"Molmo pointing for: {name}", "green")
        points = get_molmo_points_for_dir(traj_base_dir, molmo_model, molmo_processor, device, cfg.debug)
        all_query_points.update(points)

    log(f"Molmo found {len(all_query_points)} query points", "yellow")

    # Free Molmo before loading CoTracker
    del molmo_model, molmo_processor
    torch.cuda.empty_cache()

    # Phase 2: CoTracker tracking
    log("Phase 2: CoTracker tracking", "yellow")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    cotracker = cotracker.to(device)

    for name, traj_base_dir in traj_dirs_to_process:
        dir_points = {k: v for k, v in all_query_points.items() if str(k).startswith(str(traj_base_dir))}
        if dir_points:
            log(f"CoTracker tracking for: {name} ({len(dir_points)} trajs)", "green")
            track_points_for_dir(traj_base_dir, dir_points, cotracker, device, cfg.debug)


def process_task(calvin_data_dir, output_base, task_name, embedder, cfg):
    """Process task-specific trajectories."""
    segments = get_task_segments(calvin_data_dir, task_name)
    log(f"Task '{task_name}': {len(segments)} segments", "green")

    output_dir = output_base / task_name / "processed_trajs"

    for i, (start, end) in enumerate(
        tqdm.tqdm(segments, desc=f"Processing {task_name}")
    ):
        if cfg.debug and i > 5:
            break

        traj_dir = output_dir / f"traj_{i:06d}"
        if (traj_dir / "traj_data.dat").exists():
            continue

        data = load_calvin_frames(calvin_data_dir, start, end)
        if data is None:
            log(f"Skipping segment {i}: no frames found", "yellow")
            continue

        save_trajectory(traj_dir, data, embedder)

    log(f"Saved {task_name} trajectories to {output_dir}", "green")


def process_play(calvin_data_dir, output_base, embedder, cfg):
    """Process play data (chunked episodes)."""
    chunk_size = cfg.get("play_chunk_size", 64)
    segments = get_play_segments(calvin_data_dir, chunk_size)
    log(f"Play data: {len(segments)} chunks (chunk_size={chunk_size})", "green")

    output_dir = output_base / "processed_trajs"

    for i, (start, end) in enumerate(
        tqdm.tqdm(segments, desc="Processing play data")
    ):
        if cfg.debug and i > 5:
            break

        traj_dir = output_dir / f"traj_{i:06d}"
        if (traj_dir / "traj_data.dat").exists():
            continue

        data = load_calvin_frames(calvin_data_dir, start, end)
        if data is None:
            log(f"Skipping chunk {i}: no frames found", "yellow")
            continue

        save_trajectory(traj_dir, data, embedder)

    log(f"Saved play trajectories to {output_dir}", "green")


if __name__ == "__main__":
    main()

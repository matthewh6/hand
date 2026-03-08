"""
This is how the CALVIN raw dataset is organized

data_dir/
    task/
        ann_{num}.data

The data file is organized as follows:
[index, image, state, absolute_action, relative_action, joint_action, obs, scene_obs]

Example commands:
    python3 scripts/convert_calvin_to_tfds.py \
        task_name=move_slider_left \
        dataset_name=calvin \
        precompute_embeddings=True \
        embedding_model=radio-g

    python3 scripts/convert_calvin_to_tfds.py \
        task_name=play \
        dataset_name=calvin \
        precompute_embeddings=True \
        embedding_model=radio-g
    
    python3 scripts/convert_calvin_to_tfds.py \
        task_name=move_slider_left \
        dataset_name=calvin \
        precompute_embeddings=True \
        embedding_model=radio-g \
        N=6
"""

import os
import pickle as pkl
import random
from glob import glob
from pathlib import Path

import blosc
import hydra
import numpy as np
import torch
import tqdm

from hand.data.utils import save_dataset
from hand.models.image_embedder import ImageEmbedder
from hand.utils.general_utils import omegaconf_to_dict, print_dict
from hand.utils.logger import log


def convert_play_data(cfg, embedder):
    envs = cfg.calvin_envs

    for env in envs:
        task_data_dir = Path(cfg.data_dir) / env

        traj_files = list(task_data_dir.glob("*.dat"))
        log(f"Found {len(traj_files)} trajectories in play for env {env} data")

        trajectories = create_trajectories(cfg, embedder, traj_files)
        task_name = f"{env}-play-{cfg.embedding_model}"

        save_file = (
            Path(cfg.paths.data_dir)
            / "tensorflow_datasets"
            / cfg.dataset_name
            / task_name
        )
        save_dataset(trajectories, save_file)


def convert_single_task(cfg, task, replay_buffer, embedder):
    log(f"Converting {task} to TFDS", "green")
    # search for folders in replay buffer
    task_data_dir = replay_buffer / task
    traj_files = np.array(list(task_data_dir.glob("*.dat")))

    retrieval_demos = False
    if cfg.N:
        query_indices = np.random.choice(
            range(len(traj_files)), size=cfg.N, replace=False
        )
        traj_files = traj_files[query_indices]

        retrieval_demos = True

    log(f"Found {len(traj_files)} trajectories in {task}")

    trajectories = create_trajectories(
        cfg, embedder, traj_files, retrieval_demos=retrieval_demos
    )

    task_name = f"{task}-{cfg.embedding_model}"
    save_file = (
        Path(cfg.paths.data_dir) / "tensorflow_datasets" / cfg.dataset_name / task_name
    )
    save_dataset(trajectories, save_file)


def create_trajectories(cfg, embedder, trajs, retrieval_demos=False):
    """Convert CALVIN dataset trajectories to a standardized format.

    CALVIN data structure (data list indices):
        [0] - index/timesteps
        [1] - main camera images (shape: [..., 200, ...])
        [2] - wrist camera images (shape: [..., 200, ...])
        [3] - absolute actions (shape: [..., 7])
        [4] - relative actions (shape: [..., 7])
        [5] - joint actions (shape: [..., 8])
        [6] - observations (shape: [..., 15])
        [7] - scene observations (shape: [..., 24])
    """
    trajectories = []

    # Process each trajectory
    for traj_idx, traj in tqdm.tqdm(enumerate(trajs), desc="processing trajectories"):
        if cfg.debug and traj_idx > 5:
            break

        # Load and decompress trajectory data
        try:
            with open(traj, "rb") as f:
                compressed_data = f.read()
                decompressed_data = blosc.decompress(compressed_data)
                data = pkl.loads(decompressed_data)
        except EOFError:
            print(f"EOFError: {traj}")
            continue

        # Verify data dimensions
        try:
            assert (
                data[1].shape[1] == 200  # main camera images
                and data[2].shape[1] == 200  # wrist camera images
                and data[3].shape[1] == 7  # absolute actions
                and data[4].shape[1] == 7  # relative actions
                and data[5].shape[1] == 8  # joint actions
                and data[6].shape[1] == 15  # observations
                and data[7].shape[1] == 24  # scene observations
            )
        except AssertionError:
            raise ValueError(f"Invalid data shape for trajectory {traj}")

        # Select action type based on config
        actions = {"absolute": data[3], "relative": data[4], "joint": data[5]}.get(
            cfg.action_type
        )

        if actions is None:
            raise ValueError(f"Invalid action type: {cfg.action_type}")

        # Create standardized trajectory format
        final_data = {
            "observations": data[6],
            "states": data[6],
            "actions": actions,
            "scene_obs": data[7],
            "is_last": np.zeros_like(data[0]),
            "is_first": np.zeros_like(data[0]),
            "is_terminal": np.zeros_like(data[0]),
            "discount": np.ones_like(data[0]),
            "rewards": np.zeros_like(data[0]),
        }

        # Set episode boundary markers
        final_data["is_last"][-1] = 1
        final_data["is_terminal"][-1] = 1
        final_data["is_first"][0] = 1

        # Handle image data if required
        if cfg.save_imgs:
            final_data["external_imgs"] = data[1]
            final_data["wrist_imgs"] = data[2]

            if cfg.precompute_embeddings:
                # Use existing embeddings if available (indices 10 and 11)
                if len(data) > 10:
                    final_data["external_img_embeds"] = data[10]
                    final_data["wrist_imgs_embeds"] = data[11]
                else:
                    final_data["external_img_embeds"] = embedder(data[1]).cpu().numpy()
                    final_data["wrist_imgs_embeds"] = embedder(data[2]).cpu().numpy()

        # split in half if using retrieval queries
        if retrieval_demos:
            if traj_idx % 2 == 0:
                for key in final_data.keys():
                    final_data[key] = final_data[key][: len(final_data[key]) // 2]
            else:
                for key in final_data.keys():
                    final_data[key] = final_data[key][len(final_data[key]) // 2 :]

        trajectories.append(final_data)

    return trajectories


@hydra.main(
    version_base=None, config_name="convert_calvin_to_tfds", config_path="../cfg"
)
def main(cfg):
    print_dict(omegaconf_to_dict(cfg))

    # Set some random seeds
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    replay_buffer = Path(cfg.data_dir) / cfg.calvin_env
    data_dirs = glob(str(replay_buffer) + "/*")

    # Initialize embedder if requested
    embedder = None
    if cfg.precompute_embeddings:
        embedder = ImageEmbedder(
            model_name=cfg.embedding_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            feature_map_layer=cfg.feature_map_layer,
        )

    if cfg.task_name == "all":
        all_tasks = [os.path.basename(path) for path in data_dirs]

        log(
            f"======================= Converting {len(all_tasks)} tasks to TFDS ========================"
        )

        # Convert the full dataset
        for task in tqdm.tqdm(all_tasks, desc="Tasks"):
            convert_single_task(cfg, task, replay_buffer, embedder)
    elif cfg.task_name == "play":
        convert_play_data(cfg, embedder)
    else:
        convert_single_task(cfg, cfg.task_name, replay_buffer, embedder)


if __name__ == "__main__":
    main()

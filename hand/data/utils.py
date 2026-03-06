"""Data loading and saving utilities."""

import pickle as pkl
from pathlib import Path
from typing import List, Union

import blosc
import numpy as np
import tensorflow as tf
import tqdm

from hand.utils.logger import log


def get_base_trajectory(rew: np.ndarray) -> dict:
    """Create base trajectory dict with episode boundary flags."""
    trajectory = {
        "discount": np.ones_like(rew),
        "is_last": np.zeros_like(rew),
        "is_first": np.zeros_like(rew),
        "is_terminal": np.zeros_like(rew),
    }
    trajectory["is_last"][-1] = 1
    trajectory["is_terminal"][-1] = 1
    trajectory["is_first"][0] = 1
    return trajectory


def load_data_compressed(path: Union[str, Path], verbose: bool = False):
    """Load compressed pickle data."""
    if verbose:
        log(f"Loading from {path}", "yellow")
    path = Path(path)
    with open(path, "rb") as f:
        compressed_data = f.read()
        data = pkl.loads(blosc.decompress(compressed_data))
    return data


def save_data_compressed(path: Union[str, Path], data, verbose: bool = False):
    """Save data as compressed pickle."""
    if verbose:
        log(f"Saving to {path}", "yellow")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        compressed_data = blosc.compress(pkl.dumps(data))
        f.write(compressed_data)


def save_dataset(trajectories: List[dict], save_file: Union[str, Path], save_imgs: bool = False):
    """Save trajectory data as TFDS."""
    log(f"Saving dataset to: {save_file}", "green")
    save_file = Path(save_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    def generator():
        for trajectory in trajectories:
            yield trajectory

    feature_keys = list(trajectories[0].keys())
    feature_shapes = {k: trajectories[0][k].shape[1:] for k in feature_keys}
    features_dict = {
        k: tf.TensorSpec(
            shape=(None, *feature_shapes[k]),
            dtype=trajectories[0][k].dtype,
        )
        for k in feature_keys
    }
    for k, v in features_dict.items():
        log(f"{k}: {v}", "yellow")

    trajectory_tfds = tf.data.Dataset.from_generator(
        generator, output_signature=features_dict
    )
    tf.data.Dataset.save(trajectory_tfds, str(save_file))


def raw_data_to_tfds(
    traj_dirs: List[str],
    save_file: str,
    embedding_model: str,
    resnet_feature_map_layer: str = "avgpool",
    flow_suffix: str = "all",
    segments: List[List[int]] = None,
    costs: List[float] = None,
):
    """Convert raw trajectory directories to TFDS format."""
    num_transitions = 0

    available_cameras = []
    traj_dir = traj_dirs[0]
    for dat_file in Path(traj_dir).glob("*.dat"):
        if "images" in dat_file.name and "processed" not in dat_file.name:
            available_cameras.append(dat_file.name.split("_images")[0])

    log(f"Available cameras: {available_cameras}", "yellow")

    processed_trajs = []
    for i, traj_dir in tqdm.tqdm(enumerate(traj_dirs), desc="Loading trajectories"):
        traj_dir = Path(traj_dir)
        traj_data = load_data_compressed(traj_dir / "traj_data.dat")
        traj_data["rewards"] = np.zeros(len(traj_data["actions"]))
        num_transitions += len(traj_data["actions"])

        for camera_type in available_cameras:
            if camera_type == "depth":
                continue

            images_file = traj_dir / f"{camera_type}_processed_images.dat"
            if images_file.exists():
                images = load_data_compressed(images_file)
                traj_data[f"{camera_type}_images"] = images

            if "resnet" in embedding_model:
                img_embeds_file = (
                    traj_dir
                    / f"{camera_type}_img_embeds_{embedding_model}_{resnet_feature_map_layer}.dat"
                )
            else:
                img_embeds_file = (
                    traj_dir / f"{camera_type}_img_embeds_{embedding_model}.dat"
                )
            if img_embeds_file.exists():
                img_embeds = load_data_compressed(img_embeds_file)
                traj_data[f"{camera_type}_images_embeds"] = img_embeds

        flow_file = traj_dir / f"2d_flow_{flow_suffix}.dat"
        if flow_file.exists():
            flow_data = load_data_compressed(flow_file)
            traj_data.update(flow_data)

        if costs is not None:
            traj_data["costs"] = np.full(len(traj_data["actions"]), costs[i])

        if segments is not None:
            traj_data = {
                k: v[segments[i][0] : segments[i][1]]
                for k, v in traj_data.items()
            }

        log("=" * 100)
        for k, v in traj_data.items():
            if isinstance(v, np.ndarray):
                log(f"{k}: {v.shape}")

        processed_trajs.append(traj_data)

    for idx, traj in enumerate(processed_trajs):
        base_trajectory = get_base_trajectory(traj["rewards"])
        traj = {**base_trajectory, **traj}
        processed_trajs[idx] = traj

    traj = processed_trajs[0]
    for k, v in traj.items():
        if isinstance(v, np.ndarray):
            log(f"{k}: {v.shape}")

    log(f"Total number of transitions: {num_transitions} collected", "green")
    save_dataset(processed_trajs, save_file)

import pickle as pkl
from pathlib import Path

import blosc
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig

from hand.data.utils import get_base_trajectory, load_data_compressed
from hand.retrieval.utils import TASK_TO_LANG
from hand.utils.general_utils import to_numpy
from hand.utils.logger import log


def format_to_tfds(cfg: DictConfig, retrieved_trajs: np.ndarray):
    trajectories = []
    lengths = []

    for path, metrics in tqdm.tqdm(
        retrieved_trajs,
        desc="Formatting retrieved trajectories to TFDS",
    ):
        cost, start, end = metrics

        path = Path(path)
        traj_file = path / "traj_data.dat" if path.is_dir() else path
        with open(traj_file, "rb") as f:
            traj_data = f.read()
            traj_data = blosc.decompress(traj_data)
            traj_data = pkl.loads(traj_data)

        if cfg.env.env_name == "calvin":
            if isinstance(traj_data[8], dict):
                traj_data[8] = np.array(traj_data[8]["points"]).squeeze()
            else:
                traj_data[8] = np.array(traj_data[8]).squeeze()

            for j in range(len(traj_data)):
                traj_data[j] = to_numpy(traj_data[j])[
                    start : end + 1 if end != -1 else None
                ]

        if cfg.env.env_name == "calvin":
            final_data = {
                "observations": np.array(traj_data[6]),
                "actions": np.array(traj_data[4]),
                "rewards": np.zeros_like(traj_data[0]),
                "scene_obs": np.array(traj_data[7]),
            }
        elif cfg.env.env_name == "robot":
            obs_dict = traj_data[0]
            state = obs_dict["state"]
            policy_out = traj_data[1]
            actions = policy_out["actions"]

            final_data = {
                "observations": np.array(state),
                "actions": np.array(actions),
                "rewards": np.zeros(len(actions)),
            }

        base_trajectory = get_base_trajectory(final_data["rewards"])
        final_data.update(base_trajectory)

        if cfg.env.env_name == "calvin":
            if cfg.save_imgs:
                final_data["images"] = np.array(traj_data[1])
                final_data["wrist_images"] = np.array(traj_data[2])

            final_data["external_img_embeds"] = np.array(traj_data[10])
            final_data["wrist_img_embeds"] = np.array(traj_data[11])
        elif cfg.env.env_name == "robot":
            final_data.update(traj_data[2])
            final_data.update(traj_data[3])

        if cfg.save_costs:
            final_data["costs"] = np.full(len(final_data["is_first"]), cost)

        lengths.append(end - start)
        trajectories.append(final_data)

    log("trajectory keys", color="cyan")
    log("-" * 100, color="cyan")
    for k, v in trajectories[0].items():
        log(f"{k}: {v.shape}", color="cyan")
    log("-" * 100, color="cyan")

    log(f"average lengths of retrieved trajectories {np.mean(lengths)}", color="cyan")

    return trajectories


def get_data_paths(cfg: DictConfig):
    """Get query and play data paths based on config."""
    if cfg.query_source == "expert":
        query_dir = Path(cfg.paths.data_dir) / "datasets" / cfg.dataset_name / cfg.query_env / cfg.query_task / "processed_trajs"
        log(f"Query directory: {query_dir}")
        query_paths = sorted(Path(query_dir).glob("traj_*"))
        query_paths = np.array(query_paths)

    elif cfg.query_source == "lang":
        pass

    play_paths = []
    for play_env in cfg.play_envs:
        play_dir = Path(cfg.paths.data_dir) / "datasets" / cfg.dataset_name / play_env / "processed_trajs"
        log(f"Play directory: {play_dir}")
        play_paths += sorted(Path(play_dir).glob("traj_*"))

    play_paths = np.array(play_paths)

    if cfg.query_source == "expert":
        query_indices = np.random.choice(
            range(len(query_paths)), size=cfg.N, replace=False
        )
        query_paths = query_paths[query_indices]
    elif cfg.query_source == "lang":
        import clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/32", device=device)

        langs = TASK_TO_LANG[cfg.query_task]
        with torch.no_grad():
            text = clip.tokenize(langs).to(device)
            embs = model.encode_text(text)
        query_paths = embs.cpu().detach().numpy()

    log("Number of queries and play paths", color="cyan")
    if cfg.query_source == "expert":
        for i, query_path in enumerate(query_paths):
            log(f"q{i + 1}: {query_path.stem[4:]}")
    else:
        for i, query_path in enumerate(query_paths):
            log(f"q{i + 1}: {query_path.shape}")
    log(f"len(play_paths): {len(play_paths)}")
    log("-" * 50, color="cyan")
    return query_paths, play_paths

from pathlib import Path

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig

from hand.data.utils import get_base_trajectory, load_data_compressed
from hand.retrieval.utils import TASK_TO_LANG
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
        sl = slice(start, end + 1 if end != -1 else None)

        traj_file = path / "traj_data.dat" if path.is_dir() else path
        traj_data = load_data_compressed(traj_file)

        if cfg.env.env_name == "calvin":
            final_data = {
                "observations": np.array(traj_data["states"][sl]),
                "actions": np.array(traj_data["actions"][sl]),
                "rewards": np.zeros(len(traj_data["actions"][sl])),
            }
        elif cfg.env.env_name == "robot":
            obs_dict = traj_data[0]
            state = obs_dict["state"]
            policy_out = traj_data[1]
            actions = policy_out["actions"]

            final_data = {
                "observations": np.array(state[sl]),
                "actions": np.array(actions[sl]),
                "rewards": np.zeros(len(actions[sl])),
            }

        base_trajectory = get_base_trajectory(final_data["rewards"])
        final_data.update(base_trajectory)

        if cfg.env.env_name == "calvin":
            if cfg.save_imgs:
                images = load_data_compressed(path / "external_images.dat")
                final_data["images"] = np.array(images[sl])
                wrist_file = path / "wrist_images.dat"
                if wrist_file.exists():
                    wrist_images = load_data_compressed(wrist_file)
                    final_data["wrist_images"] = np.array(wrist_images[sl])

            embed_file = path / "external_img_embeds_dinov2_vitb14.dat"
            if embed_file.exists():
                embeds = load_data_compressed(embed_file)
                final_data["external_img_embeds"] = np.array(embeds[sl])
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

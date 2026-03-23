<div align="center">

  <img src="assets/logo.png" alt="HAND Logo" height="80">

  <h1>
    HAND Me the Data: Fast Robot Adaptation via Hand Path Retrieval
  </h1>

  <p>
    <a href="https://matthewh6.github.io/">Matthew M. Hong*</a>,
    <a href="https://aliang8.github.io/">Anthony Liang*</a>,
    <a href="https://minjunkevink.github.io/">Kevin Kim</a>,
    <a href="https://scholar.google.com/citations?user=Sqo9kfgAAAAJ&hl=en">Harshitha Rajaprakash</a>,<br>
    <a href="https://jessethomason.com/">Jesse Thomason&dagger;</a>,
    <a href="https://ebiyik.github.io/">Erdem B&#305;y&#305;k&dagger;</a>,
    <a href="https://www.jessezhang.net/">Jesse Zhang&dagger;</a>
  </p>

  <p><sub>* Equal contribution &nbsp; &dagger; Equal advising</sub></p>

  <h3>ICRA 2026</h3>

  <h2>
    <a href="https://arxiv.org/abs/2505.20455">Paper</a> &nbsp; · &nbsp;
    <a href="https://liralab.usc.edu/handretrieval/">Website</a> &nbsp; · &nbsp;
    <a href="https://x.com/Jesse_Y_Zhang/status/1928248082392830292?s=20">Thread</a>
  </h2>

</div>

<img src="assets/teaser.jpg" width="95%">

</div>

# Overview

**HAND** is a *simple* and *time-efficient* method for teaching robots manipulation tasks from human hand demonstrations. It extracts hand motion to retrieve relevant robot sub-trajectories from task-agnostic play data. Retrieval data is then used to LoRA fine-tune a pre-trained policy.

# Quick Start (CALVIN)

The full pipeline: download data, preprocess, retrieve from environments A/B/C using a task demo from D, train an ACT policy, and evaluate on D.

## 1. Install

```bash
git clone --recurse-submodules https://github.com/jesbu1/p-llm-hf.git
cd p-llm-hf

uv venv --python 3.10
uv sync
source .venv/bin/activate
```

## 2. Configure local paths

Create `hand/cfg/local/default.yaml`:

```yaml
# @package _global_

paths:
  project_dir: /absolute/path/to/this/repo
  data_dir: /absolute/path/to/this/repo/data
  results_dir: /absolute/path/to/this/repo/results
  wandb_dir: /absolute/path/to/this/repo/results/wandb
  root_dir: /absolute/path/to/this/repo
```

> All paths must be **absolute**. `data_dir` is where preprocessed datasets and retrieved TFDS outputs are written.

## 3. Download CALVIN data

```bash
cd calvin/dataset

# Download D (query demonstrations + evaluation)
sh download_data.sh D

# Download ABC (play data to retrieve from)
sh download_data.sh ABC

cd ../..
```

This creates `calvin/dataset/task_D_D/` and `calvin/dataset/task_ABC_D/`.

## 4. Preprocess

Convert raw CALVIN `.npz` episodes into `processed_trajs/` format. This segments trajectories, saves images/states/actions, computes DINOv2 embeddings, and runs Molmo + CoTracker for 2D flow tracking.

```bash
# Preprocess D (query tasks + play data)
uv run hand/scripts/preprocess_calvin_raw.py \
    calvin_data_dir=calvin/dataset/task_D_D/training \
    calvin_env=D+0 \
    task_name=all

# Preprocess ABC play data (all three environments as one pool)
uv run hand/scripts/preprocess_calvin_raw.py \
    calvin_data_dir=calvin/dataset/task_ABC_D/training \
    calvin_env=ABC+0 \
    task_name=play
```

Output: `data/datasets/calvin/D+0/{task}/processed_trajs/` (query) and `data/datasets/calvin/ABC+0/processed_trajs/` (play).

## 5. Retrieve from ABC using D queries

Retrieve sub-trajectories from ABC play data that match a task demonstration from D:

```bash
# HAND retrieval (visual filtering + 2D path matching)
uv run hand/retrieval/retrieval_calvin.py \
    query_task=move_slider_left \
    method=hand \
    query_env=D+0 \
    play_envs='[ABC+0]' \
    K=250 \
    K2=100 \
    save_dataset=True
```

Other retrieval methods:

```bash
# STRAP retrieval (DINOv2 embedding similarity)
uv run hand/retrieval/retrieval_calvin.py \
    query_task=move_slider_left \
    method=strap \
    query_env=D+0 \
    play_envs='[ABC+0]' \
    K=100 \
    save_dataset=True

# 3D retrieval (end-effector position matching)
uv run hand/retrieval/retrieval_calvin.py \
    query_task=move_slider_left \
    method=3d \
    query_env=D+0 \
    play_envs='[ABC+0]' \
    K=100 \
    save_dataset=True
```

Retrieved datasets are saved to `data/tensorflow_datasets/`.

## 6. Train ACT policy

Train an Action Chunking Transformer policy on the retrieved dataset:

```bash
uv run hand/scripts/train_act.py \
    dataset_path=data/tensorflow_datasets/retrieval_with_expert_two_step/calvin/hand/move-slider-left_N-6_K-250
```

Checkpoints are saved to `results/checkpoints/`.

<details>
<summary><b>Training parameters</b></summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_path` | required | Path to saved TFDS dataset |
| `batch_size` | `64` | Training batch size |
| `lr` | `1e-4` | Learning rate |
| `num_epochs` | `100` | Number of training epochs |
| `chunk_size` | `20` | Action chunk size |
| `d_model` | `256` | Transformer hidden dimension |
| `num_layers` | `4` | Transformer decoder layers |
| `image_encoder` | `dinov2_vitb14` | Frozen image encoder |

</details>

## 7. Evaluate on CALVIN D

Run the trained policy on the standard CALVIN multi-step evaluation benchmark (1000 sequences of 5 chained tasks):

```bash
uv run hand/scripts/eval_calvin.py \
    --checkpoint results/checkpoints/move-slider-left_N-6_K-250/best.pt \
    --dataset_path calvin/dataset/task_D_D
```

This reports success rates for completing 1-5 tasks in a row.

<details>
<summary><b>All retrieval parameters</b></summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query_task` | `close_drawer` | CALVIN task name (e.g., `move_slider_left`, `open_drawer`) |
| `method` | `hand` | `strap`, `hand`, `3d`, `2d`, `2d_abs`, `hand_abs` |
| `N` | `6` | Number of query trajectories |
| `K` | `100` | Retrieved sub-trajectories (for HAND: first-step filter size) |
| `K2` | `100` | Second-step retrieval size (HAND only) |
| `query_env` | `D+0` | Environment split for query trajectories |
| `play_envs` | `[D+0]` | Environment splits to search over |
| `with_expert` | `True` | Include query trajectories in retrieved set |
| `save_dataset` | `False` | Save as TensorFlow dataset |
| `use_wandb` | `False` | Log videos/plots to W&B |

</details>

# Real World (Robot)

<details>
<summary><b>Click to expand</b></summary>

### Data Preprocessing

Split trajectories into subtrajectories before retrieval:

```bash
uv run hand/retrieval/scripts/split_data_into_subtrajs.py \
    dataset_name=robot \
    task=play
```

### Optional: Grounded-SAM-2

```bash
uv pip install -e Grounded-SAM-2
```

### HAND Retrieval

```bash
uv run hand/retrieval/retrieval.py \
    env=robot \
    query_task=keurig \
    other_tasks=[play] \
    query_source=expert \
    method=hand \
    N=1 \
    K=250 \
    K2=100 \
    with_expert=False \
    query_files=[/path/to/hand/demo/1]
```

### STRAP Retrieval

```bash
uv run hand/retrieval/retrieval.py \
    env=robot \
    query_task=keurig \
    other_tasks=[play] \
    query_source=expert \
    method=strap \
    N=2 \
    K=100 \
    with_expert=True \
    query_files=[/path/to/expert/demo/1,/path/to/expert/demo/2]
```

</details>

# Citation

```bibtex
@article{hong2025handdatafastrobot,
      title={HAND Me the Data: Fast Robot Adaptation via Hand Path Retrieval},
      author={Matthew M Hong and Anthony Liang and Kevin Kim and Harshitha Rajaprakash and Jesse Thomason and Erdem Bıyık and Jesse Zhang},
      year={2025},
      eprint={2505.20455},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      journal={arXiv preprint arxiv:2505.20455},
}
```

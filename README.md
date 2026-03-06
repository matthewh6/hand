<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <div>
        <img src="assets/logo.png" alt="HAND Logo" style="height:5em;">
      </div>
      <h1>
        HAND Me the Data: Fast Robot Adaptation via Hand Path Retrieval
      </h1>
      <h2>
        <a href="https://arxiv.org/abs/2505.20455">Paper</a> &emsp;
        <a href="https://liralab.usc.edu/handretrieval/">Website</a> &emsp;
        <a href="https://x.com/Jesse_Y_Zhang/status/1928248082392830292?s=20">Thread</a>
      </h2>
    </summary>
  </ul>
</div>

<img src="assets/teaser.jpg" width="95%">

</div>

# Overview

**HAND** is a *simple* and *time-efficient* method for teaching robots manipulation tasks from human hand demonstrations. It extracts hand motion to retrieve relevant robot sub-trajectories from task-agnostic play data. The retrieved datasets can be used with any behavior cloning (BC) framework for policy training and LoRA fine-tuning.

# Quick Start

```bash
git clone --recurse-submodules https://github.com/jesbu1/p-llm-hf.git
cd p-llm-hf

uv venv --python 3.10
uv sync
source .venv/bin/activate
```

This installs `hand`, `cotracker`, and all dependencies in one `.venv`. Submodules (`co-tracker`, `Grounded-SAM-2`, `calvin`) are cloned automatically. **No robot_learning dependency**—hand is fully standalone.

### Optional: Grounded-SAM-2 (for HAND retrieval on real robot)

```bash
uv pip install -e Grounded-SAM-2
```

### Optional: CALVIN (for CALVIN experiments)

```bash
cd calvin && git submodule update --init --recursive && cd ..
export CALVIN_ROOT=$(pwd)/calvin
uv pip install -e calvin/calvin_env
uv pip install -e calvin/calvin_env/tacto
```

## Local Config

Create a file `hand/cfg/local/default.yaml` with paths for your machine:

```yaml
# @package _global_

paths:
  project_dir: /path/to/project
  data_dir: /path/to/data
  results_dir: /path/to/results
  wandb_dir: /path/to/wandb
```

# Play Data

## CALVIN

1. **Download** the CALVIN dataset from the [official repo](https://github.com/mees/calvin):

```bash
cd $CALVIN_ROOT/dataset
sh download_data.sh D      # or ABC, ABCD, or debug (1.3 GB)
```

2. **Convert** to the format expected by retrieval (with precomputed embeddings):

```bash
uv run hand/scripts/convert_calvin_to_tfds.py \
    data_dir=/path/to/calvin/task_D_D \
    env=D+0 \
    task_name=move_slider_left \
    precompute_embeddings=True \
    embedding_model=radio-g

uv run hand/scripts/convert_calvin_to_tfds.py \
    data_dir=/path/to/calvin/task_D_D \
    env=D+0 \
    task_name=play \
    precompute_embeddings=True \
    embedding_model=radio-g
```

3. Set `paths.data_dir` in `hand/cfg/local/default.yaml` to the directory containing `datasets/` and `tensorflow_datasets/`.

   Retrieval expects `data_dir/datasets/calvin/{env}/{task}/processed_trajs/` (e.g. `D+0/move_slider_left`, `A+0/play`). The convert script writes to `tensorflow_datasets/` for training. For retrieval, ensure your data is in the `processed_trajs` layout; see `hand/scripts/preprocess_calvin_data.py` for the expected format.

# Experiments

## Retrieval

### CALVIN

<details>
<summary><b>Click to expand the full list of commands</b></summary>

#### Two-Step Retrieval (HAND)

```bash
uv run hand/retrieval/retrieval_calvin.py \
    query_task=move_slider_left \
    query_source=expert \
    method=hand \
    K=1000 \
    K2=100 \
    save_dataset=True
```

#### One-Step Retrieval (STRAP)

```bash
uv run hand/retrieval/retrieval_calvin.py \
    query_task=move_slider_left \
    query_source=expert \
    method=strap \
    K=100 \
    save_dataset=True
```

</details>

### Real World (Robot)

<details>
<summary><b>Click to expand the full list of commands</b></summary>

#### STRAP Retrieval

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

#### HAND Retrieval

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

</details>

## Data Preprocessing (Real World)

Before running retrieval on real robot data, split trajectories into subtrajectories:

```bash
uv run hand/retrieval/scripts/split_data_into_subtrajs.py \
    dataset_name=robot \
    task=play
```

## Retrieval Pipeline (Real World)

For batch retrieval over multiple parameter combinations:

```bash
uv run hand/retrieval/scripts/pipeline.py \
    env=robot \
    query_task=keurig \
    other_tasks=[play] \
    ...
```

Requires preprocessed data (traj_data.dat, embeddings) in `data_dir/datasets/{env}/{task}/`. The pipeline runs retrieval and produces TFDS datasets; training is done with your preferred BC framework.

## Policy Training

HAND focuses on **retrieval**. The retrieved TFDS datasets can be used with any BC framework (e.g., [LeRobot](https://github.com/huggingface/lerobot), [ACT](https://github.com/tonyzhaozh/act)) for base policy training and LoRA fine-tuning. See the [paper](https://arxiv.org/abs/2505.20455) for training details.

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
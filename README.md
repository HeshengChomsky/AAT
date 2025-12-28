# environment

## Installation

Build the environment

git clone https://github.com/kristery/Elastic-DT.git
cd Elastic-DT
pip install requirist.txt

If the installation of dee and gfootball fails, please refer to [Gfootball](https://github.com/google-research/footb and [dm_control](https://github.com/google-deepmind/dm_control)

## Atari

## dm_control
Google DeepMind's software stack for physics-based simulation and Reinforcement Learning environments, using MuJoCo physics.

## Gfootball

This repository contains an RL environment based on open-source game Gameplay Football.
It was created by the Google Brain team for research purposes (https://github.com/google-research/football)





```
conda env create -f conda_env.yml
```

## Downloading datasets

Create a directory for the dataset and load the dataset using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)
```
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

## Example usage

```
python run_dt_atari.py --seed 123 --block_size 90 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 --data_dir_prefix [DIRECTORY_NAME]
```

## Ancknowledge
The implementation of AAT is based on [DT](https://github.com/kzl/decision-transformer).

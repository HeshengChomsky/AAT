# environment

## Installation

Build the environment
```
git clone https://github.com/HeshengChomsky/AAT.git
cd AAT
pip install -e.
```

If the installation of dm_control and gfootball fails, please refer to [Gfootball](https://github.com/google-research/football) and [dm_control](https://github.com/google-deepmind/dm_control)


## create training datasets

Run create_training_data.py to get the AAT training data
```
python create_training_data.py
```

## training AAT

```
python run_aat.py --seed 123 --block_size 90 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 128 --data_dir_prefix [DIRECTORY_NAME]
```

## evaluation AAT
```
python evaluation_aat.py --policy_path [DIRECTORY_NAME] --epochs 10
```

## Ancknowledge
The implementation of AAT is based on [DT](https://github.com/kzl/decision-transformer).

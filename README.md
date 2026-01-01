# environment

## Installation

Build the environment
```
git clone https://github.com/HeshengChomsky/AAT.git
cd AAT
conda env create -f environment.yml
```

If the installation of dm_control and gfootball fails, please refer to [Gfootball](https://github.com/google-research/football) and [dm_control](https://github.com/google-deepmind/dm_control)


## create training datasets
You can  download the training data from the cloud disk to train the AAT:[Datasets](https://huggingface.co/datasets/tianleh/training_datasets/tree/main)
or run the following script.
```
python create_training_data.py
```

## training AAT

```
python run_aat.py --seed 123 --block_size 90 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 128 --data_dir_prefix [DIRECTORY_NAME]
```

## evaluation AAT
Please download the [model parameters](https://huggingface.co/tianleh/aat/tree/main) and put them in the models file. Or run the following script directly, it will automatically load the model parameters from huggingface.
```
python evaluation_aat.py --epochs 10
```

## Ancknowledge
The implementation of AAT is based on [DT](https://github.com/kzl/decision-transformer).










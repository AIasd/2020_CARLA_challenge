# Learning by Cheating

![teaser](https://github.com/dianchen96/LearningByCheating/blob/release-0.9.6/figs/fig1.png "Pipeline")
> [**Learning by Cheating**](https://arxiv.org/abs/1912.12294)    
> Dian Chen, Brady Zhou, Vladlen Koltun, Philipp Kr&auml;henb&uuml;hl,        
> [Conference on Robot Learning](https://www.robot-learning.org) (CoRL 2019)      
> _arXiv 1912.12294_

If you find our repo to be useful in your research, please consider citing our work
```
@inproceedings{chen2019lbc
  author    = {Dian Chen and Brady Zhou and Vladlen Koltun and Philipp Kr\"ahenb\"uhl},
  title     = {Learning by Cheating},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2019},
}
```

## Installation

Clone this repo with all its submodules

```bash
git clone https://github.com/bradyz/2020_CARLA_challenge.git --recursive
```

All python packages used are specified in `carla_project/requirements.txt`.

This code uses CARLA 0.9.9 and works with CARLA 0.9.8.

You will also need to install CARLA 0.9.9, along with the additional maps.
See [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) for more instructions.



## Installation of Carla 0.9.9
The following commands can be used to install carla 0.9.9

Create a new conda environment:
```
conda create --name carla99 python=3.7
conda activate carla99
```
Download CARLA_0.9.9.tar.gz and AdditionalMaps_0.9.9.tar.gz from [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) and run
```
mkdir carla_099
tar -xvzf CARLA_0.9.9.tar.gz -C carla_099
```
unzip AdditionalMaps_0.9.9.tar.gz and merge the extracted two folders with the content inside carla_099.
Then, run
```
cd carla_099/PythonAPI/carla/dist
easy_install carla-0.9.9-py3.7-linux-x86_64.egg
```
Test the installation by running
```
cd ../../..
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```



## Dataset

We provide a dataset of over 70k samples collected over the 75 routes provided in `leaderboard/data/routes_*.xml`.

[Link to full dataset (9 GB)](https://drive.google.com/file/d/1dwt9_EvXB1a6ihlMVMyYx0Bw0mN27SLy/view?usp=sharing).

![sample](assets/sample_route.gif)

The dataset is collected using `scenario_runner/team_code/autopilot.py`, using painfully hand-designed rules (i.e. if pedestrian is 5 meters ahead, then brake).

Additionally, we change the weather for a single route once every couple of seconds to add visual diversity as a sort of on-the-fly augmentation.
The simulator is run at 20 FPS, and we save the following data at 2 Hz.

* Left, Center, and Right RGB Images at 256 x 144 resolution
* A semantic segmentation rendered in the overhead view
* World position and heading
* Raw control (steer, throttle, brake)

Note: the overhead view does nothing to address obstructions, like overhead highways, etc.

We provide a sample trajectory in `sample_data`, which you can visualize by running

```
python3 -m carla_project.src.dataset sample_data/route_00/
```

## Data Collection

The autopilot that we used to collect the data can use a lot of work and currently does not support stop signs.

If you're interested in recollecting data after changing the autopilot's driving behavior in `scenario_runner/team_code/autopilot.py`, you can collect your own dataset by running the following.

First, spin up a CARLA server

```bash
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```

then run the agent.

```bash
./run_agent.sh
```

## Run a pretrained model

Download the checkpoint from our [Wandb project](https://app.wandb.ai/bradyz/2020_carla_challenge_lbc).

Navigate to one of the runs, like https://app.wandb.ai/bradyz/2020_carla_challenge_lbc/runs/command_coefficient=0.01_sample_by=even_stage2/files

Go to the "files" tab, and download the model weights, named "epoch=24.ckpt", and pass in the file path as the `TEAM_CONFIG` below.

Spin up a CARLA server

```bash
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```

then run the agent.

```bash
./run_agent.sh
```

## Training models from scratch

First, download and extract our provided dataset.

Then run the stage 1 training of the privileged agent.

```python
python3 -m carla_project/src/map_model --dataset_dir /path/to/data
```

We use wandb for logging, so navigate to the generated experiment page to visualize training.

![sample](assets/stage_1.gif)

If you're interested in tuning hyperparameters, see `carla_project/src/map_model.py` for more detail.

Training the sensorimotor agent (acts only on raw images) is similar, and can be done by

```python
python3 -m carla_project/src/image_model --dataset_dir /path/to/data
```

## Docker

Build the docker container to submit, make sure to edit `scripts/Dockerfile.master` appropriately.

```bash
sudo ./scripts/make_docker.sh
```

Spin up a CARLA server

```bash
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```

Now you can either run the docker container or run it interactively.

To run the docker container,

```bash
sudo docker run --net=host --gpus all -e NVIDIA_VISIBLE_DEVICES=0 -e REPETITIONS=1 -e DEBUG_CHALLENGE=0 -e PORT=2000 -e ROUTES=leaderboard/data/routes_devtest.xml -e CHECKPOINT_ENDPOINT=tmp.txt -e SCENARIOS=leaderboard/data/all_towns_traffic_scenarios_public.json leaderboard-user:latest ./leaderboard/scripts/run_evaluation.sh
```

Or if you need to debug something, you can run it interactively

```bash
sudo docker run --net=host --gpus all -it leaderboard-user:latest /bin/bash
```

Run the evaluation through the interactive shell.

```bash
export PORT=2000
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export ROUTES=leaderboard/data/routes_devtest.xml
export CHECKPOINT_ENDPOINT=tmp.txt
export SCENARIOS=leaderboard/data/all_towns_traffic_scenarios_public.json

conda activate python37

./leaderboard/scripts/run_evaluation.sh
```

## Installation

Clone this repo with all its submodules

```
git clone https://github.com/AIasd/2020_CARLA_challenge.git --recursive
```

All python packages used are specified in `carla_project/requirements.txt`.

This code uses CARLA 0.9.9.

You will also need to install CARLA 0.9.9, along with the additional maps.
See [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) for more instructions.



## Installation of Carla 0.9.9
The following commands can be used to install carla 0.9.9

Create a new conda environment:
```
conda create --name carla99 python=3.7
conda activate carla99
```
Download CARLA_0.9.9.4.tar.gz and AdditionalMaps_0.9.9.4.tar.gz from [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) and run
```
mkdir carla_0994_no_rss
tar -xvzf CARLA_0.9.9.4.tar.gz -C carla_0994_no_rss
```
move `AdditionalMaps_0.9.9.4.tar.gz` to `carla_0994_no_rss/Import/` and in the folder `carla_0994_no_rss/` run:
```
./ImportAssets.sh
```
Then, run
```
cd carla_0994_no_rss/PythonAPI/carla/dist
easy_install carla-0.9.9-py3.7-linux-x86_64.egg
```
Test the installation by running
```
cd ../../..
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```
A window should pop up.


## Run an autopilot model
comment out the pretrained model lines in run_agent.sh and modify the paths inside.

Spin up a CARLA server

```
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```

then run the agent in a separate window.

```
./run_agent.sh
```

## Run a pretrained model

Download the checkpoint from [Wandb project](https://app.wandb.ai/bradyz/2020_carla_challenge_lbc).

Navigate to one of the runs, like https://app.wandb.ai/bradyz/2020_carla_challenge_lbc/runs/command_coefficient=0.01_sample_by=even_stage2/files

Go to the "files" tab, and download the model weights, named "epoch=24.ckpt", and pass in the file path as the `TEAM_CONFIG` in `run_agent.sh`. comment out the autopilot model lines in `run_agent.sh` and modify the paths inside.

Spin up a CARLA server

```
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```

then run the agent in a separate window.

```
./run_agent.sh
```

## Fuzzing
Spin up a CARLA server

```
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```

then run the agent in a separate window.

```
./run_fuzzing.sh
```

## Genetic Fuzzing (Searching for error scenario)
Spin up a CARLA server

```
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```

then run the agent in a separate window.

```
python ga_fuzzing.py
```


# Reference
This repo is modified from [here](https://github.com/bradyz/2020_CARLA_challenge)

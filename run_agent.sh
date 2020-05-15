#!/bin/bash
# modification: pre-trained model
export CARLA_ROOT=/home/zhongzzy9/Documents/self-driving-car/carla_099
export PORT=2000
export ROUTES=leaderboard/data/routes/route_19.xml
export TEAM_AGENT=scenario_runner/team_code/image_agent.py
export TEAM_CONFIG=/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/models/epoch=24.ckpt
export HAS_DISPLAY=1
export PYTHONPATH=$PYTHONPATH:.

# modification: data collection
# export CARLA_ROOT=/home/zhongzzy9/Documents/self-driving-car/carla_099
# export PORT=2000
# export ROUTES=leaderboard/data/routes/route_19.xml
# export TEAM_AGENT=scenario_runner/team_code/auto_pilot.py
# export TEAM_CONFIG=sample_data
# export HAS_DISPLAY=1
# export PYTHONPATH=$PYTHONPATH:.


export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg           # 0.9.8
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg           # 0.9.8
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner




if [ -d "$TEAM_CONFIG" ]; then
    CHECKPOINT_ENDPOINT="$TEAM_CONFIG/$(basename $ROUTES .xml).txt"
else
    CHECKPOINT_ENDPOINT="$(dirname $TEAM_CONFIG)/$(basename $ROUTES .xml).txt"
fi

python leaderboard/leaderboard/leaderboard_evaluator.py \
--challenge-mode \
--track=dev_track_3 \
--scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTES} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--port=${PORT}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

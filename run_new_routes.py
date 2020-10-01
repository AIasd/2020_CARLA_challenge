import sys
import os
sys.path.append('pymoo')
carla_root = '../carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')
sys.path.append('.')
sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')
sys.path.append('scenario_runner')
sys.path.append('carla_project/src')




from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.duplicate import ElementwiseDuplicateElimination, NoDuplicateElimination

from pymoo.model.population import Population
from pymoo.model.evaluator import Evaluator

from pymoo.algorithms.nsga2 import NSGA2, binary_tournament
from pymoo.algorithms.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.algorithms.random import RandomAlgorithm

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover

from pymoo.performance_indicator.hv import Hypervolume

import matplotlib.pyplot as plt

from object_types import WEATHERS, pedestrian_types, vehicle_types, static_types, vehicle_colors, car_types, motorcycle_types, cyclist_types

from customized_utils import create_transform, rand_real,  convert_x_to_customized_data, make_hierarchical_dir, exit_handler, arguments_info, is_critical_region, setup_bounds_mask_labels_distributions_stage1, setup_bounds_mask_labels_distributions_stage2, customize_parameters, customized_bounds_and_distributions, static_general_labels, pedestrian_general_labels, vehicle_general_labels, waypoint_labels, waypoints_num_limit, if_violate_constraints, customized_routes, get_distinct_data_points, is_similar, check_bug, is_distinct, filter_critical_regions, parse_scenario, parse_route_file, estimate_objectives, parse_route_and_scenario_plain


from collections import deque


import numpy as np
import carla

from leaderboard.fuzzing import LeaderboardEvaluator
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.customized.object_params import Static, Pedestrian, Vehicle

import traceback
import json
import re
import time
from datetime import datetime

import pathlib
import shutil
import dill as pickle
# import pickle
import argparse
import atexit
import traceback
import math



import copy

from pymoo.factory import get_termination
from pymoo.model.termination import Termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from pymoo.model.individual import Individual
from pymoo.model.repair import Repair
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_crossover, get_mutation
from pymoo.model.mating import Mating

from dask.distributed import Client, LocalCluster

from pymoo.model.initialization import Initialization
from pymoo.model.duplicate import NoDuplicateElimination
from pymoo.model.individual import Individual
from pymoo.operators.sampling.random_sampling import FloatRandomSampling









parser = argparse.ArgumentParser()
parser.add_argument('-p','--ports', type=int, default=2003, help='TCP port to listen to (default: 2003)')
parser.add_argument("-r", "--route_type", type=str, default='town05_right_0')
parser.add_argument("-c", "--scenario_type", type=str, default='default')

parser.add_argument("-m", "--ego_car_model", type=str, default='lbc')
parser.add_argument("--has_display", type=str, default='0')
parser.add_argument("--root_folder", type=str, default='run_results')

parser.add_argument("--episode_max_time", type=int, default=50)

arguments = parser.parse_args()


ports = arguments.ports


# ['none', 'town01_left_0', 'town07_front_0', 'town05_front_0', 'town05_right_0']
route_type = arguments.route_type
# ['default', 'leading_car_braking', 'vehicles_only', 'no_static']
scenario_type = arguments.scenario_type
# ['lbc', 'auto_pilot', 'pid_agent']
ego_car_model = arguments.ego_car_model

os.environ['HAS_DISPLAY'] = arguments.has_display
root_folder = arguments.root_folder


episode_max_time = arguments.episode_max_time




random_seed = 0
rng = np.random.default_rng(random_seed)

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

scenario_folder = 'scenario_files'
if not os.path.exists('scenario_files'):
    os.mkdir(scenario_folder)
scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

# This is used to control how this program use GPU
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
default_objectives = [0, 7, 7, 7, 0, 0, 0, 0]




def run_simulation(customized_data, launch_server, episode_max_time, route_path, route_str, scenario_file, ego_car_model):
    arguments = arguments_info()
    arguments.port = customized_data['port']

    # model path and checkpoint path
    if ego_car_model == 'lbc':
        arguments.agent = 'scenario_runner/team_code/image_agent.py'
        arguments.agent_config = 'models/epoch=24.ckpt'
        base_save_folder = 'collected_data_customized'
    elif ego_car_model == 'auto_pilot':
        arguments.agent = 'leaderboard/team_code/auto_pilot.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_autopilot'
    elif ego_car_model == 'pid_agent':
        arguments.agent = 'scenario_runner/team_code/pid_agent.py'
        arguments.agent_config = ''
        base_save_folder = 'collected_data_pid_agent'
    elif ego_car_model == 'map_model':
        arguments.agent = 'scenario_runner/team_code/map_agent.py'
        arguments.agent_config = 'models/stage1_default_50_epoch=16.ckpt'
        base_save_folder = 'collected_data_map_model'
    else:
        print('unknown ego_car_model:', ego_car_model)




    statistics_manager = StatisticsManager()
    # Fixed Hyperparameters
    sample_factor = 5



    # Laundry Stuff-------------------------------------------------------------
    arguments.weather_index = customized_data['weather_index']
    os.environ['WEATHER_INDEX'] = str(customized_data['weather_index'])


    # used to read scenario file
    arguments.scenarios = scenario_file

    # used to compose folder to save real-time data
    os.environ['SAVE_FOLDER'] = arguments.save_folder

    # used to read route to run; used to compose folder to save real-time data
    arguments.routes = route_path
    os.environ['ROUTES'] = arguments.routes

    # used to record real time deviation data
    arguments.deviations_folder = arguments.save_folder + '/' + pathlib.Path(os.environ['ROUTES']).stem

    # used to read real-time data
    save_path = arguments.save_folder + '/' + pathlib.Path(os.environ['ROUTES']).stem






    # extract waypoints along route
    import xml.etree.ElementTree as ET
    tree = ET.parse(arguments.routes)
    route_waypoints = []


    for route in tree.iter("route"):
        for waypoint in route.iter('waypoint'):
            route_waypoints.append(create_transform(float(waypoint.attrib['x']), float(waypoint.attrib['y']), float(waypoint.attrib['z']), float(waypoint.attrib['pitch']), float(waypoint.attrib['yaw']), float(waypoint.attrib['roll'])))

    # --------------------------------------------------------------------------

    customized_data['using_customized_route_and_scenario'] = True
    customized_data['destination'] = route_waypoints[-1].location
    customized_data['sample_factor'] = sample_factor
    customized_data['number_of_attempts_to_request_actor'] = 10

    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager, launch_server, episode_max_time)
        leaderboard_evaluator.run(arguments, customized_data)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator
        # collect signals for estimating objectives

        objectives, loc, object_type = estimate_objectives(save_path, default_objectives)

    return objectives, loc, object_type, save_path







def sample_within_bounds(xl, xu, mask, labels, parameters_distributions, customized_constraints):
    max_sample_times = 100
    sample_times = 0
    while sample_times < max_sample_times:
        sample_times += 1
        x = []
        for i, dist in enumerate(parameters_distributions):
            typ = mask[i]
            lower = xl[i]
            upper = xu[i]
            assert lower <= upper, problem.labels[i]+','+str(lower)+'>'+str(upper)
            label = labels[i]
            if typ == 'int':
                val = rng.integers(lower, upper+1)
            elif typ == 'real':
                if dist[0] == 'normal':
                    if dist[1] == None:
                        mean = (lower+upper)/2
                    else:
                        mean = dist[1]
                    val = rng.normal(mean, dist[2], 1)[0]
                else: # default is uniform
                    val = rand_real(rng, lower, upper)
                val = np.clip(val, lower, upper)
            x.append(val)
        if not if_violate_constraints(x, customized_constraints, labels):
            x = np.array(x).astype(float)
            break
    return x



def get_bounds(customized_config, use_fine_grained_weather):
    customized_parameters_bounds = customized_config['customized_parameters_bounds']
    customized_parameters_distributions = customized_config['customized_parameters_distributions']
    customized_center_transforms = customized_config['customized_center_transforms']
    customized_constraints = customized_config['customized_constraints']

    fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels = setup_bounds_mask_labels_distributions_stage1(use_fine_grained_weather)
    customize_parameters(parameters_min_bounds, customized_parameters_bounds)
    customize_parameters(parameters_max_bounds, customized_parameters_bounds)


    fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels, parameters_distributions, n_var = setup_bounds_mask_labels_distributions_stage2(fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels)
    customize_parameters(parameters_min_bounds, customized_parameters_bounds)
    customize_parameters(parameters_max_bounds, customized_parameters_bounds)
    customize_parameters(parameters_distributions, customized_parameters_distributions)



    xl = [pair[1] for pair in parameters_min_bounds.items()]
    xu = [pair[1] for pair in parameters_max_bounds.items()]


    return xl, xu, mask, labels, parameters_distributions, parameters_min_bounds, parameters_max_bounds, customized_center_transforms, customized_constraints





def get_customized_data(port, scenario_type, route_type):
    if route_type == 'none':
        use_fine_grained_weather = False
    else:
        use_fine_grained_weather = True
    customized_d = customized_bounds_and_distributions[scenario_type]
    xl, xu, mask, labels, parameters_distributions, parameters_min_bounds, parameters_max_bounds, customized_center_transforms, customized_constraints = get_bounds(customized_d, use_fine_grained_weather)
    x = sample_within_bounds(xl, xu, mask, labels, parameters_distributions, customized_constraints)


    num_of_static_max = parameters_max_bounds['num_of_static_max']
    num_of_pedestrians_max = parameters_max_bounds['num_of_pedestrians_max']
    num_of_vehicles_max = parameters_max_bounds['num_of_vehicles_max']


    x = np.append(x, port)

    customized_data = convert_x_to_customized_data(x, waypoints_num_limit, num_of_static_max, num_of_pedestrians_max, num_of_vehicles_max, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms, parameters_min_bounds, parameters_max_bounds)

    return customized_data




def run_one_route(port, scenario_type, route_type):
    launch_server = True
    episode_max_time = 50

    route_info = customized_routes[route_type]
    town_name = route_info['town_name']
    route = route_info['route_id']
    location_list = route_info['location_list']

    route_str = str(route)
    if route < 10:
        route_str = '0'+route_str

    route_path = parse_route_and_scenario_plain(location_list, town_name, route_str, scenario_file)

    customized_data = get_customized_data(port, scenario_type, route_type)

    objectives, _, _, _ = run_simulation(customized_data, launch_server, episode_max_time, route_path, route_str, scenario_file, ego_car_model)


def run_multiple_short_routes(port, scenario_type, route_type):

    launch_server = True
    episode_max_time = 50


    route_folder = 'leaderboard/data/new_routes'


    route_files = os.listdir(route_folder)
    for route_file in route_files:
        print(route_file)
        if not route_file.endswith('xml'):
            continue

        route_path = os.path.join(route_folder, route_file)
        route_str = route_file[6:-4]



        route_id, town_name, transform_list = parse_route_file(route_path)[0]
        x_0, y_0 = transform_list[0][:2]

        parse_scenario(scenario_file, town_name, str(route_id), x_0, y_0)

        customized_data = get_customized_data(port, scenario_type, route_type)



        objectives, _, _, _ = run_simulation(customized_data, launch_server, episode_max_time, route_path, route_str, scenario_file, ego_car_model)

        if launch_server:
            launch_server = False


        # if check_bug(objectives):
        #     bug_str = None
        #     if objectives[0] > 0:
        #         collision_types = {'pedestrian_collision':pedestrian_types, 'car_collision':car_types, 'motercycle_collision':motorcycle_types, 'cyclist_collision':cyclist_types, 'static_collision':static_types}
        #         for k,v in collision_types.items():
        #             if object_type in v:
        #                 bug_str = k
        #         if not bug_str:
        #             bug_str = 'unknown_collision'+'_'+object_type
        #         bug_type = 1
        #     elif objectives[5]:
        #         bug_str = 'offroad'
        #         bug_type = 2
        #     elif objectives[6]:
        #         bug_str = 'wronglane'
        #         bug_type = 3
        #     else:
        #         bug_str = 'unknown'
        #         bug_type = 4
        #     print(bug_str, bug_type)


if __name__ == '__main__':
    port = 2003
    atexit.register(exit_handler, [port])

    scenario_type = 'one_pedestrians_cross_street_town05'
    route_type = 'town05_right_0'

    # scenario_type = 'default_dense'
    # route_type = 'none'

    os.environ['HAS_DISPLAY'] = '1'

    if route_type == 'none':
        run_multiple_short_routes(port, scenario_type, route_type)
    else:
        run_one_route(port, scenario_type, route_type)

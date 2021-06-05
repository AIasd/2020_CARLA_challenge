import argparse
import carla
import os
import numpy as np

from dask.distributed import Client, LocalCluster
from psutil import process_iter
from signal import SIGTERM
import socket
from collections import OrderedDict

import sys
import xml.etree.ElementTree as ET
import pathlib

import json
from sklearn import tree
import shlex
import subprocess
import time
import re
import math
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import logging
import traceback

from collections import deque

from carla_specific_utils.object_types import (
    weather_names,
    vehicle_colors,
    car_types,
    motorcycle_types,
    cyclist_types,
    pedestrian_types,
    vehicle_types,
    static_types
)


# ---------------- Misc -------------------
class arguments_info:
    def __init__(self):
        self.host = "localhost"
        self.port = "2000"
        self.sync = False
        self.debug = 0
        self.spectator = True
        self.record = ""
        self.timeout = "30.0"
        self.challenge_mode = True
        self.routes = None
        self.scenarios = "leaderboard/data/all_towns_traffic_scenarios_public.json"
        self.repetitions = 1
        self.agent = "scenario_runner/team_code/image_agent.py"
        self.agent_config = "models/epoch=24.ckpt"
        self.track = "SENSORS"
        self.resume = False
        self.checkpoint = ""
        self.weather_index = 19
        self.save_folder = "collected_data_customized"
        self.deviations_folder = ""
        self.background_vehicles = False
        self.save_action_based_measurements = 0
        self.changing_weather = False
        self.record_every_n_step = 2000

def specify_args():
    # general parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default="localhost", help="IP of the host server (default: localhost)"
    )
    parser.add_argument(
        "--port", default="2000", help="TCP port to listen to (default: 2000)"
    )
    parser.add_argument(
        "--sync", action="store_true", help="Forces the simulation to run synchronously"
    )
    parser.add_argument("--debug", type=int, help="Run with debug output", default=0)
    parser.add_argument(
        "--spectator", type=bool, help="Switch spectator view on?", default=True
    )
    parser.add_argument(
        "--record",
        type=str,
        default="",
        help="Use CARLA recording feature to create a recording of the scenario",
    )
    # modification: 30->40
    parser.add_argument(
        "--timeout",
        default="30.0",
        help="Set the CARLA client timeout value in seconds",
    )

    # simulation setup
    parser.add_argument(
        "--challenge-mode", action="store_true", help="Switch to challenge mode?"
    )
    parser.add_argument(
        "--routes",
        help="Name of the route to be executed. Point to the route_xml_file to be executed.",
        required=False,
    )
    parser.add_argument(
        "--scenarios",
        help="Name of the scenario annotation file to be mixed with the route.",
        required=False,
    )
    parser.add_argument(
        "--repetitions", type=int, default=1, help="Number of repetitions per route."
    )

    # agent-related options
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        help="Path to Agent's py file to evaluate",
        required=False,
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        help="Path to Agent's configuration file",
        default="",
    )

    parser.add_argument(
        "--track", type=str, default="SENSORS", help="Participation track: SENSORS, MAP"
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Resume execution from last checkpoint?",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./simulation_results.json",
        help="Path to checkpoint used for saving statistics and resuming",
    )

    # addition
    parser.add_argument(
        "--weather-index", type=int, default=0, help="see WEATHER for reference"
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="collected_data",
        help="Path to save simulation data",
    )
    parser.add_argument(
        "--deviations-folder",
        type=str,
        default="",
        help="Path to the folder that saves deviations data",
    )
    parser.add_argument("--save_action_based_measurements", type=int, default=0)
    parser.add_argument("--changing_weather", type=int, default=0)

    parser.add_argument('--record_every_n_step', type=int, default=2000)

    arguments = parser.parse_args()

    return arguments

def make_hierarchical_dir(folder_names):
    cur_folder_name = ""
    for i in range(len(folder_names)):
        cur_folder_name += folder_names[i]
        if not os.path.exists(cur_folder_name):
            os.mkdir(cur_folder_name)
        cur_folder_name += "/"
    return cur_folder_name

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", int(port))) == 0

def port_to_gpu(port):
    import torch

    n = torch.cuda.device_count()
    # n = 2
    gpu = port % n

    return gpu

def exit_handler(ports):
    for port in ports:
        while is_port_in_use(port):
            try:
                subprocess.run("kill $(lsof -t -i :" + str(port) + ")", shell=True)
                print("-" * 20, "kill server at port", port)
            except:
                continue

def get_sorted_subfolders(parent_folder, folder_type='all'):
    if 'rerun_bugs' in os.listdir(parent_folder):
        bug_folder = os.path.join(parent_folder, "rerun_bugs")
        non_bug_folder = os.path.join(parent_folder, "rerun_non_bugs")
    else:
        bug_folder = os.path.join(parent_folder, "bugs")
        non_bug_folder = os.path.join(parent_folder, "non_bugs")

    if folder_type == 'all':
        sub_folders = [
            os.path.join(bug_folder, sub_name) for sub_name in os.listdir(bug_folder)
        ] + [
            os.path.join(non_bug_folder, sub_name)
            for sub_name in os.listdir(non_bug_folder)
        ]
    elif folder_type == 'bugs':
        sub_folders = [
            os.path.join(bug_folder, sub_name) for sub_name in os.listdir(bug_folder)
        ]
    elif folder_type == 'non_bugs':
        sub_folders = [
            os.path.join(non_bug_folder, sub_name) for sub_name in os.listdir(non_bug_folder)
        ]

    ind_sub_folder_list = []
    for sub_folder in sub_folders:
        if os.path.isdir(sub_folder):
            ind = int(re.search(".*bugs/([0-9]*)", sub_folder).group(1))
            ind_sub_folder_list.append((ind, sub_folder))
            # print(sub_folder)
    ind_sub_folder_list_sorted = sorted(ind_sub_folder_list)
    subfolders = [filename for i, filename in ind_sub_folder_list_sorted]
    # print('len(subfolders)', len(subfolders))
    return subfolders

def load_data(subfolders):
    data_list = []
    is_bug_list = []

    objectives_list = []
    mask, labels, cur_info = None, None, None
    for sub_folder in subfolders:
        if os.path.isdir(sub_folder):
            pickle_filename = os.path.join(sub_folder, "cur_info.pickle")

            with open(pickle_filename, "rb") as f_in:
                cur_info = pickle.load(f_in)
                data, objectives, is_bug, mask, labels = reformat(cur_info)
                data_list.append(data)

                is_bug_list.append(is_bug)
                objectives_list.append(objectives)

    return data_list, np.array(is_bug_list), np.array(objectives_list), mask, labels, cur_info

def get_picklename(parent_folder):
    pickle_folder = parent_folder + "/bugs/"
    if not os.path.isdir(pickle_folder):
        pickle_folder = parent_folder + "/0/bugs/"
    i = 1
    while i < len(os.listdir(pickle_folder)):
        if os.path.isdir(pickle_folder + str(i)):
            pickle_folder = pickle_folder + str(i) + "/cur_info.pickle"
            break
        i += 1
    return pickle_folder

def reformat(cur_info):
    objectives = cur_info["objectives"]
    is_bug = cur_info["is_bug"]

    (
        ego_linear_speed,
        min_d,
        d_angle_norm,
        offroad_d,
        wronglane_d,
        dev_dist,
        is_collision,
        is_offroad,
        is_wrong_lane,
        is_run_red_light,
    ) = objectives
    # accident_x, accident_y = cur_info["loc"]

    # route_completion = cur_info['route_completion']

    # result_info = [ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, accident_x, accident_y, is_bug, route_completion]
    # hack: backward compatibility that removes the port info in x

    x, mask, labels = (
        cur_info["x"],
        cur_info["mask"],
        cur_info["labels"]
    )
    if x.shape[0] == len(labels) + 1:
        x = x[:-1]

    return x, objectives, int(is_bug), mask, labels

def set_general_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

def rand_real(rng, low, high):
    return rng.random() * (high - low) + low
# ---------------- Misc -------------------



# ---------------- Uniqueness -------------------
# def eliminate_duplicates_for_list(
#     mask, xl, xu, p, c, th, X, prev_unique_bugs, tmp_off=[]
# ):
#     new_X = []
#     similar = False
#     for x in X:
#         for x2 in prev_unique_bugs:
#             if is_similar(x, x2, mask, xl, xu, p, c, th):
#                 similar = True
#                 break
#         if not similar:
#             for x2 in tmp_off:
#                 # print(x)
#                 # print(x2)
#                 # print(mask, xl, xu, p, c, th)
#                 # print(len(x), len(x2), len(mask), len(xl), len(xu))
#                 if is_similar(x, x2, mask, xl, xu, p, c, th):
#                     similar = True
#                     break
#         if not similar:
#             new_X.append(x)
#     return new_X

# def is_similar(
#     x_1,
#     x_2,
#     mask,
#     xl,
#     xu,
#     p,
#     c,
#     th,
#     y_i=-1,
#     y_j=-1,
#     verbose=False,
#     labels=[],
# ):
#
#     if y_i == y_j:
#         eps = 1e-8
#
#         # only consider those fields that can change when considering diversity
#         variant_fields = (xu - xl) > eps
#         mask = mask[variant_fields]
#         xl = xl[variant_fields]
#         xu = xu[variant_fields]
#         x_1 = x_1[variant_fields]
#         x_2 = x_2[variant_fields]
#         variant_fields_num = np.sum(variant_fields)
#         if verbose:
#             print(
#                 variant_fields_num,
#                 "/",
#                 len(variant_fields),
#                 "fields are used for checking similarity",
#             )
#
#         int_inds = mask == "int"
#         real_inds = mask == "real"
#         # print(int_inds, real_inds)
#         int_diff_raw = np.abs(x_1[int_inds] - x_2[int_inds])
#         int_diff = np.ones(int_diff_raw.shape) * (int_diff_raw > eps)
#
#         real_diff_raw = (
#             np.abs(x_1[real_inds] - x_2[real_inds]) / (np.abs(xu - xl) + eps)[real_inds]
#         )
#         # print(int_diff_raw, real_diff_raw)
#         real_diff = np.ones(real_diff_raw.shape) * (real_diff_raw > c)
#
#         diff = np.concatenate([int_diff, real_diff])
#         # print(diff, p)
#         diff_norm = np.linalg.norm(diff, p)
#
#         th_num = np.max([np.round(th * variant_fields_num), 1])
#         equal = diff_norm < th_num
#
#         if verbose:
#             print("diff_norm, th_num", diff_norm, th_num)
#
#     else:
#         equal = False
#     return equal
#
# def is_distinct(x, X, mask, xl, xu, p, c, th, verbose=True):
#     verbose = False
#     if len(X) == 0:
#         return True
#     else:
#         mask_np = np.array(mask)
#         xl_np = np.array(xl)
#         xu_np = np.array(xu)
#         x = np.array(x)
#         X = np.stack(X)
#         for i, x_i in enumerate(X):
#             # if verbose:
#             #     print(i, '- th prev x checking similarity')
#             similar = is_similar(
#                 x,
#                 x_i,
#                 mask_np,
#                 xl_np,
#                 xu_np,
#                 p,
#                 c,
#                 th,
#                 verbose=verbose,
#             )
#             if similar:
#                 if verbose:
#                     print("similar with", i)
#                 return False
#         return True

def is_distinct_vectorized(cur_X, prev_X, mask, xl, xu, p, c, th, verbose=True):
    cur_X = np.array(cur_X)
    prev_X = np.array(prev_X)
    eps = 1e-10
    remaining_inds = np.arange(cur_X.shape[0])

    mask = np.array(mask)
    xl = np.array(xl)
    xu = np.array(xu)

    n = len(mask)

    variant_fields = (xu - xl) > eps
    variant_fields_num = np.sum(variant_fields)
    th_num = np.max([np.round(th * variant_fields_num), 1])

    mask = mask[variant_fields]
    int_inds = mask == "int"
    real_inds = mask == "real"
    xl = xl[variant_fields]
    xu = xu[variant_fields]
    xl = np.concatenate([np.zeros(np.sum(int_inds)), xl[real_inds]])
    xu = np.concatenate([0.99*np.ones(np.sum(int_inds)), xu[real_inds]])

    # hack: backward compatibility with previous run data
    # if cur_X.shape[1] == n-1:
    #     cur_X = np.concatenate([cur_X, np.zeros((cur_X.shape[0], 1))], axis=1)
    cur_X = cur_X[:, variant_fields]
    cur_X = np.concatenate([cur_X[:, int_inds], cur_X[:, real_inds]], axis=1) / (np.abs(xu - xl) + eps)

    if len(prev_X) > 0:
        prev_X = prev_X[:, variant_fields]
        prev_X = np.concatenate([prev_X[:, int_inds], prev_X[:, real_inds]], axis=1) / (np.abs(xu - xl) + eps)

        diff_raw = np.abs(np.expand_dims(cur_X, axis=1) - np.expand_dims(prev_X, axis=0))
        diff = np.ones(diff_raw.shape) * (diff_raw > c)
        diff_norm = np.linalg.norm(diff, p, axis=2)
        equal = diff_norm < th_num
        remaining_inds = np.mean(equal, axis=1) == 0
        remaining_inds = np.arange(cur_X.shape[0])[remaining_inds]

        # print('remaining_inds', remaining_inds, np.arange(cur_X.shape[0])[remaining_inds], cur_X[np.arange(cur_X.shape[0])[remaining_inds]])
        if verbose:
            print('prev X filtering:',cur_X.shape[0], '->', len(remaining_inds))

    if len(remaining_inds) == 0:
        return []

    cur_X_remaining = cur_X[remaining_inds]
    print('len(cur_X_remaining)', len(cur_X_remaining))
    unique_inds = []
    for i in range(len(cur_X_remaining)-1):
        diff_raw = np.abs(np.expand_dims(cur_X_remaining[i], axis=0) - cur_X_remaining[i+1:])
        diff = np.ones(diff_raw.shape) * (diff_raw > c)
        diff_norm = np.linalg.norm(diff, p, axis=1)
        equal = diff_norm < th_num
        if np.mean(equal) == 0:
            unique_inds.append(i)

    unique_inds.append(len(cur_X_remaining)-1)

    if verbose:
        print('cur X filtering:',cur_X_remaining.shape[0], '->', len(unique_inds))

    if len(unique_inds) == 0:
        return []
    remaining_inds = remaining_inds[np.array(unique_inds)]


    return remaining_inds

def eliminate_repetitive_vectorized(cur_X, mask, xl, xu, p, c, th, verbose=True):
    cur_X = np.array(cur_X)
    eps = 1e-8
    verbose = False
    remaining_inds = np.arange(cur_X.shape[0])
    if len(cur_X) == 0:
        return remaining_inds
    else:
        mask = np.array(mask)
        xl = np.array(xl)
        xu = np.array(xu)

        variant_fields = (xu - xl) > eps
        variant_fields_num = np.sum(variant_fields)
        th_num = np.max([np.round(th * variant_fields_num), 1])

        mask = mask[variant_fields]
        xl = xl[variant_fields]
        xu = xu[variant_fields]

        cur_X = cur_X[:, variant_fields]

        int_inds = mask == "int"
        real_inds = mask == "real"

        xl = np.concatenate([np.zeros(np.sum(int_inds)), xl[real_inds]])
        xu = np.concatenate([0.99*np.ones(np.sum(int_inds)), xu[real_inds]])

        cur_X = np.concatenate([cur_X[:, int_inds], cur_X[:, real_inds]], axis=1) / (np.abs(xu - xl) + eps)


        unique_inds = []
        for i in range(len(cur_X)-1):
            diff_raw = np.abs(np.expand_dims(cur_X[i], axis=0) - cur_X[i+1:])
            diff = np.ones(diff_raw.shape) * (diff_raw > c)
            diff_norm = np.linalg.norm(diff, p, axis=1)
            equal = diff_norm < th_num
            if np.mean(equal) == 0:
                unique_inds.append(i)

        if len(unique_inds) == 0:
            return []
        remaining_inds = np.array(unique_inds)
        if verbose:
            print('cur X filtering:',cur_X.shape[0], '->', len(remaining_inds))

        return remaining_inds

# def get_distinct_data_points(data_points, mask, xl, xu, p, c, th, y=[]):
#
#     # ['forward', 'backward']
#     order = "forward"
#
#     mask_arr = np.array(mask)
#     xl_arr = np.array(xl)
#     xu_arr = np.array(xu)
#     # print(data_points)
#     if len(data_points) == 0:
#         return [], []
#     if len(data_points) == 1:
#         return data_points, [0]
#     else:
#         if order == "backward":
#             distinct_inds = []
#             for i in range(len(data_points) - 1):
#                 similar = False
#                 for j in range(i + 1, len(data_points)):
#                     if len(y) > 0:
#                         y_i = y[i]
#                         y_j = y[j]
#                     else:
#                         y_i = -1
#                         y_j = -1
#                     similar = is_similar(
#                         data_points[i],
#                         data_points[j],
#                         mask_arr,
#                         xl_arr,
#                         xu_arr,
#                         p,
#                         c,
#                         th,
#                         y_i=y_i,
#                         y_j=y_j,
#                     )
#                     if similar:
#                         break
#                 if not similar:
#                     distinct_inds.append(i)
#             distinct_inds.append(len(data_points) - 1)
#         elif order == "forward":
#             distinct_inds = [0]
#             for i in range(1, len(data_points)):
#                 similar = False
#                 for j in distinct_inds:
#                     if len(y) > 0:
#                         y_i = y[i]
#                         y_j = y[j]
#                     else:
#                         y_i = -1
#                         y_j = -1
#                     similar = is_similar(
#                         data_points[i],
#                         data_points[j],
#                         mask_arr,
#                         xl_arr,
#                         xu_arr,
#                         p,
#                         c,
#                         th,
#                         y_i=y_i,
#                         y_j=y_j,
#                     )
#                     if similar:
#                         # print(i, j)
#                         break
#                 if not similar:
#                     distinct_inds.append(i)
#
#     return list(np.array(data_points)[distinct_inds]), distinct_inds
# ---------------- Uniqueness -------------------



# ---------------- Bug, Objective -------------------

def check_bug(objectives):
    # speed needs to be larger than 0.1 to avoid false positive
    return objectives[0] > 0.1 or objectives[-3] or objectives[-2] or objectives[-1]

def get_if_bug_list(objectives_list):
    if_bug_list = []
    for objective in objectives_list:
        if_bug_list.append(check_bug(objective))
    return np.array(if_bug_list)


def process_specific_bug(
    bug_type_ind, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
):
    verbose = True
    chosen_bugs = np.array(bugs_type_list) == bug_type_ind

    specific_bugs = np.array(bugs)[chosen_bugs]
    specific_bugs_inds_list = np.array(bugs_inds_list)[chosen_bugs]

    # unique_specific_bugs, specific_distinct_inds = get_distinct_data_points(
    #     specific_bugs, mask, xl, xu, p, c, th
    # )

    specific_distinct_inds = is_distinct_vectorized(specific_bugs, [], mask, xl, xu, p, c, th, verbose=verbose)
    unique_specific_bugs = specific_bugs[specific_distinct_inds]

    unique_specific_bugs_inds_list = specific_bugs_inds_list[specific_distinct_inds]

    return (
        list(unique_specific_bugs),
        list(unique_specific_bugs_inds_list),
        len(unique_specific_bugs),
    )

def classify_bug_type(objectives, object_type=''):
    bug_str = ''
    bug_type = 5
    if objectives[0] > 0.1:
        collision_types = {'pedestrian_collision':pedestrian_types, 'car_collision':car_types, 'motercycle_collision':motorcycle_types, 'cyclist_collision':cyclist_types, 'static_collision':static_types}
        for k,v in collision_types.items():
            if object_type in v:
                bug_str = k
        if not bug_str:
            bug_str = 'unknown_collision'+'_'+object_type
        bug_type = 1
    elif objectives[-3]:
        bug_str = 'offroad'
        bug_type = 2
    elif objectives[-2]:
        bug_str = 'wronglane'
        bug_type = 3
    if objectives[-1]:
        bug_str += 'run_red_light'
        if bug_type > 4:
            bug_type = 4
    return bug_type, bug_str

def get_unique_bugs(
    X, objectives_list, mask, xl, xu, unique_coeff, objective_weights, return_mode='unique_inds_and_interested_and_bugcounts', consider_interested_bugs=1, bugs_type_list=[], bugs=[], bugs_inds_list=[]
):
    p, c, th = unique_coeff
    # hack:
    if len(bugs) == 0:
        for i, (x, objectives) in enumerate(zip(X, objectives_list)):
            if check_bug(objectives):
                bug_type, _ = classify_bug_type(objectives)
                bugs_type_list.append(bug_type)
                bugs.append(x)
                bugs_inds_list.append(i)

    (
        unique_collision_bugs,
        unique_collision_bugs_inds_list,
        unique_collision_num,
    ) = process_specific_bug(
        1, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_offroad_bugs,
        unique_offroad_bugs_inds_list,
        unique_offroad_num,
    ) = process_specific_bug(
        2, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_wronglane_bugs,
        unique_wronglane_bugs_inds_list,
        unique_wronglane_num,
    ) = process_specific_bug(
        3, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_redlight_bugs,
        unique_redlight_bugs_inds_list,
        unique_redlight_num,
    ) = process_specific_bug(
        4, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )

    unique_bugs = unique_collision_bugs + unique_offroad_bugs + unique_wronglane_bugs + unique_redlight_bugs
    unique_bugs_num = len(unique_bugs)
    unique_bugs_inds_list = unique_collision_bugs_inds_list + unique_offroad_bugs_inds_list + unique_wronglane_bugs_inds_list + unique_redlight_bugs_inds_list

    if consider_interested_bugs:
        collision_activated = np.sum(objective_weights[:3] != 0) > 0
        offroad_activated = (np.abs(objective_weights[3]) > 0) | (
            np.abs(objective_weights[5]) > 0
        )
        wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
            np.abs(objective_weights[5]) > 0
        )
        red_light_activated = np.abs(objective_weights[-1]) > 0

        interested_unique_bugs = []
        if collision_activated:
            interested_unique_bugs += unique_collision_bugs
        if offroad_activated:
            interested_unique_bugs += unique_offroad_bugs
        if wronglane_activated:
            interested_unique_bugs += unique_wronglane_bugs
        if red_light_activated:
            interested_unique_bugs += unique_redlight_bugs
    else:
        interested_unique_bugs = unique_bugs

    num_of_collisions = np.sum(np.array(bugs_type_list)==1)
    num_of_offroad = np.sum(np.array(bugs_type_list)==2)
    num_of_wronglane = np.sum(np.array(bugs_type_list)==3)
    num_of_redlight = np.sum(np.array(bugs_type_list)==4)

    if return_mode == 'unique_inds_and_interested_and_bugcounts':
        return unique_bugs, unique_bugs_inds_list, interested_unique_bugs, [num_of_collisions, num_of_offroad, num_of_wronglane, num_of_redlight,
        unique_collision_num, unique_offroad_num, unique_wronglane_num, unique_redlight_num]
    elif return_mode == 'return_bug_info':
        return unique_bugs, (bugs, bugs_type_list, bugs_inds_list, interested_unique_bugs)
    elif return_mode == 'return_indices':
        return unique_bugs, unique_bugs_inds_list
    else:
        return unique_bugs



def choose_weight_inds(objective_weights):
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    if collision_activated:
        weight_inds = np.arange(0,3)
    elif offroad_activated or wronglane_activated:
        weight_inds = np.arange(3,6)
    elif red_light_activated:
        weight_inds = np.arange(9,10)
    else:
        raise
    return weight_inds

def determine_y_upon_weights(objective_list, objective_weights):
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    y = np.zeros(len(objective_list))
    for i, obj in enumerate(objective_list):
        cond = 0
        if collision_activated:
            cond |= obj[0] > 0.1
        if offroad_activated:
            cond |= obj[-3] == 1
        if wronglane_activated:
            cond |= obj[-2] == 1
        if red_light_activated:
            cond |= obj[-1] == 1
        y[i] = cond

    return y

def get_all_y(objective_list, objective_weights):
    # is_collision, is_offroad, is_wrong_lane, is_run_red_light
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    y_list = np.zeros((4, len(objective_list)))

    for i, obj in enumerate(objective_list):
        if collision_activated:
            y_list[0, i] = obj[0] > 0.1
        if offroad_activated:
            y_list[1, i] = obj[-3] == 1
        if wronglane_activated:
            y_list[2, i] = obj[-2] == 1
        if red_light_activated:
            y_list[3, i] = obj[-1] == 1

    return y_list

def get_F(current_objectives, all_objectives, objective_weights, use_single_objective):
    # standardize current objectives using all objectives so far
    all_objectives = np.stack(all_objectives)
    current_objectives = np.stack(current_objectives)

    standardize = StandardScaler()
    standardize.fit(all_objectives)
    standardize.transform(current_objectives)

    current_objectives *= objective_weights

    if use_single_objective:
        current_F = np.expand_dims(np.sum(current_objectives, axis=1), axis=1)
    else:
        current_F = np.row_stack(current_objectives)
    return current_F

def count_and_group_output_unique_bugs(inds, outputs, labels, min_bounds, max_bounds, diff_th):
    '''
    ***grid counting: maximum number of distinct elements
    distinct counting: minimum number of distinct elements
    1.general
    bug type, normalized (/|start location - end location|) bug location, ego car speed when bug happens

    2.collision specific
    collision object type (i.e. pedestrian, bicyclist, small vehicle, or truck), normalized (/car width) relative angle of the other involved object at collision

    '''

    m = len(labels)

    # print(outputs.shape, outputs)
    outputs_grid_inds = ((outputs - min_bounds)*diff_th) / (max_bounds - min_bounds)
    outputs_grid_inds = outputs_grid_inds.astype(int)
    # print(outputs_grid_inds.shape, outputs_grid_inds)

    from collections import defaultdict
    unique_bugs_group = defaultdict(list)

    for i in range(outputs.shape[0]):
        unique_bugs_group[tuple(outputs_grid_inds[i])].append((inds[i], outputs[i]))

    return unique_bugs_group
# ---------------- Bug, Objective -------------------



# ---------------- NN -------------------
# dependent on description labels
def encode_fields(x, labels, labels_to_encode):
    keywords_dict = {
        "num_of_weathers": len(weather_names),
        "num_of_vehicle_colors": len(vehicle_colors),
        "num_of_pedestrian_types": len(pedestrian_types),
        "num_of_vehicle_types": len(vehicle_types),
    }
    # keywords_dict = {'num_of_weathers': len(weather_names)}

    x = np.array(x).astype(np.float)

    encode_fields = []
    inds_to_encode = []
    for label in labels_to_encode:
        for k, v in keywords_dict.items():
            if k in label:
                ind = labels.index(label)
                inds_to_encode.append(ind)

                encode_fields.append(v)
                break
    inds_non_encode = list(set(range(x.shape[1])) - set(inds_to_encode))

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

    embed_dims = int(np.sum(encode_fields))
    embed_fields_num = len(encode_fields)
    data_for_fit_encode = np.zeros((embed_dims, embed_fields_num))
    counter = 0
    for i, encode_field in enumerate(encode_fields):
        for j in range(encode_field):
            data_for_fit_encode[counter, i] = j
            counter += 1
    enc.fit(data_for_fit_encode)

    embed = np.array(x[:, inds_to_encode].astype(np.int))
    embed = enc.transform(embed)

    x = np.concatenate([embed, x[:, inds_non_encode]], axis=1).astype(np.float)

    return x, enc, inds_to_encode, inds_non_encode, encode_fields

# dependent on description labels
def get_labels_to_encode(labels):
    # hack: explicitly listing keywords for encode to be imported
    keywords_for_encode = [
        "num_of_weathers",
        "num_of_vehicle_colors",
        "num_of_pedestrian_types",
        "num_of_vehicle_types",
    ]
    labels_to_encode = []
    for label in labels:
        for keyword in keywords_for_encode:
            if keyword in label:
                labels_to_encode.append(label)
    return labels_to_encode

def max_one_hot_op(images, encode_fields):
    m = np.sum(encode_fields)
    one_hotezed_images_embed = np.zeros([images.shape[0], m])
    s = 0
    for field_len in encode_fields:
        max_inds = np.argmax(images[:, s : s + field_len], axis=1)
        one_hotezed_images_embed[np.arange(images.shape[0]), s + max_inds] = 1
        s += field_len
    images[:, :m] = one_hotezed_images_embed

def customized_fit(X_train, standardize, one_hot_fields_len, partial=True):
    # print('\n'*2, 'customized_fit X_train.shape', X_train.shape, '\n'*2)
    if partial:
        standardize.fit(X_train[:, one_hot_fields_len:])
    else:
        standardize.fit(X_train)

def customized_standardize(X, standardize, m, partial=True, scale_only=False):
    # print(X[:, :m].shape, standardize.transform(X[:, m:]).shape)
    if partial:
        if scale_only:
            res_non_encode = X[:, m:] * standardize.scale_
        else:
            res_non_encode = standardize.transform(X[:, m:])
        res = np.concatenate([X[:, :m], standardize.transform(X[:, m:])], axis=1)
    else:
        if scale_only:
            res = X * standardize.scale_
        else:
            res = standardize.transform(X)
    return res

def customized_inverse_standardize(X, standardize, m, partial=True, scale_only=False):
    if partial:
        if scale_only:
            res_non_encode = X[:, m:] * standardize.scale_
        else:
            res_non_encode = standardize.inverse_transform(X[:, m:])
        res = np.concatenate([X[:, :m], res_non_encode], axis=1)
    else:
        if scale_only:
            res = X * standardize.scale_
        else:
            res = standardize.inverse_transform(X)
    return res

def decode_fields(x, enc, inds_to_encode, inds_non_encode, encode_fields, adv=False):
    n = x.shape[0]
    m = len(inds_to_encode) + len(inds_non_encode)
    embed_dims = np.sum(encode_fields)

    embed = x[:, :embed_dims]
    kept = x[:, embed_dims:]

    if adv:
        one_hot_embed = np.zeros(embed.shape)
        s = 0
        for field_len in encode_fields:
            max_inds = np.argmax(x[:, s : s + field_len], axis=1)
            one_hot_embed[np.arange(x.shape[0]), s + max_inds] = 1
            s += field_len
        embed = one_hot_embed

    x_encoded = enc.inverse_transform(embed)
    # print('encode_fields', encode_fields)
    # print('embed', embed[0], x_encoded[0])
    x_decoded = np.zeros([n, m])
    x_decoded[:, inds_non_encode] = kept
    x_decoded[:, inds_to_encode] = x_encoded

    return x_decoded

def remove_fields_not_changing(x, embed_dims=0, xl=[], xu=[]):
    eps = 1e-8
    if len(xl) > 0:
        cond = xu - xl > eps
    else:
        cond = np.std(x, axis=0) > eps
    kept_fields = np.where(cond)[0]
    if embed_dims > 0:
        kept_fields = list(set(kept_fields).union(set(range(embed_dims))))

    removed_fields = list(set(range(x.shape[1])) - set(kept_fields))
    x_removed = x[:, removed_fields]
    x = x[:, kept_fields]
    return x, x_removed, kept_fields, removed_fields

def recover_fields_not_changing(x, x_removed, kept_fields, removed_fields):
    n = x.shape[0]
    m = len(kept_fields) + len(removed_fields)

    # this is True usually when adv is used
    if x_removed.shape[0] != n:
        x_removed = np.array([x_removed[0] for _ in range(n)])
    x_recovered = np.zeros([n, m])
    x_recovered[:, kept_fields] = x
    x_recovered[:, removed_fields] = x_removed

    return x_recovered

def process_X(
    initial_X,
    labels,
    xl_ori,
    xu_ori,
    cutoff,
    cutoff_end,
    partial,
    unique_bugs_len,
    standardize_prev=None,
):

    labels_to_encode = get_labels_to_encode(labels)
    X, enc, inds_to_encode, inds_non_encode, encoded_fields = encode_fields(
        initial_X, labels, labels_to_encode
    )
    one_hot_fields_len = np.sum(encoded_fields)

    xl, xu = encode_bounds(
        xl_ori, xu_ori, inds_to_encode, inds_non_encode, encoded_fields
    )

    labels_non_encode = np.array(labels)[inds_non_encode]
    # print(np.array(X).shape)
    X, X_removed, kept_fields, removed_fields = remove_fields_not_changing(
        X, one_hot_fields_len, xl=xl, xu=xu
    )
    # print(np.array(X).shape)

    param_for_recover_and_decode = (
        X_removed,
        kept_fields,
        removed_fields,
        enc,
        inds_to_encode,
        inds_non_encode,
        encoded_fields,
        xl_ori,
        xu_ori,
        unique_bugs_len,
    )

    xl = xl[kept_fields]
    xu = xu[kept_fields]

    kept_fields_non_encode = kept_fields - one_hot_fields_len
    kept_fields_non_encode = kept_fields_non_encode[kept_fields_non_encode >= 0]
    labels_used = labels_non_encode[kept_fields_non_encode]

    X_train, X_test = X[:cutoff], X[cutoff:cutoff_end]
    # print('X_train.shape, X_test.shape', X_train.shape, X_test.shape, one_hot_fields_len)
    if standardize_prev:
        standardize = standardize_prev
    else:
        standardize = StandardScaler()
        customized_fit(X_train, standardize, one_hot_fields_len, partial)
    X_train = customized_standardize(X_train, standardize, one_hot_fields_len, partial)
    if len(X_test) > 0:
        X_test = customized_standardize(X_test, standardize, one_hot_fields_len, partial)
    xl = customized_standardize(
        np.array([xl]), standardize, one_hot_fields_len, partial
    )[0]
    xu = customized_standardize(
        np.array([xu]), standardize, one_hot_fields_len, partial
    )[0]

    return (
        X_train,
        X_test,
        xl,
        xu,
        labels_used,
        standardize,
        one_hot_fields_len,
        param_for_recover_and_decode,
    )


def inverse_process_X(
    initial_test_x_adv_list,
    standardize,
    one_hot_fields_len,
    partial,
    X_removed,
    kept_fields,
    removed_fields,
    enc,
    inds_to_encode,
    inds_non_encode,
    encoded_fields,
):
    test_x_adv_list = customized_inverse_standardize(
        initial_test_x_adv_list, standardize, one_hot_fields_len, partial
    )
    X = recover_fields_not_changing(
        test_x_adv_list, X_removed, kept_fields, removed_fields
    )
    X_final_test = decode_fields(
        X, enc, inds_to_encode, inds_non_encode, encoded_fields, adv=True
    )
    return X_final_test
# ---------------- NN -------------------



# ---------------- ADV -------------------
def if_violate_constraints_vectorized(X, customized_constraints, labels, verbose=False):
    labels_to_id = {label: i for i, label in enumerate(labels)}

    keywords = ["coefficients", "labels", "value"]
    extra_keywords = ["power"]

    if_violate = False
    violated_constraints = []
    involved_labels = set()

    X = np.array(X)
    remaining_inds = np.arange(X.shape[0])

    for i, constraint in enumerate(customized_constraints):
        for k in keywords:
            assert k in constraint
        assert len(constraint["coefficients"]) == len(constraint["labels"])

        ids = np.array([labels_to_id[label] for label in constraint["labels"]])


        # x_ids = [x[id] for id in ids]
        if "powers" in constraint:
            powers = np.array(constraint["powers"])
        else:
            powers = np.array([1 for _ in range(len(ids))])

        coeff = np.array(constraint["coefficients"])
        # features = np.array(x_ids)
        # print(X.shape)
        # print(type(remaining_inds))
        # print(type(ids))
        # print(X[remaining_inds, ids].shape)
        # print(powers.shape)
        if_violate_current = (
            np.sum(coeff * np.power(X[remaining_inds[:, None], ids], powers), axis=1) > constraint["value"]
        )
        remaining_inds = remaining_inds[if_violate_current==0]
    if verbose:
        print('constraints filtering', len(X), '->', len(remaining_inds))

    return remaining_inds

def if_violate_constraints(x, customized_constraints, labels, verbose=False):
    labels_to_id = {label: i for i, label in enumerate(labels)}

    keywords = ["coefficients", "labels", "value"]
    extra_keywords = ["power"]

    if_violate = False
    violated_constraints = []
    involved_labels = set()

    for i, constraint in enumerate(customized_constraints):
        for k in keywords:
            assert k in constraint
        assert len(constraint["coefficients"]) == len(constraint["labels"])

        ids = [labels_to_id[label] for label in constraint["labels"]]
        x_ids = [x[id] for id in ids]
        if "powers" in constraint:
            powers = np.array(constraint["powers"])
        else:
            powers = np.array([1 for _ in range(len(ids))])

        coeff = np.array(constraint["coefficients"])
        features = np.array(x_ids)

        if_violate_current = (
            np.sum(coeff * np.power(features, powers)) > constraint["value"]
        )
        if if_violate_current:
            if_violate = True
            violated_constraints.append(constraint)
            involved_labels = involved_labels.union(set(constraint["labels"]))
            if verbose:
                print("\n" * 1, "violate_constraints!!!!", "\n" * 1)
                print(
                    coeff,
                    features,
                    powers,
                    np.sum(coeff * np.power(features, powers)),
                    constraint["value"],
                    constraint["labels"],
                )

    return if_violate, [violated_constraints, involved_labels]

def encode_bounds(xl, xu, inds_to_encode, inds_non_encode, encode_fields):
    m1 = np.sum(encode_fields)
    m2 = len(inds_non_encode)
    m = m1 + m2

    xl_embed, xu_embed = np.zeros(m1), np.ones(m1)

    xl_new = np.concatenate([xl_embed, xl[inds_non_encode]])
    xu_new = np.concatenate([xu_embed, xu[inds_non_encode]])

    return xl_new, xu_new
# ---------------- ADV -------------------



# ---------------- NSGA2-DT -------------------
# check if x is in critical regions of the tree
def is_critical_region(x, estimator, critical_unique_leaves):
    leave_id = estimator.apply(x.reshape(1, -1))[0]
    print(leave_id, critical_unique_leaves)
    return leave_id in critical_unique_leaves

def filter_critical_regions(X, y):
    print("\n" * 20)
    print("+" * 100, "filter_critical_regions", "+" * 100)

    min_samples_split = np.max([int(0.1 * X.shape[0]), 2])
    # estimator = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, min_impurity_decrease=0.01, random_state=0)
    estimator = tree.DecisionTreeClassifier(
        min_samples_split=min_samples_split,
        min_impurity_decrease=0.0001,
        random_state=0,
    )
    print(X.shape, y.shape)
    # print(X, y)
    estimator = estimator.fit(X, y)

    leave_ids = estimator.apply(X)
    print("leave_ids", leave_ids)

    unique_leave_ids = np.unique(leave_ids)
    unique_leaves_bug_num = np.zeros(unique_leave_ids.shape[0])
    unique_leaves_normal_num = np.zeros(unique_leave_ids.shape[0])

    for j, unique_leave_id in enumerate(unique_leave_ids):
        for i, leave_id in enumerate(leave_ids):
            if leave_id == unique_leave_id:
                if y[i] == 0:
                    unique_leaves_normal_num[j] += 1
                else:
                    unique_leaves_bug_num[j] += 1

    for i, unique_leave_i in enumerate(unique_leave_ids):
        print(
            "unique_leaves",
            unique_leave_i,
            unique_leaves_bug_num[i],
            unique_leaves_normal_num[i],
        )

    critical_unique_leaves = unique_leave_ids[
        unique_leaves_bug_num >= unique_leaves_normal_num
    ]

    print("critical_unique_leaves", critical_unique_leaves)

    inds = np.array([leave_id in critical_unique_leaves for leave_id in leave_ids])
    print("\n" * 20)

    return estimator, inds, critical_unique_leaves
# ---------------- NSGA2-DT -------------------



# ---------------- NSGA2-SM -------------------
def pretrain_regression_nets(initial_X, initial_objectives_list, objective_weights, xl_ori, xu_ori, labels, customized_constraints, cutoff, cutoff_end):

    # we are not using it so set it to 0 for placeholding
    unique_bugs_len = 0
    partial = True
    # print('pretrain initial_X.shape', np.array(initial_X).shape)
    # print('pretrain len(labels)', len(labels))
    print(np.array(initial_X).shape, cutoff, cutoff_end)
    (
        X_train,
        X_test,
        xl,
        xu,
        labels_used,
        standardize,
        one_hot_fields_len,
        param_for_recover_and_decode,
    ) = process_X(
        initial_X, labels, xl_ori, xu_ori, cutoff, cutoff_end, partial, unique_bugs_len
    )

    (
        X_removed,
        kept_fields,
        removed_fields,
        enc,
        inds_to_encode,
        inds_non_encode,
        encoded_fields,
        _,
        _,
        unique_bugs_len,
    ) = param_for_recover_and_decode

    weight_inds = choose_weight_inds(objective_weights)


    from pgd_attack import train_regression_net
    chosen_weights = objective_weights[weight_inds]
    clfs = []
    confs = []
    for weight_ind in weight_inds:
        y_i = np.array([obj[weight_ind] for obj in initial_objectives_list])
        y_train_i, y_test_i = y_i[:cutoff], y_i[cutoff:cutoff_end]

        clf_i, conf_i = train_regression_net(
            X_train, y_train_i, X_test, y_test_i, batch_train=200, return_test_err=True
        )
        clfs.append(clf_i)
        confs.append(conf_i)

    confs = np.array(confs)*chosen_weights
    return clfs, confs, chosen_weights, standardize
# ---------------- NSGA2-SM -------------------



# ---------------- acquisition related -------------------
# TBD: greedily add point
def calculate_rep_d(clf, X_train, X_test):
    X_train_embed = clf.extract_embed(X_train)
    X_test_embed = clf.extract_embed(X_test)
    X_combined_embed = np.concatenate([X_train_embed, X_test_embed])

    d_list = []
    for x_test_embed in X_test_embed:
        d = np.linalg.norm(X_combined_embed - x_test_embed, axis=1)
        # sorted_d = np.sort(d)
        # d_list.append(sorted_d[1])
        d_list.append(d)
    return np.array(d_list)

def select_batch_max_d_greedy(d_list, train_test_cutoff, batch_size):
    consider_inds = np.arange(train_test_cutoff)
    remaining_inds = np.arange(len(d_list))
    chosen_inds = []

    print('d_list.shape', d_list.shape)
    print('remaining_inds.shape', remaining_inds.shape)
    print('consider_inds.shape', consider_inds.shape)
    for i in range(batch_size):
        # print(i)
        # print('d_list[np.ix_(remaining_inds, consider_inds)].shape', d_list[np.ix_(remaining_inds, consider_inds)].shape)
        min_d_list = np.min(d_list[np.ix_(remaining_inds, consider_inds)], axis=1)
        # print('min_d_list', min_d_list.shape, min_d_list)
        remaining_inds_top_ind = np.argmax(min_d_list)
        chosen_ind = remaining_inds[remaining_inds_top_ind]

        # print('chosen_ind', chosen_ind)
        consider_inds = np.append(consider_inds, chosen_ind)
        # print('remaining_inds before', remaining_inds)
        # print('remaining_inds_top_ind', remaining_inds_top_ind)
        remaining_inds = np.delete(remaining_inds, remaining_inds_top_ind)
        # print('remaining_inds after', remaining_inds)
        chosen_inds.append(chosen_ind)
    return chosen_inds
# ---------------- acquisition related -------------------
if __name__ == "__main__":
    print('ok')

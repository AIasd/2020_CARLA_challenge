import argparse
import carla
import os
import numpy as np
from leaderboard.customized.object_params import Static, Pedestrian, Vehicle
from dask.distributed import Client, LocalCluster
from psutil import process_iter
from signal import SIGTERM
import socket
from collections import OrderedDict
from object_types import WEATHERS, pedestrian_types, vehicle_types, static_types, vehicle_colors, car_types, motorcycle_types, cyclist_types

import sys
import xml.etree.ElementTree as ET
import pathlib
from leaderboard.utils.route_parser import RouteParser

import json
from sklearn import tree
import shlex
import subprocess
import time
import re
import math

def visualize_route(route):
    n = len(route)

    x_list = []
    y_list = []

    # The following code prints out the planned route
    for i, (transform, command) in enumerate(route):
        x = transform.location.x
        y = transform.location.y
        z = transform.location.z
        pitch = transform.rotation.pitch
        yaw = transform.rotation.yaw
        if i == 0:
            s = 'start'
            x_s = [x]
            y_s = [y]
        elif i == n-1:
            s = 'end'
            x_e = [x]
            y_e = [y]
        else:
            s = 'point'
            x_list.append(x)
            y_list.append(y)

        # print(s, x, y, z, pitch, yaw, command

    import matplotlib.pyplot as plt
    plt.gca().invert_yaxis()
    plt.scatter(x_list, y_list)
    plt.scatter(x_s, y_s, c='red', linewidths=5)
    plt.scatter(x_e, y_e, c='black', linewidths=5)

    plt.show()


def perturb_route(route, perturbation):
    num_to_perturb = min([len(route), len(perturbation)+2])
    for i in range(num_to_perturb):
        if i != 0 and i != num_to_perturb-1:
            route[i][0].location.x += perturbation[i-1][0]
            route[i][0].location.y += perturbation[i-1][1]


def create_transform(x, y, z, pitch, yaw, roll):
    location = carla.Location(x, y, z)
    rotation = carla.Rotation(pitch, yaw, roll)
    transform = carla.Transform(location, rotation)
    return transform

def copy_transform(t):
    return create_transform(t.location.x, t.location.y, t.location.z, t.rotation.pitch, t.rotation.yaw, t.rotation.roll)


def rand_real(rng, low, high):
    return rng.random()*(high-low)+low


def specify_args():
    # general parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--sync', action='store_true',
                        help='Forces the simulation to run synchronously')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--spectator', type=bool, help='Switch spectator view on?', default=True)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    # modification: 30->40
    parser.add_argument('--timeout', default="15.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--challenge-mode', action="store_true", help='Switch to challenge mode?')
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=False)
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=False)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=False)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    # addition
    parser.add_argument("--weather-index", type=int, default=0, help="see WEATHER for reference")
    parser.add_argument("--save-folder", type=str, default='collected_data', help="Path to save simulation data")
    parser.add_argument("--deviations-folder", type=str, default='', help="Path to the folder that saves deviations data")
    parser.add_argument("--save_action_based_measurements", type=int, default=0)
    parser.add_argument("--changing_weather", type=int, default=0)

    arguments = parser.parse_args()

    return arguments




class arguments_info:
    def __init__(self):
        self.host = 'localhost'
        self.port = '2000'
        self.sync = False
        self.debug = 0
        self.spectator = True
        self.record = ''
        self.timeout = '15.0'
        self.challenge_mode = True
        self.routes = None
        self.scenarios = 'leaderboard/data/all_towns_traffic_scenarios_public.json'
        self.repetitions = 1
        self.agent = 'scenario_runner/team_code/image_agent.py'
        self.agent_config = 'models/epoch=24.ckpt'
        self.track = 'SENSORS'
        self.resume = False
        self.checkpoint = ''
        self.weather_index = 19
        self.save_folder = 'collected_data_customized'
        self.deviations_folder = ''
        self.background_vehicles = False
        self.save_action_based_measurements = 0
        self.changing_weather = False







def add_transform(transform1, transform2):
    x = transform1.location.x + transform2.location.x
    y = transform1.location.y + transform2.location.y
    z = transform1.location.z + transform2.location.z
    pitch = transform1.rotation.pitch + transform2.rotation.pitch
    yaw = transform1.rotation.yaw + transform2.rotation.yaw
    roll = transform1.rotation.roll + transform2.rotation.roll
    return create_transform(x, y, z, pitch, yaw, roll)




def convert_x_to_customized_data(x, waypoints_num_limit, max_num_of_static, max_num_of_pedestrians, max_num_of_vehicles, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms, parameters_min_bounds, parameters_max_bounds):

    # parameters
    # global
    friction = x[0]
    weather_index = int(x[1])
    num_of_static = int(x[2])
    num_of_pedestrians = int(x[3])
    num_of_vehicles = int(x[4])

    ind = 5

    # if use_fine_grained_weather:
    if weather_index == -1:
        fine_grained_weather = carla.WeatherParameters(x[ind], x[ind+1], x[ind+2], x[ind+3], x[ind+4], x[ind+5], x[ind+6], x[ind+7], x[ind+8], x[ind+9])
        print('weather params:', x[ind], x[ind+1], x[ind+2], x[ind+3], x[ind+4], x[ind+5], x[ind+6], x[ind+7], x[ind+8], x[ind+9])
        ind += 10
    else:
        fine_grained_weather = None

    # ego car
    ego_car_waypoints_perturbation = []
    for _ in range(waypoints_num_limit):
        dx = x[ind]
        dy = x[ind+1]
        ego_car_waypoints_perturbation.append([dx, dy])
        ind += 2

    # static
    static_list = []
    for i in range(max_num_of_static):
        if i < num_of_static:
            static_type_i = static_types[int(x[ind])]
            static_transform_i = create_transform(x[ind+1], x[ind+2], 0, 0, x[ind+3], 0)
            static_i = Static(model=static_type_i, spawn_transform=static_transform_i)
            static_list.append(static_i)
        ind += 4

    # pedestrians
    pedestrian_list = []
    for i in range(max_num_of_pedestrians):
        if i < num_of_pedestrians:
            pedestrian_type_i = pedestrian_types[int(x[ind])]
            pedestrian_transform_i = create_transform(x[ind+1], x[ind+2], 0, 0, x[ind+3], 0)
            pedestrian_i = Pedestrian(model=pedestrian_type_i, spawn_transform=pedestrian_transform_i, trigger_distance=x[ind+4], speed=x[ind+5], dist_to_travel=x[ind+6], after_trigger_behavior='stop')
            pedestrian_list.append(pedestrian_i)
        ind += 7

    # vehicles
    vehicle_list = []
    for i in range(max_num_of_vehicles):
        if i < num_of_vehicles:
            vehicle_type_i = vehicle_types[int(x[ind])]

            vehicle_transform_i = create_transform(x[ind+1], x[ind+2], 0, 0, x[ind+3], 0)

            vehicle_initial_speed_i = x[ind+4]
            vehicle_trigger_distance_i = x[ind+5]

            vehicle_targeted_speed_i = x[ind+6]
            vehicle_waypoint_follower_i = bool(x[ind+7])

            vehicle_targeted_waypoint_i = create_transform(x[ind+8], x[ind+9], 0, 0, 0, 0)

            vehicle_avoid_collision_i = bool(x[ind+10])
            vehicle_dist_to_travel_i = x[ind+11]
            vehicle_target_yaw_i = x[ind+12]
            x_dir = np.cos(np.deg2rad(vehicle_target_yaw_i))
            y_dir = np.sin(np.deg2rad(vehicle_target_yaw_i))
            vehicle_target_direction_i = carla.Vector3D(x_dir, y_dir, 0)

            vehicle_color_i = vehicle_colors[int(x[ind+13])]

            ind += 14

            vehicle_waypoints_perturbation_i = []
            for _ in range(waypoints_num_limit):
                dx = x[ind]
                dy = x[ind+1]
                vehicle_waypoints_perturbation_i.append([dx, dy])
                ind += 2

            vehicle_i = Vehicle(model=vehicle_type_i, spawn_transform=vehicle_transform_i, avoid_collision=vehicle_avoid_collision_i, initial_speed=vehicle_initial_speed_i, trigger_distance=vehicle_trigger_distance_i, waypoint_follower=vehicle_waypoint_follower_i, targeted_waypoint=vehicle_targeted_waypoint_i, dist_to_travel=vehicle_dist_to_travel_i,
            target_direction=vehicle_target_direction_i,
            targeted_speed=vehicle_targeted_speed_i, after_trigger_behavior='stop', color=vehicle_color_i, waypoints_perturbation=vehicle_waypoints_perturbation_i)
            # print('\n'*3, 'vehicle', i, vehicle_transform_i, vehicle_avoid_collision_i, vehicle_initial_speed_i, vehicle_trigger_distance_i, vehicle_waypoint_follower_i, vehicle_targeted_waypoint_i, vehicle_dist_to_travel_i, vehicle_target_direction_i, vehicle_targeted_speed_i, '\n'*3)
            vehicle_list.append(vehicle_i)
        else:
            ind += 14 + waypoints_num_limit*2


    # for parallel simulation
    port = int(x[ind])

    customized_data = {
    'friction': friction,
    'weather_index': weather_index,
    'num_of_static': num_of_static,
    'num_of_pedestrians': num_of_pedestrians,
    'num_of_vehicles': num_of_vehicles,
    'static_list': static_list,
    'pedestrian_list': pedestrian_list,
    'vehicle_list': vehicle_list,
    'using_customized_route_and_scenario': True,
    'ego_car_waypoints_perturbation': ego_car_waypoints_perturbation,
    'add_center': True,
    'port': port,
    'customized_center_transforms': customized_center_transforms,
    'parameters_min_bounds': parameters_min_bounds,
    'parameters_max_bounds': parameters_max_bounds,
    'fine_grained_weather': fine_grained_weather,
    'tmp_travel_dist_file': 'tmp_travel_dist_file_'+str(port)+'.txt'}


    return customized_data



def interpret_x_using_labels(x, labels):
    assert len(x) == len(labels)
    for i in range(len(x)):
        print(labels[i], x[i])



def make_hierarchical_dir(folder_names):
    cur_folder_name = ''
    for i in range(len(folder_names)):
        cur_folder_name += folder_names[i]
        if not os.path.exists(cur_folder_name):
            os.mkdir(cur_folder_name)
        cur_folder_name += '/'
    return cur_folder_name

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', int(port))) == 0


def exit_handler(ports):
    for port in ports:
        while is_port_in_use(port):
            try:
                subprocess.run('kill $(lsof -t -i :'+str(port)+')', shell=True)
                print('-'*20, 'kill server at port', port)
            except:
                continue


def get_angle(x1, y1, x2, y2):
    angle = np.arctan2(x1*y2-y1*x2, x1*x2+y1*y2)

    return angle


# check if x is in critical regions of the tree
def is_critical_region(x, estimator, critical_unique_leaves):
    leave_id = estimator.apply(x.reshape(1, -1))[0]
    print(leave_id, critical_unique_leaves)
    return leave_id in critical_unique_leaves


def filter_critical_regions(X, y):
    print('\n'*20)
    print('+'*100, 'filter_critical_regions', '+'*100)

    min_samples_split = np.max([int(0.1*X.shape[0]), 2])
    # estimator = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, min_impurity_decrease=0.01, random_state=0)
    estimator = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, min_impurity_decrease=0.0001, random_state=0)
    print(X.shape, y.shape, X, y)
    estimator = estimator.fit(X, y)

    leave_ids = estimator.apply(X)
    print('leave_ids', leave_ids)

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
        print('unique_leaves', unique_leave_i, unique_leaves_bug_num[i],  unique_leaves_normal_num[i])

    critical_unique_leaves = unique_leave_ids[unique_leaves_bug_num >= unique_leaves_normal_num]

    print('critical_unique_leaves', critical_unique_leaves)


    inds = np.array([leave_id in critical_unique_leaves for leave_id in leave_ids])
    print('\n'*20)

    return estimator, inds, critical_unique_leaves

# hack:
general_labels = ['friction', 'num_of_weathers', 'num_of_static', 'num_of_pedestrians', 'num_of_vehicles']

weather_labels = ['cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity', 'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density', 'fog_distance', 'wetness', 'fog_falloff']

# number of waypoints to perturb
waypoints_num_limit = 0

waypoint_labels = ['perturbation_x', 'perturbation_y']

static_general_labels = ['num_of_static_types', 'static_x', 'static_y', 'static_yaw']

pedestrian_general_labels = ['num_of_pedestrian_types', 'pedestrian_x', 'pedestrian_y', 'pedestrian_yaw', 'pedestrian_trigger_distance', 'pedestrian_speed', 'pedestrian_dist_to_travel']

vehicle_general_labels = ['num_of_vehicle_types', 'vehicle_x', 'vehicle_y', 'vehicle_yaw', 'vehicle_initial_speed', 'vehicle_trigger_distance', 'vehicle_targeted_speed', 'vehicle_waypoint_follower', 'vehicle_targeted_x', 'vehicle_targeted_y', 'vehicle_avoid_collision', 'vehicle_dist_to_travel', 'vehicle_targeted_yaw', 'num_of_vehicle_colors']






def setup_bounds_mask_labels_distributions_stage1(use_fine_grained_weather=False):

    parameters_min_bounds = OrderedDict()
    parameters_max_bounds = OrderedDict()
    mask = []
    labels = []



    fixed_hyperparameters = {
        'num_of_weathers': len(WEATHERS),
        'num_of_static_types': len(static_types),
        'num_of_pedestrian_types': len(pedestrian_types),
        'num_of_vehicle_types': len(vehicle_types),
        'num_of_vehicle_colors': len(vehicle_colors),
        'waypoints_num_limit': waypoints_num_limit
    }


    general_min = [0.5, 0, 0, 0, 0]
    general_max = [0.9, fixed_hyperparameters['num_of_weathers']-1, 2, 2, 2]
    general_mask = ['real', 'int', 'int', 'int', 'int']



    if use_fine_grained_weather:
        general_min[1] = -1
        general_max[1] = -1


    # general
    mask.extend(general_mask)
    for j in range(len(general_labels)):
        general_label = general_labels[j]
        k_min = '_'.join([general_label, 'min'])
        k_max = '_'.join([general_label, 'max'])
        k = '_'.join([general_label])

        labels.append(k)
        parameters_min_bounds[k_min] = general_min[j]
        parameters_max_bounds[k_max] = general_max[j]



    if use_fine_grained_weather:
        weather_min = [0, 0, 0, 0, 0, -90, 0, 0, 0, 0]
        weather_max = [100, 80, 80, 50, 360, 90, 15, 100, 40, 2]
        # [100, 100, 100, 100, 360, 90, 100, 100, inf, 5]
        weather_mask = ['real']*10


        mask.extend(weather_mask)
        for j in range(len(weather_labels)):
            weather_label = weather_labels[j]
            k_min = '_'.join([weather_label, 'min'])
            k_max = '_'.join([weather_label, 'max'])
            k = '_'.join([weather_label])

            labels.append(k)
            parameters_min_bounds[k_min] = weather_min[j]
            parameters_max_bounds[k_max] = weather_max[j]



    return fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels






# Set up default bounds, mask, labels, and distributions for a Problem object
def setup_bounds_mask_labels_distributions_stage2(fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels):

    waypoint_min = [-0.5, 0.5]
    waypoint_max = [0.5, 0.5]
    waypoint_mask = ['real', 'real']


    static_general_min = [0, -20, -20, 0]
    static_general_max = [fixed_hyperparameters['num_of_static_types']-1, 20, 20, 360]
    static_mask = ['int'] + ['real']*3

    # pedestrian activation threshold: 2->8
    pedestrian_general_min = [0, -20, -20, 0, 10, 0, 0]
    pedestrian_general_max = [fixed_hyperparameters['num_of_pedestrian_types']-1, 20, 20, 360, 50, 4, 50]
    pedestrian_mask = ['int'] + ['real']*6

    # vehicle activation threshold: 0->10
    vehicle_general_min = [0, -20, -20, 0, 0, 10, 0, 0, -20, -20, 0, 0, 0, 0]
    vehicle_general_max = [fixed_hyperparameters['num_of_vehicle_types']-1, 20, 20, 360, 10, 50, 10, 1, 20, 20, 1, 50, 360, fixed_hyperparameters['num_of_vehicle_colors']-1]
    vehicle_mask = ['int'] + ['real']*6 + ['int'] + ['real']*2 + ['int'] + ['real']*2 + ['int']




    assert len(waypoint_min) == len(waypoint_max)
    assert len(waypoint_min) == len(waypoint_mask)
    assert len(waypoint_mask) == len(waypoint_labels)

    assert len(static_general_min) == len(static_general_max)
    assert len(static_general_min) == len(static_mask)
    assert len(static_mask) == len(static_general_labels)

    assert len(pedestrian_general_min) == len(pedestrian_general_max)
    assert len(pedestrian_general_min) == len(pedestrian_mask)
    assert len(pedestrian_mask) == len(pedestrian_general_labels)

    assert len(vehicle_general_min) == len(vehicle_general_max)
    assert len(vehicle_general_min) == len(vehicle_mask)
    assert len(vehicle_mask) == len(vehicle_general_labels)





    # ego_car waypoint
    for i in range(fixed_hyperparameters['waypoints_num_limit']):
        mask.extend(waypoint_mask)

        for j in range(len(waypoint_labels)):
            waypoint_label = waypoint_labels[j]
            k_min = '_'.join(['ego_car', waypoint_label, 'min', str(i)])
            k_max = '_'.join(['ego_car', waypoint_label, 'max', str(i)])
            k = '_'.join(['ego_car', waypoint_label, str(i)])

            labels.append(k)
            parameters_min_bounds[k_min] = waypoint_min[j]
            parameters_max_bounds[k_max] = waypoint_max[j]


    # static
    for i in range(parameters_max_bounds['num_of_static_max']):
        mask.extend(static_mask)

        for j in range(len(static_general_labels)):
            static_general_label = static_general_labels[j]
            k_min = '_'.join([static_general_label, 'min', str(i)])
            k_max = '_'.join([static_general_label, 'max', str(i)])
            k = '_'.join([static_general_label, str(i)])

            labels.append(k)
            parameters_min_bounds[k_min] = static_general_min[j]
            parameters_max_bounds[k_max] = static_general_max[j]


    # pedestrians
    for i in range(parameters_max_bounds['num_of_pedestrians_max']):
        mask.extend(pedestrian_mask)

        for j in range(len(pedestrian_general_labels)):
            pedestrian_general_label = pedestrian_general_labels[j]
            k_min = '_'.join([pedestrian_general_label, 'min', str(i)])
            k_max = '_'.join([pedestrian_general_label, 'max', str(i)])
            k = '_'.join([pedestrian_general_label, str(i)])

            labels.append(k)
            parameters_min_bounds[k_min] = pedestrian_general_min[j]
            parameters_max_bounds[k_max] = pedestrian_general_max[j]

    # vehicles
    for i in range(parameters_max_bounds['num_of_vehicles_max']):
        mask.extend(vehicle_mask)

        for j in range(len(vehicle_general_labels)):
            vehicle_general_label = vehicle_general_labels[j]
            k_min = '_'.join([vehicle_general_label, 'min', str(i)])
            k_max = '_'.join([vehicle_general_label, 'max', str(i)])
            k = '_'.join([vehicle_general_label, str(i)])

            labels.append(k)
            parameters_min_bounds[k_min] = vehicle_general_min[j]
            parameters_max_bounds[k_max] = vehicle_general_max[j]

        for p in range(fixed_hyperparameters['waypoints_num_limit']):
            mask.extend(waypoint_mask)

            for q in range(len(waypoint_labels)):
                waypoint_label = waypoint_labels[q]
                k_min = '_'.join(['vehicle', str(i), waypoint_label, 'min', str(p)])
                k_max = '_'.join(['vehicle', str(i), waypoint_label, 'max', str(p)])
                k = '_'.join(['vehicle', str(i), waypoint_label, str(p)])

                labels.append(k)
                parameters_min_bounds[k_min] = waypoint_min[q]
                parameters_max_bounds[k_max] = waypoint_max[q]

    parameters_distributions = OrderedDict()
    for label in labels:
        if 'perturbation' in label:
            parameters_distributions[label] = ('normal', 0, 0.25)
        else:
            parameters_distributions[label] = ('uniform')


    n_var = 5+fixed_hyperparameters['waypoints_num_limit']*2+parameters_max_bounds['num_of_static_max']*4+parameters_max_bounds['num_of_pedestrians_max']*7+parameters_max_bounds['num_of_vehicles_max']*(14+fixed_hyperparameters['waypoints_num_limit']*2)

    return fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels, parameters_distributions, n_var

# Customize parameters
def customize_parameters(parameters, customized_parameters):
    for k, v in customized_parameters.items():
        if k in parameters:
            parameters[k] = v
        else:
            # print(k, 'is not defined in the parameters.')
            pass





'''
customized non-default center transforms for actors
['waypoint_ratio', 'absolute_location']
'''

customized_bounds_and_distributions = {
    'default': {'customized_parameters_bounds':{},
    'customized_parameters_distributions':{},
    'customized_center_transforms':{},
    'customized_constraints':[]},

    'turn_left_town01': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 1,
        'num_of_pedestrians_max': 1,
        'num_of_vehicles_min': 1,
        'num_of_vehicles_max': 1,
    },
    'customized_parameters_distributions':{},
    'customized_center_transforms':{},
    'customized_constraints':[]},

    'go_straight_town07': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 2,
        'num_of_pedestrians_max': 2,
        'num_of_vehicles_min': 2,
        'num_of_vehicles_max': 2,
    },
    'customized_parameters_distributions':{},
    'customized_center_transforms':{},
    'customized_constraints': []
    },


    'leading_car_braking_town05': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 1,
        'num_of_pedestrians_min': 1,
        'num_of_pedestrians_max': 2,
        'num_of_vehicles_min': 1,
        'num_of_vehicles_max': 2,

        'static_y_max_0': 5,

        'vehicle_x_min_0': -0.5,
        'vehicle_x_max_0': 0.5,
        'vehicle_y_min_0': -12,
        'vehicle_y_max_0': -5,

        'vehicle_y_max_1': 5,

        'vehicle_initial_speed_min_0': 2,
        'vehicle_initial_speed_max_0': 5,
        'vehicle_targeted_speed_min_0': 0,
        'vehicle_targeted_speed_max_0': 2,
        'vehicle_trigger_distance_min_0': 5,
        'vehicle_trigger_distance_max_0': 12,

        'vehicle_avoid_collision_min_0': 1,
        'vehicle_avoid_collision_max_0': 1,

        'vehicle_avoid_collision_min_1': 1,
        'vehicle_avoid_collision_max_1': 1,

        'vehicle_dist_to_travel_min_0': 5,
        'vehicle_dist_to_travel_max_0': 30,
        'vehicle_yaw_min_0': 270,
        'vehicle_yaw_max_0': 270
    },
    'customized_parameters_distributions':{
        'vehicle_x_0': ('normal', None, 0.5),
        'vehicle_y_0': ('normal', None, 4)
    },
    'customized_center_transforms':{
        'vehicle_center_transform_0': ('waypoint_ratio', 0)
    },
    'customized_constraints': [
    {'coefficients': [1, 1],
    'labels': ['vehicle_y_0', 'vehicle_trigger_distance_0'],
    'value': 0}
    ]
    },

    'leading_car_braking_town05_fixed_npc_num': {'customized_parameters_bounds':{

        'friction_min': 0.9,
        'friction_max': 0.9,
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 1,
        'num_of_pedestrians_max': 1,
        'num_of_vehicles_min': 1,
        'num_of_vehicles_max': 1,


        'vehicle_x_min_0': -0.5,
        'vehicle_x_max_0': 0.5,
        'vehicle_y_min_0': -12,
        'vehicle_y_max_0': -5,


        'vehicle_initial_speed_min_0': 2,
        'vehicle_initial_speed_max_0': 5,
        'vehicle_targeted_speed_min_0': 0,
        'vehicle_targeted_speed_max_0': 2,
        'vehicle_trigger_distance_min_0': 5,
        'vehicle_trigger_distance_max_0': 12,

        'vehicle_avoid_collision_min_0': 1,
        'vehicle_avoid_collision_max_0': 1,


        'vehicle_dist_to_travel_min_0': 5,
        'vehicle_dist_to_travel_max_0': 30,
        'vehicle_yaw_min_0': 270,
        'vehicle_yaw_max_0': 270
    },
    'customized_parameters_distributions':{
        'vehicle_x_0': ('normal', None, 0.5),
        'vehicle_y_0': ('normal', None, 4)
    },
    'customized_center_transforms':{
        'vehicle_center_transform_0': ('waypoint_ratio', 0)
    },
    'customized_constraints': [
    {'coefficients': [1, 1],
    'labels': ['vehicle_y_0', 'vehicle_trigger_distance_0'],
    'value': 0}
    ]
    },


    'change_lane_town05': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 1,
        'num_of_pedestrians_min': 0,
        'num_of_pedestrians_max': 1,
        'num_of_vehicles_min': 1,
        'num_of_vehicles_max': 3,


        'vehicle_x_min_0': -4,
        'vehicle_x_max_0': -3,
        'vehicle_y_min_0': -20,
        'vehicle_y_max_0': 20,
        'vehicle_yaw_min_0': 270,
        'vehicle_yaw_max_0': 270,
        'vehicle_initial_speed_min_0': 3,
        'vehicle_initial_speed_max_0': 7,
        'vehicle_trigger_distance_min_0': 0,
        'vehicle_trigger_distance_max_0': 0,
        'vehicle_dist_to_travel_min_0': 30,
        'vehicle_dist_to_travel_max_0': 50,

    },
    'customized_parameters_distributions':{
    },
    'customized_center_transforms':{
        'vehicle_center_transform_0': ('waypoint_ratio', 0)
    },
    'customized_constraints': []
    },


    'change_lane_town05_fixed_npc_num': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 1,
        'num_of_pedestrians_max': 1,
        'num_of_vehicles_min': 1,
        'num_of_vehicles_max': 1,


        'vehicle_x_min_0': -4,
        'vehicle_x_max_0': -3,
        'vehicle_y_min_0': -20,
        'vehicle_y_max_0': -1,
        'vehicle_yaw_min_0': 270,
        'vehicle_yaw_max_0': 270,
        'vehicle_initial_speed_min_0': 3,
        'vehicle_initial_speed_max_0': 7,
        'vehicle_trigger_distance_min_0': 0,
        'vehicle_trigger_distance_max_0': 0,
        'vehicle_dist_to_travel_min_0': 15,
        'vehicle_dist_to_travel_max_0': 40,
        'vehicle_avoid_collision_min_0': 1,
        'vehicle_avoid_collision_max_0': 1,

    },
    'customized_parameters_distributions':{
    },
    'customized_center_transforms':{
        'vehicle_center_transform_0': ('waypoint_ratio', 0),
        'pedestrian_center_transform_0': ('waypoint_ratio', 20)
    },
    'customized_constraints': []
    },


    'pedestrians_cross_street_town04': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 10,
        'num_of_pedestrians_max': 10,
        'num_of_vehicles_min': 10,
        'num_of_vehicles_max': 10,


        'vehicle_waypoint_follower_min_0': 1,
        'vehicle_waypoint_follower_max_0': 1,

        'vehicle_waypoint_follower_min_1': 1,
        'vehicle_waypoint_follower_max_1': 1,

        'vehicle_waypoint_follower_min_2': 1,
        'vehicle_waypoint_follower_max_2': 1,

        'vehicle_waypoint_follower_min_3': 1,
        'vehicle_waypoint_follower_max_3': 1,

        'vehicle_waypoint_follower_min_4': 1,
        'vehicle_waypoint_follower_max_4': 1,

        'vehicle_waypoint_follower_min_5': 1,
        'vehicle_waypoint_follower_max_5': 1,

        'vehicle_waypoint_follower_min_6': 1,
        'vehicle_waypoint_follower_max_6': 1,

        'vehicle_waypoint_follower_min_7': 1,
        'vehicle_waypoint_follower_max_7': 1,

        'vehicle_waypoint_follower_min_8': 1,
        'vehicle_waypoint_follower_max_8': 1,

        'vehicle_waypoint_follower_min_9': 1,
        'vehicle_waypoint_follower_max_9': 1,


        'vehicle_avoid_collision_min_0': 1,
        'vehicle_avoid_collision_max_0': 1,

        'vehicle_avoid_collision_min_1': 1,
        'vehicle_avoid_collision_max_1': 1,

        'vehicle_avoid_collision_min_2': 1,
        'vehicle_avoid_collision_max_2': 1,

        'vehicle_avoid_collision_min_3': 1,
        'vehicle_avoid_collision_max_3': 1,

        'vehicle_avoid_collision_min_4': 1,
        'vehicle_avoid_collision_max_4': 1,

        'vehicle_avoid_collision_min_5': 1,
        'vehicle_avoid_collision_max_5': 1,

        'vehicle_avoid_collision_min_6': 1,
        'vehicle_avoid_collision_max_6': 1,

        'vehicle_avoid_collision_min_7': 1,
        'vehicle_avoid_collision_max_7': 1,

        'vehicle_avoid_collision_min_8': 1,
        'vehicle_avoid_collision_max_8': 1,

        'vehicle_avoid_collision_min_9': 1,
        'vehicle_avoid_collision_max_9': 1,


        'pedestrian_x_min_0': -12,
        'pedestrian_x_max_0': -4,
        'pedestrian_y_min_0': -20,
        'pedestrian_y_max_0': -10,
        'pedestrian_yaw_min_0': -45,
        'pedestrian_yaw_max_0': 45,
        'pedestrian_speed_min_0': 1,
        'pedestrian_speed_max_0': 5,
        'pedestrian_trigger_distance_min_0': 10,
        'pedestrian_trigger_distance_max_0': 20,
        'pedestrian_dist_to_travel_min_0': 5,
        'pedestrian_dist_to_travel_max_0': 30,

        'pedestrian_x_min_1': -12,
        'pedestrian_x_max_1': -4,
        'pedestrian_y_min_1': -20,
        'pedestrian_y_max_1': -10,
        'pedestrian_yaw_min_1': -45,
        'pedestrian_yaw_max_1': 45,
        'pedestrian_speed_min_1': 1,
        'pedestrian_speed_max_1': 5,
        'pedestrian_trigger_distance_min_1': 10,
        'pedestrian_trigger_distance_max_1': 20,
        'pedestrian_dist_to_travel_min_1': 5,
        'pedestrian_dist_to_travel_max_1': 30,

        'pedestrian_x_min_2': -12,
        'pedestrian_x_max_2': -4,
        'pedestrian_y_min_2': -20,
        'pedestrian_y_max_2': -10,
        'pedestrian_yaw_min_2': -45,
        'pedestrian_yaw_max_2': 45,
        'pedestrian_speed_min_2': 1,
        'pedestrian_speed_max_2': 5,
        'pedestrian_trigger_distance_min_2': 10,
        'pedestrian_trigger_distance_max_2': 20,
        'pedestrian_dist_to_travel_min_2': 5,
        'pedestrian_dist_to_travel_max_2': 30,

        'pedestrian_x_min_3': -12,
        'pedestrian_x_max_3': -4,
        'pedestrian_y_min_3': -20,
        'pedestrian_y_max_3': -10,
        'pedestrian_yaw_min_3': -45,
        'pedestrian_yaw_max_3': 45,
        'pedestrian_speed_min_3': 1,
        'pedestrian_speed_max_3': 5,
        'pedestrian_trigger_distance_min_3': 10,
        'pedestrian_trigger_distance_max_3': 20,
        'pedestrian_dist_to_travel_min_3': 5,
        'pedestrian_dist_to_travel_max_3': 30,

        'pedestrian_x_min_4': -12,
        'pedestrian_x_max_4': -4,
        'pedestrian_y_min_4': -20,
        'pedestrian_y_max_4': -10,
        'pedestrian_yaw_min_4': -45,
        'pedestrian_yaw_max_4': 45,
        'pedestrian_speed_min_4': 1,
        'pedestrian_speed_max_4': 5,
        'pedestrian_trigger_distance_min_4': 10,
        'pedestrian_trigger_distance_max_4': 20,
        'pedestrian_dist_to_travel_min_4': 5,
        'pedestrian_dist_to_travel_max_4': 30,

        'pedestrian_x_min_5': 2,
        'pedestrian_x_max_5': 8,
        'pedestrian_y_min_5': -20,
        'pedestrian_y_max_5': -10,
        'pedestrian_yaw_min_5': 135,
        'pedestrian_yaw_max_5': 225,
        'pedestrian_speed_min_5': 1,
        'pedestrian_speed_max_5': 5,
        'pedestrian_trigger_distance_min_5': 10,
        'pedestrian_trigger_distance_max_5': 20,
        'pedestrian_dist_to_travel_min_5': 5,
        'pedestrian_dist_to_travel_max_5': 30,

        'pedestrian_x_min_6': 2,
        'pedestrian_x_max_6': 8,
        'pedestrian_y_min_6': -20,
        'pedestrian_y_max_6': -10,
        'pedestrian_yaw_min_6': 135,
        'pedestrian_yaw_max_6': 225,
        'pedestrian_speed_min_6': 1,
        'pedestrian_speed_max_6': 5,
        'pedestrian_trigger_distance_min_6': 10,
        'pedestrian_trigger_distance_max_6': 20,
        'pedestrian_dist_to_travel_min_6': 5,
        'pedestrian_dist_to_travel_max_6': 30,

        'pedestrian_x_min_7': 2,
        'pedestrian_x_max_7': 8,
        'pedestrian_y_min_7': -20,
        'pedestrian_y_max_7': -10,
        'pedestrian_yaw_min_7': 135,
        'pedestrian_yaw_max_7': 225,
        'pedestrian_speed_min_7': 1,
        'pedestrian_speed_max_7': 5,
        'pedestrian_trigger_distance_min_7': 10,
        'pedestrian_trigger_distance_max_7': 20,
        'pedestrian_dist_to_travel_min_7': 5,
        'pedestrian_dist_to_travel_max_7': 30,

        'pedestrian_x_min_8': 2,
        'pedestrian_x_max_8': 8,
        'pedestrian_y_min_8': -20,
        'pedestrian_y_max_8': -10,
        'pedestrian_yaw_min_8': 135,
        'pedestrian_yaw_max_8': 225,
        'pedestrian_speed_min_8': 1,
        'pedestrian_speed_max_8': 5,
        'pedestrian_trigger_distance_min_8': 10,
        'pedestrian_trigger_distance_max_8': 20,
        'pedestrian_dist_to_travel_min_8': 5,
        'pedestrian_dist_to_travel_max_8': 30,

        'pedestrian_x_min_9': 2,
        'pedestrian_x_max_9': 8,
        'pedestrian_y_min_9': -20,
        'pedestrian_y_max_9': -10,
        'pedestrian_yaw_min_9': 135,
        'pedestrian_yaw_max_9': 225,
        'pedestrian_speed_min_9': 1,
        'pedestrian_speed_max_9': 5,
        'pedestrian_trigger_distance_min_9': 10,
        'pedestrian_trigger_distance_max_9': 20,
        'pedestrian_dist_to_travel_min_9': 5,
        'pedestrian_dist_to_travel_max_9': 30,

    },
    'customized_parameters_distributions':{},
    'customized_center_transforms':{
        'pedestrian_center_transform_0': ('waypoint_ratio', 0),
        'pedestrian_center_transform_1': ('waypoint_ratio', 0),
        'pedestrian_center_transform_2': ('waypoint_ratio', 0),
        'pedestrian_center_transform_3': ('waypoint_ratio', 0),
        'pedestrian_center_transform_4': ('waypoint_ratio', 0),
        'pedestrian_center_transform_5': ('waypoint_ratio', 0),
        'pedestrian_center_transform_6': ('waypoint_ratio', 0),
        'pedestrian_center_transform_7': ('waypoint_ratio', 0),
        'pedestrian_center_transform_8': ('waypoint_ratio', 0),
        'pedestrian_center_transform_9': ('waypoint_ratio', 0),
    },
    'customized_constraints': []
    },





    'highway_town04': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 0,
        'num_of_pedestrians_max': 0,
        'num_of_vehicles_min': 2,
        'num_of_vehicles_max': 5,
        'vehicle_x_min_0': 0,
        'vehicle_x_max_0': 6,
        'vehicle_y_min_0': -3,
        'vehicle_y_max_0': -10,
        'vehicle_yaw_min_0': 270,
        'vehicle_yaw_max_0': 270,
        'vehicle_initial_speed_min_0': 4,
        'vehicle_initial_speed_max_0': 10,
        'vehicle_targeted_speed_min_0': 4,
        'vehicle_targeted_speed_max_0': 10,
        'vehicle_trigger_distance_min_0': 15,
        'vehicle_trigger_distance_max_0': 15,
        'vehicle_dist_to_travel_min_0': 5,
        'vehicle_dist_to_travel_max_0': 30,

        'vehicle_x_min_1': 0,
        'vehicle_x_max_1': 6,
        'vehicle_y_min_1': -3,
        'vehicle_y_max_1': -10,
        'vehicle_yaw_min_1': 270,
        'vehicle_yaw_max_1': 270,
        'vehicle_initial_speed_min_1': 4,
        'vehicle_initial_speed_max_1': 10,
        'vehicle_targeted_speed_min_1': 4,
        'vehicle_targeted_speed_max_1': 10,
        'vehicle_trigger_distance_min_1': 15,
        'vehicle_trigger_distance_max_1': 15,
        'vehicle_dist_to_travel_min_1': 5,
        'vehicle_dist_to_travel_max_1': 30,


    },
    'customized_parameters_distributions':{},
    'customized_center_transforms':{
        'vehicle_center_transform_0': ('waypoint_ratio', 0),
        'vehicle_center_transform_1': ('waypoint_ratio', 0)
    },
    'customized_constraints':[]},


    'no_static': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
    },
    'customized_parameters_distributions':{},
    'customized_center_transforms':{},
    'customized_constraints':[]},


    'one_pedestrians_cross_street_town05': {'customized_parameters_bounds':{
        'num_of_weathers_min': 0,
        'num_of_weathers_max': 0,

        'num_of_pedestrian_types_min_0': 12,
        'num_of_pedestrian_types_max_0': 12,

        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 1,
        'num_of_pedestrians_max': 1,
        'num_of_vehicles_min': 0,
        'num_of_vehicles_max': 0,


        'pedestrian_x_min_0': -8,
        'pedestrian_x_max_0': 8,
        'pedestrian_y_min_0': -8,
        'pedestrian_y_max_0': 8,
        'pedestrian_speed_min_0': 1,
        'pedestrian_speed_max_0': 4,
        'pedestrian_trigger_distance_min_0': 3,
        'pedestrian_trigger_distance_max_0': 15,
        'pedestrian_dist_to_travel_min_0': 30,
        'pedestrian_dist_to_travel_max_0': 30,



        'friction_min': 0.9,
        'friction_max': 0.9,


        'ego_car_perturbation_x_min_0': 0,
        'ego_car_perturbation_x_max_0': 0,
        'ego_car_perturbation_x_min_1': 0,
        'ego_car_perturbation_x_max_1': 0,
        'ego_car_perturbation_x_min_2': 0,
        'ego_car_perturbation_x_max_2': 0,
        'ego_car_perturbation_x_min_3': 0,
        'ego_car_perturbation_x_max_3': 0,
        'ego_car_perturbation_x_min_4': 0,
        'ego_car_perturbation_x_max_4': 0,

        'ego_car_perturbation_y_min_0': 0,
        'ego_car_perturbation_y_max_0': 0,
        'ego_car_perturbation_y_min_1': 0,
        'ego_car_perturbation_y_max_1': 0,
        'ego_car_perturbation_y_min_2': 0,
        'ego_car_perturbation_y_max_2': 0,
        'ego_car_perturbation_y_min_3': 0,
        'ego_car_perturbation_y_max_3': 0,
        'ego_car_perturbation_y_min_4': 0,
        'ego_car_perturbation_y_max_4': 0,

    },
    'customized_parameters_distributions':{},
    'customized_center_transforms':{
        'pedestrian_center_transform_0': ('waypoint_ratio', 70),
    },
    'customized_constraints': []
    },


    'two_pedestrians_cross_street_town05': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 2,
        'num_of_pedestrians_max': 2,
        'num_of_vehicles_min': 0,
        'num_of_vehicles_max': 0,

        'num_of_weathers_min': 0,
        'num_of_weathers_max': 0,

        'pedestrian_x_min_0': -8,
        'pedestrian_x_max_0': 8,
        'pedestrian_y_min_0': -8,
        'pedestrian_y_max_0': 8,
        'pedestrian_speed_min_0': 1,
        'pedestrian_speed_max_0': 5,
        'pedestrian_trigger_distance_min_0': 3,
        'pedestrian_trigger_distance_max_0': 15,
        'pedestrian_dist_to_travel_min_0': 5,
        'pedestrian_dist_to_travel_max_0': 30,

        'pedestrian_x_min_1': -8,
        'pedestrian_x_max_1': 8,
        'pedestrian_y_min_1': -8,
        'pedestrian_y_max_1': 8,
        'pedestrian_speed_min_1': 1,
        'pedestrian_speed_max_1': 5,
        'pedestrian_trigger_distance_min_1': 3,
        'pedestrian_trigger_distance_max_1': 15,
        'pedestrian_dist_to_travel_min_1': 5,
        'pedestrian_dist_to_travel_max_1': 30,


        'friction_min': 0.9,
        'friction_max': 0.9,


        'ego_car_perturbation_x_min_0': 0,
        'ego_car_perturbation_x_max_0': 0,
        'ego_car_perturbation_x_min_1': 0,
        'ego_car_perturbation_x_max_1': 0,
        'ego_car_perturbation_x_min_2': 0,
        'ego_car_perturbation_x_max_2': 0,
        'ego_car_perturbation_x_min_3': 0,
        'ego_car_perturbation_x_max_3': 0,
        'ego_car_perturbation_x_min_4': 0,
        'ego_car_perturbation_x_max_4': 0,

        'ego_car_perturbation_y_min_0': 0,
        'ego_car_perturbation_y_max_0': 0,
        'ego_car_perturbation_y_min_1': 0,
        'ego_car_perturbation_y_max_1': 0,
        'ego_car_perturbation_y_min_2': 0,
        'ego_car_perturbation_y_max_2': 0,
        'ego_car_perturbation_y_min_3': 0,
        'ego_car_perturbation_y_max_3': 0,
        'ego_car_perturbation_y_min_4': 0,
        'ego_car_perturbation_y_max_4': 0,

    },
    'customized_parameters_distributions':{},
    'customized_center_transforms':{
        'pedestrian_center_transform_0': ('waypoint_ratio', 40),
        'pedestrian_center_transform_1': ('waypoint_ratio', 40),
    },
    'customized_constraints': []
    },


    'default_dense': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 5,
        'num_of_pedestrians_max': 10,
        'num_of_vehicles_min': 10,
        'num_of_vehicles_max': 10,

        'friction_min': 0.9,
        'friction_max': 0.9,

        'vehicle_waypoint_follower_min_0': 1,
        'vehicle_waypoint_follower_max_0': 1,
        'vehicle_avoid_collision_min_0': 1,
        'vehicle_avoid_collision_max_0': 1,
        'vehicle_waypoint_follower_min_1': 1,
        'vehicle_waypoint_follower_max_1': 1,
        'vehicle_avoid_collision_min_1': 1,
        'vehicle_avoid_collision_max_1': 1,
        'vehicle_waypoint_follower_min_2': 1,
        'vehicle_waypoint_follower_max_2': 1,
        'vehicle_avoid_collision_min_2': 1,
        'vehicle_avoid_collision_max_2': 1,
        'vehicle_waypoint_follower_min_3': 1,
        'vehicle_waypoint_follower_max_3': 1,
        'vehicle_avoid_collision_min_3': 1,
        'vehicle_avoid_collision_max_3': 1,
        'vehicle_waypoint_follower_min_4': 1,
        'vehicle_waypoint_follower_max_4': 1,
        'vehicle_avoid_collision_min_4': 1,
        'vehicle_avoid_collision_max_4': 1,
        'vehicle_waypoint_follower_min_5': 1,
        'vehicle_waypoint_follower_max_5': 1,
        'vehicle_avoid_collision_min_5': 1,
        'vehicle_avoid_collision_max_5': 1,
        'vehicle_waypoint_follower_min_6': 1,
        'vehicle_waypoint_follower_max_6': 1,
        'vehicle_avoid_collision_min_6': 1,
        'vehicle_avoid_collision_max_6': 1,
        'vehicle_waypoint_follower_min_7': 1,
        'vehicle_waypoint_follower_max_7': 1,
        'vehicle_avoid_collision_min_7': 1,
        'vehicle_avoid_collision_max_7': 1,
        'vehicle_waypoint_follower_min_8': 1,
        'vehicle_waypoint_follower_max_8': 1,
        'vehicle_avoid_collision_min_8': 1,
        'vehicle_avoid_collision_max_8': 1,
        'vehicle_waypoint_follower_min_9': 1,
        'vehicle_waypoint_follower_max_9': 1,
        'vehicle_avoid_collision_min_9': 1,
        'vehicle_avoid_collision_max_9': 1,
        'vehicle_waypoint_follower_min_10': 1,
        'vehicle_waypoint_follower_max_10': 1,
        'vehicle_avoid_collision_min_10': 1,
        'vehicle_avoid_collision_max_10': 1,
    },
    'customized_parameters_distributions':{},
    'customized_center_transforms':{},
    'customized_constraints': []
    },

    'pedestrians_only': {'customized_parameters_bounds':{
        'friction_min': 0.9,
        'friction_max': 0.9,

        'num_of_weathers_min': 0,
        'num_of_weathers_max': 0,

        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 5,
        'num_of_pedestrians_max': 10,
        'num_of_vehicles_min': 0,
        'num_of_vehicles_max': 0,

        'ego_car_perturbation_x_0_min': 0,
        'ego_car_perturbation_x_0_max': 0,
        'ego_car_perturbation_x_1_min': 0,
        'ego_car_perturbation_x_1_max': 0,
        'ego_car_perturbation_x_2_min': 0,
        'ego_car_perturbation_x_2_max': 0,
        'ego_car_perturbation_x_3_min': 0,
        'ego_car_perturbation_x_3_max': 0,
        'ego_car_perturbation_x_4_min': 0,
        'ego_car_perturbation_x_4_max': 0,

    },
    'customized_parameters_distributions':{},
    'customized_center_transforms':{},
    'customized_constraints': []
    },


    'leading_car_braking_only_car_town05': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 0,
        'num_of_pedestrians_max': 0,
        'num_of_vehicles_min': 1,
        'num_of_vehicles_max': 1,

        'vehicle_x_min_0': -0.5,
        'vehicle_x_max_0': 0.5,
        'vehicle_y_min_0': -10,
        'vehicle_y_max_0': -4,


        'vehicle_initial_speed_min_0': 2,
        'vehicle_initial_speed_max_0': 5,
        'vehicle_targeted_speed_min_0': 0,
        'vehicle_targeted_speed_max_0': 2,
        'vehicle_trigger_distance_min_0': 4,
        'vehicle_trigger_distance_max_0': 10,

        'vehicle_avoid_collision_min_0': 1,
        'vehicle_avoid_collision_max_0': 1,


        'vehicle_dist_to_travel_min_0': 5,
        'vehicle_dist_to_travel_max_0': 30,
        'vehicle_yaw_min_0': 270,
        'vehicle_yaw_max_0': 270
    },
    'customized_parameters_distributions':{
        'vehicle_x_0': ('normal', None, 0.5),
        'vehicle_y_0': ('normal', None, 4)
    },
    'customized_center_transforms':{
        'vehicle_center_transform_0': ('waypoint_ratio', 0)
    },
    'customized_constraints': [
    {'coefficients': [1, 1],
    'labels': ['vehicle_y_0', 'vehicle_trigger_distance_0'],
    'value': 0}
    ]
    },



    'two_pedestrians_cross_street_straight_town05': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 6,
        'num_of_pedestrians_max': 6,
        'num_of_vehicles_min': 0,
        'num_of_vehicles_max': 0,

        'pedestrian_x_min_0': -7,
        'pedestrian_x_max_0': 7,
        'pedestrian_y_min_0': -10,
        'pedestrian_y_max_0': -4,

        'pedestrian_x_min_1': -7,
        'pedestrian_x_max_1': 7,
        'pedestrian_y_min_1': -10,
        'pedestrian_y_max_1': -4,

        'pedestrian_x_min_2': -7,
        'pedestrian_x_max_2': 7,
        'pedestrian_y_min_2': -10,
        'pedestrian_y_max_2': -4,

        'pedestrian_x_min_3': -7,
        'pedestrian_x_max_3': 7,
        'pedestrian_y_min_3': -10,
        'pedestrian_y_max_3': -4,

        'pedestrian_x_min_4': -7,
        'pedestrian_x_max_4': 7,
        'pedestrian_y_min_4': -10,
        'pedestrian_y_max_4': -4,

        'pedestrian_x_min_5': -7,
        'pedestrian_x_max_5': 7,
        'pedestrian_y_min_5': -10,
        'pedestrian_y_max_5': -4,

    },
    'customized_parameters_distributions':{
    },
    'customized_center_transforms':{
        'pedestrian_center_transform_0': ('waypoint_ratio', 0),
        'pedestrian_center_transform_1': ('waypoint_ratio', 0)
    },
    'customized_constraints': []
    },


    'none': {'customized_parameters_bounds':{
        'num_of_static_min': 0,
        'num_of_static_max': 0,
        'num_of_pedestrians_min': 0,
        'num_of_pedestrians_max': 0,
        'num_of_vehicles_min': 0,
        'num_of_vehicles_max': 0,

    },
    'customized_parameters_distributions':{
    },
    'customized_center_transforms':{
    },
    'customized_constraints': []
    },
}


customized_routes = {
    # pick: right turn + leading car stops / slow down, town
    'town05_right_0': {
    'town_name': 'Town05',
    'direction': 'right',
    'route_id': 0,
    'location_list': [(-120, 30), (-103, 4)]
    },

    # pick: change lane, town
    'town05_front_0': {
    'town_name': 'Town05',
    'direction': 'front',
    'route_id': 0,
    'location_list': [(-120, 60), (-124, 26)]
    },

    # pick: turn left non-siginalized intersection, town
    'town01_left_0': {
    'town_name': 'Town01',
    'direction': 'left',
    'route_id': 0,
    'location_list': [(89.1, 300.8), (110.4, 330.5)]
    },

    # pick: go through non-signalized intersection, rural
    'town07_front_0': {
    'town_name': 'Town07',
    'direction': 'front',
    'route_id': 0,
    'location_list': [(-151, -60), (-151, -15)]
    },

    # pick: go through non-signalized intersection, town
    'town04_front_0': {
    'town_name': 'Town04',
    'direction': 'front',
    'route_id': 0,
    'location_list': [(258, -230), (258, -270)]
    },




    # pick: go through signalized crossroad
    'town03_front_0': {
    'town_name': 'Town03',
    'direction': 'front',
    'route_id': 0,
    'location_list': [(9, -105), (9, -155)]
    },


    # go across street, town
    'town05_front_1': {
    'town_name': 'Town05',
    'direction': 'front',
    'route_id': 1,
    'location_list': [(-120, 15), (-120, -20)]
    },




    # change lane, city
    'town10HD_front_0': {
    'town_name': 'Town10HD',
    'direction': 'front',
    'route_id': 0,
    'location_list': [(-35, 138), (-17, 143)]
    },
    # go through non-signalized intersection, rural, error: other cars are not moving
    'town07_left_0': {
    'town_name': 'Town07',
    'direction': 'left',
    'route_id': 0,
    'location_list': [(-75, -64), (-102, -42)]
    },
    # change lane, highway, error: other cars are not moving
    'town04_front_1': {
    'town_name': 'Town04',
    'direction': 'front',
    'route_id': 1,
    'location_list': [(8, 256), (11, 216)]
    },

}





def if_violate_constraints(x, customized_constraints, labels, verbose=False):
    labels_to_id = {label:i for i, label in enumerate(labels)}

    keywords = ['coefficients', 'labels', 'value']
    extra_keywords = ['power']

    for constraint in customized_constraints:
        for k in keywords:
            assert k in constraint
        assert len(constraint['coefficients']) == len(constraint['labels'])

        ids = [labels_to_id[label] for label in constraint['labels']]
        x_ids = [x[id] for id in ids]
        if 'powers' in constraint:
            powers = np.array(constraint['powers'])
        else:
            powers = np.array([1 for _ in range(len(ids))])

        coeff = np.array(constraint['coefficients'])
        features = np.array(x_ids)

        if_violate = np.sum(coeff * np.power(features, powers)) > constraint['value']
        if if_violate:
            if verbose:
                print('\n'*5, 'violate_constraints!!!!', '\n'*5)
                print(coeff, features, powers, np.sum(coeff * np.power(features, powers)), constraint['value'], constraint['labels'])
            return True




    return False

def parse_route_and_scenario(location_list, town_name, scenario, direction, route_str, scenario_file):

    # Parse Route
    TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
    <routes>
    %s
    </routes>"""

    print(location_list, town_name, scenario, direction, route_str)


    pitch = 0
    roll = 0
    yaw = 0
    z = 0

    start_str = '<route id="{}" town="{}">\n'.format(route_str, town_name)
    waypoint_template = '\t<waypoint pitch="{}" roll="{}" x="{}" y="{}" yaw="{}" z="{}" />\n'
    end_str = '</route>'

    wp_str = ''

    for x, y in location_list:
        wp = waypoint_template.format(pitch, roll, x, y, yaw, z)
        wp_str += wp

    final_str = start_str+wp_str+end_str

    folder = make_hierarchical_dir(['leaderboard/data/customized_routes', town_name, scenario, direction])


    pathlib.Path(folder+'/route_{}.xml'.format(route_str)).write_text(TEMPLATE % final_str)


    # Parse Scenario
    x_0, y_0 = location_list[0]
    parse_scenario(scenario_file, town_name, route_str, x_0, y_0)


def parse_route_and_scenario_plain(location_list, town_name, route_str, scenario_file):

    # Parse Route
    TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
    <routes>
    %s
    </routes>"""



    pitch = 0
    roll = 0
    yaw = 0
    z = 0

    start_str = '<route id="{}" town="{}">\n'.format(route_str, town_name)
    waypoint_template = '\t<waypoint pitch="{}" roll="{}" x="{}" y="{}" yaw="{}" z="{}" />\n'
    end_str = '</route>'

    wp_str = ''

    for x, y in location_list:
        wp = waypoint_template.format(pitch, roll, x, y, yaw, z)
        wp_str += wp

    final_str = start_str+wp_str+end_str

    folder = make_hierarchical_dir(['leaderboard/data/temporary_routes', town_name])
    route_path = folder+'/route_{}.xml'.format(route_str)


    pathlib.Path(route_path).write_text(TEMPLATE % final_str)


    # Parse Scenario
    x_0, y_0 = location_list[0]
    parse_scenario(scenario_file, town_name, route_str, x_0, y_0)

    return route_path


def parse_scenario(scenario_file, town_name, route_str, x_0, y_0):
    # Parse Scenario
    x_0_str = str(x_0)
    y_0_str = str(y_0)

    new_scenario = {
    "available_scenarios": [
            {
                town_name: [
                    {
                        "available_event_configurations": [
                            {
                                "route": int(route_str),
                                "center": {
                                    "pitch": "0.0",
                                    "x": x_0_str,
                                    "y": y_0_str,
                                    "yaw": "270",
                                    "z": "0.0"
                                },
                                "transform": {
                                    "pitch": "0.0",
                                    "x": x_0_str,
                                    "y": y_0_str,
                                    "yaw": "270",
                                    "z": "0.0"
                                }
                            }
                        ],
                        "scenario_type": "Scenario12"
                    }
                ]
            }
        ]
    }

    with open(scenario_file, 'w') as f_out:
        annotation_dict = json.dump(new_scenario, f_out, indent=4)


def parse_route_file(route_filename, route_length_lower_bound=50):
    def l2_dist(x, y, prev_x, prev_y):
        return np.sqrt((x-prev_x)**2+(y-prev_y)**2)



    config_list = []
    tree = ET.parse(route_filename)


    for route in tree.iter("route"):
        route_id = int(route.attrib['id'])
        town_name = route.attrib['town']

        transform_list = []
        first_waypoint = True
        d = 0
        for waypoint in route.iter('waypoint'):
            x, y, z = float(waypoint.attrib['x']), float(waypoint.attrib['y']), float(waypoint.attrib['z'])
            pitch, yaw, roll = float(waypoint.attrib['pitch']), float(waypoint.attrib['yaw']), float(waypoint.attrib['roll'])


            if first_waypoint:
                first_waypoint = False
            else:
                d += l2_dist(x, y, prev_x, prev_y)

            transform_list.append((x, y, z, pitch, yaw, roll))
            if d > route_length_lower_bound:
                first_waypoint = True
                d = 0

                config_list.append([route_id, town_name, transform_list])
                transform_list = []

            prev_x, prev_y = x, y


    return config_list

def is_similar(x_1, x_2, mask, xl, xu, p, c, th, verbose=False, labels=[], diff_th=0.1):

    eps = 1e-8

    int_inds = mask == 'int'
    real_inds = mask == 'real'
    int_diff_raw = np.abs(x_1[int_inds] - x_2[int_inds])
    int_diff = np.ones(int_diff_raw.shape) * (int_diff_raw > eps)

    real_diff_raw = np.abs(x_1[real_inds] - x_2[real_inds]) / (np.abs(xu - xl) + eps)[real_inds]

    real_diff = np.ones(real_diff_raw.shape) * (real_diff_raw > c)

    diff = np.concatenate([int_diff, real_diff])

    diff_norm = np.linalg.norm(diff, p)
    equal = diff_norm < th

    # if verbose:
    #     print('diff_raw', int_diff_raw, real_diff_raw, p, c, th, diff, diff_norm)
    #
    #     x_1_r = np.concatenate([x_1[int_inds], x_1[real_inds]])
    #     x_2_r = np.concatenate([x_2[int_inds], x_2[real_inds]])
    #     print(diff[42], x_1_r[42], x_2_r[42], diff[62], x_1_r[62], x_2_r[62])
    #     if len(labels)>0:
    #         labels = np.array(labels)
    #         # print(labels[int_inds])
    #         # print(labels[real_inds])
    #         labels_r = np.concatenate([labels[int_inds], labels[real_inds]])
    #         print(labels_r[42], labels_r[62])

    return equal


def is_distinct(x, X, mask, xl, xu, p, c, th, diff_th=0.1):

    if len(X) == 0:
        return True
    else:
        mask_arr = np.array(mask)
        xl_arr = np.array(xl)
        xu_arr = np.array(xu)
        x = np.array(x)
        X = np.stack(X)
        for x_i in X:
            similar = is_similar(x, x_i, mask_arr, xl_arr, xu_arr, p, c, th, diff_th)
            if similar:
                # print('similar\n')
                return False
        return True



def get_distinct_data_points(data_points, mask, xl, xu, p, c, th, diff_th=0.1):

    # ['forward', 'backward']
    order = 'forward'

    mask_arr = np.array(mask)
    xl_arr = np.array(xl)
    xu_arr = np.array(xu)
    # print(data_points)
    if len(data_points) == 0:
        return [], []
    if len(data_points) == 1:
        return data_points, [0]
    else:
        if order == 'backward':
            distinct_inds = []
            for i in range(len(data_points)-1):
                similar = False
                for j in range(i+1, len(data_points)):

                    similar = is_similar(data_points[i], data_points[j], mask_arr, xl_arr, xu_arr, p, c, th, diff_th)
                    if similar:
                        break
                if not similar:
                    distinct_inds.append(i)
            distinct_inds.append(len(data_points)-1)
        elif order == 'forward':
            distinct_inds = [0]
            for i in range(1, len(data_points)):
                similar = False
                for j in distinct_inds:
                    similar = is_similar(data_points[i], data_points[j], mask_arr, xl_arr, xu_arr, p, c, th, diff_th)
                    if similar:
                        break
                if not similar:
                    distinct_inds.append(i)

    return list(np.array(data_points)[distinct_inds]), distinct_inds


def check_bug(objectives):
    # speed needs to be large than 0.2 to avoid false positive
    return objectives[0] > 0.2 or objectives[5] or objectives[6] or objectives[7]


def start_server(port):
    # hack: this heavily relies on the relative path of carla
    cmd_list = shlex.split('sh ../carla_0994_no_rss/CarlaUE4.sh -opengl -carla-rpc-port='+str(port)+' -carla-streaming-port=0')
    while is_port_in_use(int(port)):
        try:
            # show_ports_cmd = shlex.split('lsof -t -i:'+str(port))
            # result = subprocess.run(show_ports_cmd, stdout=subprocess.PIPE)
            # pids = result.stdout.decode("utf-8").strip().split('\n')
            # own_pid = str(os.getpid())
            #
            # if own_pid in pids:
            #     pids.remove(own_pid)
            #     if len(pids) > 0:
            #         pid_to_kill = pids[0]
            #         print('pid_to_kill', pid_to_kill)
            #         subprocess.run('kill -9 '+pid_to_kill, shell=True)
            # else:
            #     subprocess.run('kill $(lsof -t -i:'+str(port)+')', shell=True)
            subprocess.run('kill $(lsof -t -i:'+str(port)+')', shell=True)
            print('-'*20, 'kill server at port', port)
            time.sleep(2)
        except:
            import traceback
            traceback.print_exc()
            continue
    subprocess.Popen(cmd_list)
    print('-'*20, 'start server at port', port)
    # 10s is usually enough
    time.sleep(10)





def port_to_gpu(port):
    gpu = port % 2
    return gpu







def estimate_objectives(save_path, default_objectives):

    events_path = os.path.join(save_path, 'events.txt')
    deviations_path = os.path.join(save_path, 'deviations.txt')

    # set thresholds to avoid too large influence
    ego_linear_speed = 0
    min_d = 7
    offroad_d = 7
    wronglane_d = 7
    dev_dist = 0

    is_offroad = 0
    is_wrong_lane = 0
    is_run_red_light = 0
    is_collision = 0

    with open(deviations_path, 'r') as f_in:
        for line in f_in:
            type, d = line.split(',')
            d = float(d)
            if type == 'min_d':
                min_d = np.min([min_d, d])
            elif type == 'offroad_d':
                offroad_d = np.min([offroad_d, d])
            elif type == 'wronglane_d':
                wronglane_d = np.min([wronglane_d, d])
            elif type == 'dev_dist':
                dev_dist = np.max([dev_dist, d])

    x = None
    y = None
    object_type = None

    infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'off_road']

    try:
        with open(events_path) as json_file:
            events = json.load(json_file)
    except:
        print('events_path', events_path, 'is not found')
        return default_objectives, (None, None), None
    infractions = events['_checkpoint']['records'][0]['infractions']
    status = events['_checkpoint']['records'][0]['status']


    route_completion = float(events['values'][1])

    for infraction_type in infraction_types:
        for infraction in infractions[infraction_type]:
            if 'collisions' in infraction_type:
                typ = re.search('.*with type=(.*) and id.*', infraction)
                print(infraction, typ)
                if typ:
                    object_type = typ.group(1)
                loc = re.search('.*x=(.*), y=(.*), z=(.*), ego_linear_speed=(.*), other_actor_linear_speed=(.*)\)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    ego_linear_speed = float(loc.group(4))
                    other_actor_linear_speed = float(loc.group(5))
                is_collision = 1

            elif infraction_type == 'off_road':
                loc = re.search('.*x=(.*), y=(.*), z=(.*)\)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    is_offroad = 1
            else:
                if infraction_type == 'wrong_lane':
                    is_wrong_lane = 1
                elif infraction_type == 'red_light':
                    is_run_red_light = 1
                loc = re.search('.*x=(.*), y=(.*), z=(.*)[\),]', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))

    # limit impact of too large values
    ego_linear_speed = np.min([ego_linear_speed, 7])
    dev_dist = np.min([dev_dist, 7])

    return [ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, is_collision], (x, y), object_type, route_completion


def norm_2d(loc_1, loc_2):
    return np.sqrt((loc_1.x-loc_2.x)**2+(loc_1.y-loc_2.y)**2)

def get_bbox(vehicle):
    current_tra = vehicle.get_transform()
    current_loc = current_tra.location

    heading_vec = current_tra.get_forward_vector()
    heading_vec.z = 0
    heading_vec = heading_vec / math.sqrt(math.pow(heading_vec.x, 2) + math.pow(heading_vec.y, 2))
    perpendicular_vec = carla.Vector3D(-heading_vec.y, heading_vec.x, 0)

    extent = vehicle.bounding_box.extent
    x_boundary_vector = heading_vec * extent.x
    y_boundary_vector = perpendicular_vec * extent.y

    bbox = [
        current_loc + carla.Location(x_boundary_vector - y_boundary_vector),
        current_loc + carla.Location(x_boundary_vector + y_boundary_vector),
        current_loc + carla.Location(-1 * x_boundary_vector - y_boundary_vector),
        current_loc + carla.Location(-1 * x_boundary_vector + y_boundary_vector)]

    return bbox



def correct_travel_dist(data, labels, tmp_travel_dist_file):
    from collections import OrderedDict

    if os.path.exists(tmp_travel_dist_file):
        label_to_id = {label:i for i, label in enumerate(labels)}
        # add label and value of resulting variables to x
        id_to_label = {}
        id_to_dist = {}
        with open(tmp_travel_dist_file, 'r') as f_in:
            for line in f_in:
                tokens = line.strip().split(',')
                if len(tokens) == 3:
                    actor_id, general_actor_type, index  = tokens
                    id_to_label[actor_id] = '_'.join([general_actor_type, 'dist_to_travel', index])
                elif len(tokens) == 2:
                    actor_id = tokens[0]
                    dist = float(tokens[1])
                    if actor_id not in id_to_dist or (actor_id in id_to_dist and dist > id_to_dist[actor_id]):
                        id_to_dist[actor_id] = dist

        for actor_id in id_to_label:
            label = id_to_label[actor_id]
            if actor_id in id_to_dist:
                dist = id_to_dist[actor_id]
            else:
                dist = 0
            entry_i = labels.index(label)
            data[entry_i] = dist
    else:
        pass
        # print('\n'*3, tmp_travel_dist_file, 'does not exist', '\n'*3)


def angle_from_center_view_fov(target, ego, fov=90):
    target_location = target.get_location()
    ego_location = ego.get_location()
    ego_orientation = ego.get_transform().rotation.yaw

    target_vector = np.array([target_location.x - ego_location.x, target_location.y - ego_location.y])
    norm_target = np.linalg.norm(target_vector)

    if norm_target < 0.001:
        return 0

    forward_vector = np.array([math.cos(math.radians(ego_orientation)), math.sin(math.radians(ego_orientation))])

    try:
        d_angle = np.abs(math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target)))
    except:
        print('\n'*3, 'np.dot(forward_vector, target_vector)', np.dot(forward_vector, target_vector), norm_target, '\n'*3)
        d_angle = 0
    d_angle_norm = np.clip((d_angle - fov / 2) / (180 - fov / 2), 0, 1)

    return d_angle_norm














# analysis

def draw_auc_roc_for_scores(scores, y_test):
    inds_sorted = np.argsort(scores)
    tp, fp = 0, 0
    fpr_list, tpr_list = [], []
    n = len(inds_sorted)
    t = np.sum(y_test==1)
    f = n - t
    for ind in inds_sorted:
        if y_test[ind] == 1:
            tp += 1
        else:
            fp += 1
        fpr_list.append(fp/f)
        tpr_list.append(tp/t)
    from matplotlib import pyplot as plt
    plt.plot(fpr_list, tpr_list)
    plt.plot(np.arange(0,1.2,0.2), np.arange(0,1.2,0.2))
    plt.show()

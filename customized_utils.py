import argparse
import carla
import os
import numpy as np
from leaderboard.customized.object_params import Static, Pedestrian, Vehicle
from dask.distributed import Client, LocalCluster
from psutil import process_iter
from signal import SIGTERM
import socket

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
    # modification: 30->15
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
    parser.add_argument("--save-folder", type=str, default='/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data', help="Path to save simulation data")


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
        self.agent_config = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/models/epoch=24.ckpt'
        self.track = 'SENSORS'
        self.resume = False
        self.checkpoint = ''
        self.weather_index = 19
        self.save_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized'




def add_transform(transform1, transform2):
    x = transform1.location.x + transform2.location.x
    y = transform1.location.y + transform2.location.y
    z = transform1.location.z + transform2.location.z
    pitch = transform1.rotation.pitch + transform2.rotation.pitch
    yaw = transform1.rotation.yaw + transform2.rotation.yaw
    roll = transform1.rotation.roll + transform2.rotation.roll
    return create_transform(x, y, z, pitch, yaw, roll)


def convert_x_to_customized_data(x, waypoints_num_limit, max_num_of_static, max_num_of_pedestrians, max_num_of_vehicles, static_types, pedestrian_types, vehicle_types, vehicle_colors):

    # parameters
    # global
    friction = x[0]
    weather_index = int(x[1])
    num_of_static = int(x[2])
    num_of_pedestrians = int(x[3])
    num_of_vehicles = int(x[4])

    ind = 5
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

            targeted_speed_i = x[ind+6]
            waypoint_follower_i = bool(x[ind+7])

            targeted_waypoint_i = create_transform(x[ind+8], x[ind+9], 0, 0, 0, 0)

            vehicle_avoid_collision_i = bool(x[ind+10])
            vehicle_dist_to_travel_i = x[ind+11]
            vehicle_target_yaw_i = x[ind+12]
            x_dir = np.cos(np.deg2rad(vehicle_target_yaw_i))
            y_dir = np.sin(np.deg2rad(vehicle_target_yaw_i))
            target_direction_i = carla.Vector3D(x_dir, y_dir, 0)

            vehicle_color_i = vehicle_colors[int(x[ind+13])]

            ind += 14

            vehicle_waypoints_perturbation_i = []
            for _ in range(waypoints_num_limit):
                dx = x[ind]
                dy = x[ind+1]
                vehicle_waypoints_perturbation_i.append([dx, dy])
                ind += 2

            vehicle_i = Vehicle(model=vehicle_type_i, spawn_transform=vehicle_transform_i, avoid_collision=vehicle_avoid_collision_i, initial_speed=vehicle_initial_speed_i, trigger_distance=vehicle_trigger_distance_i, waypoint_follower=waypoint_follower_i, targeted_waypoint=targeted_waypoint_i, dist_to_travel=vehicle_dist_to_travel_i,
            target_direction=target_direction_i,
            targeted_speed=targeted_speed_i, after_trigger_behavior='stop', color=vehicle_color_i, waypoints_perturbation=vehicle_waypoints_perturbation_i)

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
    'port': port}


    return customized_data



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
        return s.connect_ex(('localhost', port)) == 0


def exit_handler(ports):
    for port in ports:
        if is_port_in_use(port):
            for proc in process_iter():
                for conns in proc.connections(kind='inet'):
                    if conns.laddr.port == port:
                        print('-'*100, 'kill server at port', port)
                        proc.send_signal(SIGTERM)

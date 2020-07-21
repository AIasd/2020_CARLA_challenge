'''

TBD:
* fix infractions = events['_checkpoint']['records'][0]['infractions'], IndexError: list index out of range
* show params
* multi-process of simulations
* free-view window of the map
'''

import sys
import os
sys.path.append('pymoo')
carla_root = '/home/zhongzzy9/Documents/self-driving-car/carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')
sys.path.append('.')
sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')
os.environ['HAS_DISPLAY'] = '1'


from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.duplicate import ElementwiseDuplicateElimination

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize

from object_types import WEATHERS, pedestrian_types, vehicle_types,
 static_types, vehicle_colors

from customized_utils import create_transform, rand_real, specify_args

import numpy as np
import carla

from leaderboard.fuzzing import LeaderboardEvaluator
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.customized.object_params import Static, Pedestrian, Vehicle

import traceback
import json


rng = np.random.default_rng(0)


class MyProblem(Problem):

    def __init__(self, n_characters=10):
        self.counter = 0
        self.bugs = []

        # Fixed hyper-parameters
        self.waypoints_num_limit = 10
        self.max_num_of_static = 2
        self.max_num_of_pedestrians = 2
        self.max_num_of_vehicles = 2
        self.num_of_weathers = len(WEATHERS)
        self.num_of_static_types = len(static_types)
        self.num_of_pedestrian_types = len(pedestrian_types)
        self.num_of_vehicle_types = len(vehicle_types)
        self.num_of_vehicle_colors = len(vehicle_colors)

        # general
        self.perturbation_min = -1.6
        self.perturbation_max = 1.6
        self.yaw_min = 0
        self.yaw_max = 360

        # static
        self.static_x_min = -20
        self.static_x_max = 20
        self.static_y_min = -20
        self.static_y_max = 20

        # pedestrians
        self.pedestrian_x_min = -20
        self.pedestrian_x_max = 20
        self.pedestrian_y_min = -20
        self.pedestrian_y_max = 20
        self.pedestrian_trigger_distance_min = 0
        self.pedestrian_trigger_distance_max = 50
        self.pedestrian_speed_min = 0
        self.pedestrian_speed_max = 4
        self.pedestrian_dist_to_travel_min = 0
        self.pedestrian_dist_to_travel_max = 50

        # vehicles
        self.vehicle_x_min = -20
        self.vehicle_x_max = 20
        self.vehicle_y_min = -20
        self.vehicle_y_max = 20
        self.vehicle_initial_speed_min = 0
        self.vehicle_initial_speed_max = 15
        self.vehicle_trigger_distance_min = 0
        self.vehicle_trigger_distance_max = 50
        self.vehicle_targeted_speed_min = 0
        self.vehicle_targeted_speed_max = 15
        self.vehicle_targeted_x_min = -20
        self.vehicle_targeted_x_max = 20
        self.vehicle_targeted_y_min = -20
        self.vehicle_targeted_y_max = 20
        self.vehicle_dist_to_travel_min = 0
        self.vehicle_dist_to_travel_max = 50



        # construct xl and xu
        xl = [0, 0, 0, 0, 0]
        xu = [1,
        self.num_of_weathers,
        self.max_num_of_static,
        self.max_num_of_pedestrians,
        self.max_num_of_vehicles
        ]
        mask = ['real', 'int', 'int', 'int', 'int']
        labels = ['friction', 'num_of_weathers', 'max_num_of_static', 'max_num_of_pedestrians', 'max_num_of_vehicles']
        # ego-car
        for i in range(self.waypoints_num_limit):
            xl.extend([self.perturbation_min]*2)
            xu.extend([self.perturbation_max]*2)
            mask.extend(['real']*2)
            labels.extend(['perturbation_x_'+str(i), 'perturbation_y_'+str(i)])
        # static
        for i in range(self.max_num_of_static):
            xl.extend([0, self.static_x_min, self.static_y_min, self.yaw_min])
            xu.extend([self.num_of_static_types, self.static_x_max, self.static_y_max, self.yaw_max])
            mask.extend(['int'] + ['real']*3)
            labels.extend(['num_of_static_types_'+str(i), 'static_x_'+str(i), 'static_y_'+str(i), 'yaw_'+str(i)])
        # pedestrians
        for i in range(self.max_num_of_pedestrians):
            xl.extend([0, self.pedestrian_x_min, self.pedestrian_y_min, self.yaw_min, self.pedestrian_trigger_distance_min, self.pedestrian_speed_min, self.pedestrian_dist_to_travel_min])
            xu.extend([self.num_of_pedestrian_types, self.pedestrian_x_max, self.pedestrian_y_max, self.yaw_max, self.pedestrian_trigger_distance_max, self.pedestrian_speed_max, self.pedestrian_dist_to_travel_max])
            mask.extend(['int'] + ['real']*6)
            labels.extend(['num_of_pedestrian_types_'+str(i), 'pedestrian_x_'+str(i), 'pedestrian_y_'+str(i), 'yaw_'+str(i), 'pedestrian_trigger_distance_'+str(i), 'pedestrian_speed_'+str(i), 'pedestrian_dist_to_travel_'+str(i)])
        # vehicles
        for i in range(self.max_num_of_vehicles):
            xl.extend([0, self.vehicle_x_min, self.vehicle_y_min, self.yaw_min, self.vehicle_initial_speed_min, self.vehicle_trigger_distance_min, self.vehicle_targeted_speed_min, 0, self.vehicle_targeted_x_min, self.vehicle_targeted_y_min, 0, self.vehicle_dist_to_travel_min, self.yaw_min, 0])

            xu.extend([self.num_of_vehicle_types, self.vehicle_x_max, self.vehicle_y_max, self.yaw_max, self.vehicle_initial_speed_max, self.vehicle_trigger_distance_max, self.vehicle_targeted_speed_max, 1, self.vehicle_targeted_x_max, self.vehicle_targeted_y_max, 1, self.vehicle_dist_to_travel_max, self.yaw_max, self.num_of_vehicle_colors])
            mask.extend(['int'] + ['real']*6 + ['int'] + ['real']*2 + ['int'] + ['real']*2 + ['int'])
            labels.extend(['num_of_vehicle_types_'+str(i), 'vehicle_x_'+str(i), 'vehicle_y_'+str(i), 'yaw_'+str(i), 'vehicle_initial_speed_'+str(i), 'vehicle_trigger_distance_'+str(i), 'vehicle_targeted_speed_'+str(i), 'waypoint_follower_'+str(i), 'vehicle_targeted_x_'+str(i), 'vehicle_targeted_y_'+str(i), 'avoid_collision_'+str(i), 'vehicle_dist_to_travel_'+str(i), 'yaw_'+str(i), 'num_of_vehicle_colors_'+str(i)])

            for j in range(self.waypoints_num_limit):
                xl.extend([self.perturbation_min]*2)
                xu.extend([self.perturbation_max]*2)
                mask.extend(['real']*2)
                labels.extend(['perturbation_x_'+str(i)+'_'+str(j), 'perturbation_y_'+str(i)+'_'+str(j)])

        n_var = 5+self.waypoints_num_limit*2+self.max_num_of_static*4+self.max_num_of_pedestrians*7+self.max_num_of_vehicles*(14+self.waypoints_num_limit*2)

        self.mask = mask
        self.labels = labels

        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)





    def _evaluate(self, x, out, *args, **kwargs):

        x = denormalize_by_entry(self, x)

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
        for _ in range(self.waypoints_num_limit):
            dx = x[ind]
            dy = x[ind+1]
            ego_car_waypoints_perturbation.append([dx, dy])
            ind += 2

        # static
        static_list = []
        for i in range(self.max_num_of_pedestrians):
            if i < num_of_static:
                static_type_i = static_types[int(x[ind])]
                static_transform_i = create_transform(x[ind+1], x[ind+2], 0, 0, x[ind+3], 0)
                static_i = Static(model=static_type_i, spawn_transform=static_transform_i)
                static_list.append(static_i)
            ind += 4

        # pedestrians
        pedestrian_list = []
        for i in range(self.max_num_of_pedestrians):
            if i < num_of_pedestrians:
                pedestrian_type_i = pedestrian_types[int(x[ind])]
                pedestrian_transform_i = create_transform(x[ind+1], x[ind+2], 0, 0, x[ind+3], 0)
                pedestrian_i = Pedestrian(model=pedestrian_type_i, spawn_transform=pedestrian_transform_i, trigger_distance=x[ind+4], speed=x[ind+5], dist_to_travel=x[ind+6], after_trigger_behavior='stop')
                pedestrian_list.append(pedestrian_i)
            ind += 7

        # vehicles
        vehicle_list = []
        for i in range(self.max_num_of_vehicles):
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
                for _ in range(self.waypoints_num_limit):
                    dx = x[ind]
                    dy = x[ind+1]
                    vehicle_waypoints_perturbation_i.append([dx, dy])
                    ind += 2

                vehicle_i = Vehicle(model=vehicle_type_i, spawn_transform=vehicle_transform_i, avoid_collision=vehicle_avoid_collision_i, initial_speed=vehicle_initial_speed_i, trigger_distance=vehicle_trigger_distance_i, waypoint_follower=waypoint_follower_i, targeted_waypoint=targeted_waypoint_i, dist_to_travel=vehicle_dist_to_travel_i,
                target_direction=target_direction_i,
                targeted_speed=targeted_speed_i, after_trigger_behavior='stop', color=vehicle_color_i, waypoints_perturbation=vehicle_waypoints_perturbation_i)

                vehicle_list.append(vehicle_i)
            else:
                ind += 14 + self.waypoints_num_limit*2




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
        'add_center': True}



        # run simulation
        ego_linear_speed, offroad_dist, is_wrong_lane, is_run_red_light = run_simulation(customized_data)

        # multi-objectives
        out["F"] = np.array([- ego_linear_speed, - offroad_dist, - is_wrong_lane, - is_run_red_light], dtype=np.float)

        # multi-constraints
        # out["G"] =

        # record specs for bugs

        if ego_linear_speed > 0 or offroad_dist > 0 or is_wrong_lane > 0 or is_run_red_light > 0:
            self.bugs.append({'counter':self.counter, 'x':x, 'ego_linear_speed':ego_linear_speed, 'offroad_dist':offroad_dist, 'is_wrong_lane':is_wrong_lane, 'is_run_red_light':is_run_red_light})

        self.counter += 1



def run_simulation(customized_data):
    arguments = specify_args()
    arguments.challenge_mode = True
    arguments.scenarios='leaderboard/data/all_towns_traffic_scenarios_public.json'
    arguments.agent='scenario_runner/team_code/image_agent.py'
    arguments.agent_config='/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/models/epoch=24.ckpt'
    os.environ['SAVE_FOLDER'] = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized'




    arguments.port = 2000
    arguments.debug = True
    statistics_manager = StatisticsManager()


    # Fixed Hyperparameters
    using_customized_route_and_scenario = True
    multi_actors_scenarios = ['Scenario12']
    arguments.scenarios = 'leaderboard/data/fuzzing_scenarios.json'
    town_name = 'Town10HD'
    scenario = 'Scenario12'
    direction = 'right'
    route = 0
    # sample_factor is an integer between [1, 8]
    sample_factor = 5
    weather_index = customized_data['weather_index']


    # Laundry Stuff-------------------------------------------------------------
    arguments.weather_index = weather_index
    os.environ['WEATHER_INDEX'] = str(weather_index)

    town_scenario_direction = town_name + '/' + scenario

    folder_1 = os.environ['SAVE_FOLDER'] + '/' + town_name
    folder_2 = folder_1 + '/' + scenario
    if not os.path.exists(folder_1):
        os.mkdir(folder_1)
    if not os.path.exists(folder_2):
        os.mkdir(folder_2)
    if scenario in multi_actors_scenarios:
        town_scenario_direction += '/' + direction
        folder_2 += '/' + direction
        if not os.path.exists(folder_2):
            os.mkdir(folder_2)

    os.environ['SAVE_FOLDER'] = folder_2
    arguments.save_folder = os.environ['SAVE_FOLDER']

    route_prefix = 'leaderboard/data/customized_routes/' + town_scenario_direction + '/route_'

    route_str = str(route)
    if route < 10:
        route_str = '0'+route_str
    arguments.routes = route_prefix+route_str+'.xml'
    os.environ['ROUTES'] = arguments.routes

    # extract waypoints along route
    import xml.etree.ElementTree as ET
    tree = ET.parse(arguments.routes)
    route_waypoints = []

    # this iteration should only go once since we only keep one route per file
    for route in tree.iter("route"):
        route_id = route.attrib['id']
        route_town = route.attrib['town']

        for waypoint in route.iter('waypoint'):
            route_waypoints.append(create_transform(float(waypoint.attrib['x']), float(waypoint.attrib['y']), float(waypoint.attrib['z']), float(waypoint.attrib['pitch']), float(waypoint.attrib['yaw']), float(waypoint.attrib['roll'])))

    # extract waypoints for the scenario
    world_annotations = RouteParser.parse_annotations_file(arguments.scenarios)
    info = world_annotations[town_name][0]["available_event_configurations"][0]

    center = info["center"]
    RouteParser.convert_waypoint_float(center)
    center_location = carla.Location(float(center['x']), float(center['y']), float(center['z']))
    center_rotation = carla.Rotation(float(center['pitch']), float(center['yaw']), 0.0)
    center_transform = carla.Transform(center_location, center_rotation)
    # --------------------------------------------------------------------------


    customized_data['center_transform'] = center_transform
    customized_data['using_customized_route_and_scenario'] = True
    customized_data['destination'] = route_waypoints[-1].location
    customized_data['sample_factor'] = sample_factor


    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments, customized_data)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator
        # collect signals for estimating objectives
        events_path = arguments.save_folder+'/route_'+route_str+'_'+str(arguments.weather_index)+'/events.txt'
        objectives = estimate_objectives(events_path)




    return objectives




def estimate_objectives(events_path):
    ego_linear_speed = -1
    offroad_dist = 0
    is_wrong_lane = 0
    is_run_red_light = 0

    infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'off_road']
    print('events_path :', events_path)
    with open(events_path) as json_file:
        events = json.load(json_file)
    infractions = events['_checkpoint']['records'][0]['infractions']
    for infraction_type in infraction_types:
        for infraction in infractions[infraction_type]:
            if 'collisions' in infraction_type:
                loc = re.search('.*x=(.*), y=(.*), z=(.*), ego_linear_speed=(.*), other_actor_linear_speed=(.*)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    ego_linear_speed = float(loc.group(4))
                    other_actor_linear_speed = float(loc.group(5))
                    events_list.append((x, y, infraction_type, ego_linear_speed, other_actor_linear_speed))
            elif infraction_type == 'off_road':
                loc = re.search('.*x=(.*), y=(.*), z=(.*), offroad distance=(.*)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    offroad_dist = float(loc.group(4))
                    events_list.append((x, y, infraction_type, offroad_dist))
            else:
                if infraction_type == 'wrong_lane':
                    is_wrong_lane = 1
                elif infraction_type == 'red_light':
                    is_run_red_light = 1
                loc = re.search('.*x=(.*), y=(.*), z=(.*)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    events_list.append((x, y, infraction_type))

    return ego_linear_speed, offroad_dist, is_wrong_lane, is_run_red_light

def object_to_ndarray():
    pass
def ndarray_to_object():
    pass


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = []
        print('n_samples', n_samples)
        for i in range(n_samples):
            '''
            dimension correspondence

            Define:
            n1=problem.waypoints_num_limit
            n2=problem.max_num_of_static
            n3=problem.max_num_of_pedestrians
            n4=problem.max_num_of_vehicles

            global
            0: friction, real, [0, 1].
            1: weather_index, int, [0, problem.num_of_weathers].
            2: num_of_static, int, [0, n2].
            3: num_of_pedestrians, int, [0, n3].
            4: num_of_vehicles, int, [0, n4].

            ego-car
            5 ~ 4+n1*2: waypoints perturbation [(dx_i, dy_i)] with length n1.
            dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].


            static
            5+n1*2 ~ 4+n1*2+n2*4: [(static_type_i, x w.r.t. center, y w.r.t. center, yaw)] with length n2.
            static_type_i, int, [0, problem.num_of_static_types).
            x_i, real, [problem.static_x_min, problem.static_x_max].
            y_i, real, [problem.static_y_min, problem.static_y_max].
            yaw_i, real, [problem.yaw_min, problem.yaw_max).

            pedestrians
            5+n1*2+n2*4 ~ 4+n1*2+n2*4+n3*7: [(pedestrian_type_i, x_i, y_i, yaw_i, trigger_distance_i, speed_i, dist_to_travel_i)] with length n3.
            pedestrian_type_i, int, [0, problem.num_of_static_types)
            x_i, real, [problem.pedestrian_x_min, problem.pedestrian_x_max].
            y_i, real, [problem.pedestrian_y_min, problem.pedestrian_y_max].
            yaw_i, real, [problem.yaw_min, problem.yaw_max).
            trigger_distance_i, real, [problem.pedestrian_trigger_distance_min, problem.pedestrian_trigger_distance_max].
            speed_i, real, [problem.pedestrian_speed_min, problem.pedestrian_speed_max].
            dist_to_travel_i, real, [problem.pedestrian_dist_to_travel_min, problem.pedestrian_dist_to_travel_max].

            vehicles
            5+n1*2+n2*4+n3*7 ~ 4+n1*2+n2*4+n3*7+n4*(14+n1*2): [(vehicle_type_i, x_i, y_i, yaw_i, initial_speed_i, trigger_distance_i, targeted_speed_i, waypoint_follower_i, targeted_x_i, targeted_y_i, avoid_collision_i, dist_to_travel_i, target_yaw_i, color_i, [(dx_i, dy_i)] with length n1)] with length n4.
            vehicle_type_i, int, [0, problem.num_of_vehicle_types)
            x_i, real, [problem.vehicle_x_min, problem.vehicle_x_max].
            y_i, real, [problem.vehicle_y_min, problem.vehicle_y_max].
            yaw_i, real, [problem.yaw_min, problem.yaw_max).
            initial_speed_i, real, [problem.vehicle_initial_speed_min, problem.vehicle_initial_speed_max].
            trigger_distance_i, real, [problem.vehicle_trigger_distance_min, problem.vehicle_trigger_distance_max].
            targeted_speed_i, real, [problem.vehicle_targeted_speed_min, problem.vehicle_targeted_speed_max].
            waypoint_follower_i, boolean, [0, 1]
            targeted_x_i, real, [problem.targeted_x_min, problem.targeted_x_max].
            targeted_y_i, real, [problem.targeted_y_min, problem.targeted_y_max].
            avoid_collision_i, boolean, [0, 1]
            dist_to_travel_i, real, [problem.vehicle_dist_to_travel_min, problem.vehicle_dist_to_travel_max].
            target_yaw_i, real, [problem.yaw_min, problem.yaw_max).
            color_i, int, [0, problem.num_of_vehicle_colors).
            dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].


            '''
            d = 4+problem.waypoints_num_limit*2+problem.max_num_of_static*4+problem.max_num_of_pedestrians*7+problem.max_num_of_vehicles*(12+problem.waypoints_num_limit*2)

            x = []

            # global
            friction = rng.random()
            weather_index = rng.integers(problem.num_of_weathers+1)
            num_of_static = rng.integers(problem.max_num_of_static+1)
            num_of_pedestrians = rng.integers(problem.max_num_of_pedestrians+1)
            num_of_vehicles = rng.integers(problem.max_num_of_vehicles+1)
            x.extend([friction, weather_index, num_of_static, num_of_pedestrians, num_of_vehicles])

            # ego car
            for _ in range(problem.waypoints_num_limit):
                dx = np.clip(rng.normal(0, 2, 1)[0], problem.perturbation_min, problem.perturbation_max)
                dy = np.clip(rng.normal(0, 2, 1)[0], problem.perturbation_min, problem.perturbation_max)
                x.extend([dx, dy])
            # static
            for i in range(problem.max_num_of_static):
                static_type_i = rng.integers(problem.num_of_static_types)
                static_x_i = rand_real(rng, problem.static_x_min, problem.static_x_max)
                static_y_i = rand_real(rng, problem.static_y_min, problem.static_y_max)
                static_yaw_i = rand_real(rng, problem.yaw_min, problem.yaw_max)
                x.extend([static_type_i, static_x_i, static_y_i, static_yaw_i])
            # pedestrians
            for i in range(problem.max_num_of_pedestrians):
                pedestrian_type_i = rng.integers(problem.num_of_static_types)
                pedestrian_x_i = rand_real(rng, problem.pedestrian_x_min, problem.pedestrian_x_max)
                pedestrian_y_i = rand_real(rng, problem.pedestrian_x_min, problem.pedestrian_x_max)
                pedestrian_yaw_i = rand_real(rng, problem.yaw_min, problem.yaw_max)
                pedestrian_trigger_distance_i = rand_real(rng, problem.pedestrian_trigger_distance_min, problem.pedestrian_trigger_distance_max)
                speed_i = rand_real(rng, problem.pedestrian_speed_min, problem.pedestrian_speed_max)
                dist_to_travel_i = rand_real(rng, problem.pedestrian_dist_to_travel_min, problem.pedestrian_dist_to_travel_max)
                x.extend([pedestrian_type_i, pedestrian_x_i, pedestrian_y_i, pedestrian_yaw_i, pedestrian_trigger_distance_i, speed_i, dist_to_travel_i])
            # vehicles
            for i in range(problem.max_num_of_vehicles):
                vehicle_type_i = rand_real(rng, 0, problem.num_of_vehicle_types)
                vehicle_x_i = rand_real(rng, problem.vehicle_x_min, problem.vehicle_x_max)
                vehicle_y_i = rand_real(rng, problem.vehicle_x_min, problem.vehicle_x_max)
                vehicle_yaw_i = rand_real(rng, problem.yaw_min, problem.yaw_max)
                vehicle_initial_speed_i = rand_real(rng, problem.vehicle_initial_speed_min, problem.vehicle_initial_speed_max)
                vehicle_trigger_distance_i = rand_real(rng, problem.vehicle_trigger_distance_min, problem.vehicle_trigger_distance_max)

                targeted_speed_i = rand_real(rng, problem.vehicle_targeted_speed_min, problem.vehicle_targeted_speed_max)
                waypoint_follower_i = rng.integers(2)

                vehicle_targeted_x_i = rand_real(rng, problem.vehicle_targeted_x_min, problem.vehicle_targeted_x_max)
                vehicle_targeted_y_i = rand_real(rng, problem.vehicle_targeted_x_min, problem.vehicle_targeted_x_max)

                vehicle_avoid_collision_i = rng.integers(2)
                vehicle_dist_to_travel_i = rand_real(rng, problem.vehicle_dist_to_travel_min, problem.vehicle_dist_to_travel_max)
                vehicle_target_yaw_i = rand_real(rng, problem.yaw_min, problem.yaw_max)
                vehicle_color_i = rng.integers(0, problem.num_of_vehicle_colors)

                x.extend([vehicle_type_i, vehicle_x_i, vehicle_y_i, vehicle_yaw_i, vehicle_initial_speed_i, vehicle_trigger_distance_i, targeted_speed_i, waypoint_follower_i, vehicle_targeted_x_i, vehicle_targeted_y_i, vehicle_avoid_collision_i, vehicle_dist_to_travel_i, vehicle_target_yaw_i, vehicle_color_i])

                for _ in range(problem.waypoints_num_limit):
                    dx = np.clip(rng.normal(0, 2, 1)[0], problem.perturbation_min, problem.perturbation_max)
                    dy = np.clip(rng.normal(0, 2, 1)[0], problem.perturbation_min, problem.perturbation_max)
                    x.extend([dx, dy])
            x = np.array(x).astype(float)
            X.append(x)
        X = np.stack(X)
        print(X.shape)
        X = normalize_by_entry(problem, X)

        return X

def normalize_by_entry(problem, X):
    for i in np.where((problem.xu - problem.xl) == 0)[0]:
        print(i, problem.labels[i])
    return (X / (problem.xu - problem.xl)) + 0.5

def denormalize_by_entry(problem, X):
    return (X - 0.5) * (problem.xu - problem.xl)



class MyCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        self.crossover_rate = 0.6
        super().__init__(2, 2, self.crossover_rate)


    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=np.object)

        # for each mating provided
        for k in range(n_matings):

            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]

            # prepare the offsprings
            off_a = ["_"] * problem.n_characters
            off_b = ["_"] * problem.n_characters

            for i in range(problem.n_characters):
                if rng.random() < 0.5:
                    off_a[i] = a[i]
                    off_b[i] = b[i]
                else:
                    off_a[i] = b[i]
                    off_b[i] = a[i]

            # join the character list and set the output
            Y[0, k, 0], Y[1, k, 0] = "".join(off_a), "".join(off_b)

        return Y



class MyMutation(Mutation):
    def __init__(self):
        super().__init__()


    def _do(self, problem, X, **kwargs):
        self.mutation_rate = 1 / problem.n_var
        # for each individual
        for i in range(len(X)):
            x = X[i]


            # with a probabilty of 40% - change the order of characters
            if r < 0.4:
                perm = rng.permutation(problem.n_characters)
                X[i, 0] = "".join(np.array([e for e in X[i, 0]])[perm])

            # also with a probabilty of 40% - change a character randomly
            elif r < 0.8:
                prob = 1 / problem.n_characters
                mut = [c if rng.random() > prob
                       else rng.choice(problem.ALPHABET) for c in X[i, 0]]
                X[i, 0] = "".join(mut)

        return X



class MyDuplicateElimination(ElementwiseDuplicateElimination):
    # TBD: should support cases that we only consider distances on the real variables
    def is_equal(self, a, b):
        return np.linalg.norm(a.X - b.X) < 0.1 * a.X.shape[0]


def main():
    problem = MyProblem()
    print(problem.n_var)
    # TBD: customize mutation and crossover to better fit our problem. e.g.
    # might deal with int and real separately
    algorithm = NSGA2(pop_size=100,
                      sampling=MySampling(),
                      eliminate_duplicates=MyDuplicateElimination())

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 20),
                   seed=0,
                   verbose=True)

    print('save', len(problem.bugs), 'bugs')
    np.savez('bugs', problem.bugs)

    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)

if __name__ == '__main__':
    main()

import os
import pathlib
import traceback
import json
import pickle
import re
from distutils.dir_util import copy_tree
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from leaderboard.fuzzing import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager
from customized_utils import arguments_info, make_hierarchical_dir, estimate_objectives, check_bug, classify_bug_type, arguments_info, create_transform, convert_x_to_customized_data
from scene_configs import customized_bounds_and_distributions, customized_routes
from setup_labels_and_bounds import emptyobject, generate_fuzzing_content

def run_carla_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port):

    customized_data = convert_x_to_customized_data(x, fuzzing_content, port)
    episode_max_time = fuzzing_arguments.episode_max_time
    ego_car_model = fuzzing_arguments.ego_car_model
    record_every_n_step = fuzzing_arguments.record_every_n_step
    debug = fuzzing_arguments.debug
    parent_folder = fuzzing_arguments.parent_folder
    route_type = fuzzing_arguments.route_type
    mean_objectives_across_generations_path = fuzzing_arguments.mean_objectives_across_generations_path
    town_name = sim_specific_arguments.town_name
    scenario = sim_specific_arguments.scenario
    direction = sim_specific_arguments.direction
    route_str = sim_specific_arguments.route_str
    scenario_file = sim_specific_arguments.scenario_file
    call_from_dt = dt_arguments.call_from_dt

    return run_carla_simulation_helper(customized_data,
    launch_server, episode_max_time, call_from_dt,
    town_name, scenario, direction, route_str, route_type, scenario_file, ego_car_model,
    record_every_n_step=record_every_n_step, debug=debug, counter=counter, parent_folder=parent_folder, mean_objectives_across_generations_path=mean_objectives_across_generations_path, fuzzing_arguments=fuzzing_arguments, dt_arguments=dt_arguments, sim_specific_arguments=sim_specific_arguments, fuzzing_content=fuzzing_content, x=x)


def run_carla_simulation_helper(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, route_type, scenario_file, ego_car_model, ego_car_model_path=None, rerun=False, record_every_n_step=2000, debug=0, counter=0, parent_folder='', mean_objectives_across_generations_path='', fuzzing_arguments=None, dt_arguments=None, sim_specific_arguments=None, fuzzing_content=None, x=None):

    arguments = arguments_info()
    arguments.record_every_n_step = record_every_n_step
    arguments.port = customized_data['port']
    arguments.debug = debug



    if ego_car_model == 'lbc':
        arguments.agent = 'scenario_runner/team_code/image_agent.py'
        arguments.agent_config = 'models/epoch=24.ckpt'
        # arguments.agent_config = 'models/stage2_0.01_augmented_epoch=11.ckpt'
        base_save_folder = 'collected_data_customized'
    elif ego_car_model == 'lbc_augment_ped':
        arguments.agent = 'scenario_runner/team_code/image_agent.py'

        arguments.agent_config = 'checkpoints/stage2_pretrained/town05_left_ped_bug_train_ped_bug_train_non_debug/epoch=0.ckpt'

        # arguments.agent_config = 'checkpoints/stage2_pretrained/town05_left_ped_bug_train_non_debug_2/epoch=0.ckpt'

        base_save_folder = 'collected_data_lbc_augment_ped'
    elif ego_car_model == 'lbc_augment':
        arguments.agent = 'scenario_runner/team_code/image_agent.py'

        # arguments.agent_config = 'checkpoints/stage2_pretrained/town05_left_vehicle_bug_train_non_debug/epoch=0.ckpt'

        arguments.agent_config = 'checkpoints/stage2_pretrained/town05_left_non_bug_train_non_debug/epoch=0.ckpt'

        # arguments.agent_config = 'checkpoints/stage2_pretrained/town05_left_ped_bug_train_non_debug_2/epoch=0.ckpt'

        base_save_folder = 'collected_data_lbc_augment'
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

    if ego_car_model_path:
        arguments.agent_config = ego_car_model_path


    if rerun:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, '/rerun', str(int(arguments.port)), str(call_from_dt)])
    else:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, str(int(arguments.port)), str(call_from_dt)])

    arguments.scenarios = scenario_file




    statistics_manager = StatisticsManager()



    # sample_factor is an integer between [1, 8]
    sample_factor = 5
    weather_index = customized_data['weather_index']


    # Laundry Stuff-------------------------------------------------------------
    arguments.weather_index = weather_index
    os.environ['WEATHER_INDEX'] = str(weather_index)



    os.environ['SAVE_FOLDER'] = make_hierarchical_dir([os.environ['SAVE_FOLDER'] + '/' + town_name, scenario, direction])
    arguments.save_folder = os.environ['SAVE_FOLDER']


    arguments.routes = 'leaderboard/data/customized_routes/' + '/'.join([town_name, scenario, direction]) + '/route_' + route_str + '.xml'
    os.environ['ROUTES'] = arguments.routes

    tmp_save_path = os.path.join(arguments.save_folder, 'route_'+route_str)

    # TBD: for convenience
    arguments.deviations_folder = tmp_save_path


    # extract waypoints along route
    tree = ET.parse(arguments.routes)
    route_waypoints = []



    # this iteration should only go once since we only keep one route per file
    for route in tree.iter("route"):
        route_id = route.attrib['id']
        route_town = route.attrib['town']

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


        objectives, loc, object_type, route_completion = estimate_objectives(tmp_save_path)

    # hack for correcting spawn locations:
    filename = 'tmp_folder/'+str(arguments.port)+'.pickle'
    with open(filename, 'rb') as f_in:
        all_final_generated_transforms = pickle.load(f_in)


    run_info = {}
    if parent_folder:
        is_bug = check_bug(objectives)
        bug_type, bug_str = classify_bug_type(objectives, object_type)
        if is_bug:
            with open(mean_objectives_across_generations_path, 'a') as f_out:
                f_out.write(str(counter)+','+bug_str+'\n')

        bug_folder = make_hierarchical_dir([parent_folder, 'bugs'])
        non_bug_folder = make_hierarchical_dir([parent_folder, 'non_bugs'])
        if is_bug:
            cur_folder = make_hierarchical_dir([bug_folder, str(counter)])
        else:
            cur_folder = make_hierarchical_dir([non_bug_folder, str(counter)])

        with open(cur_folder+'/'+'cur_info.pickle', 'wb') as f_out:
            pickle.dump(run_info, f_out)

        try:
            print('tmp_save_path, cur_folder', tmp_save_path, cur_folder)
            copy_tree(tmp_save_path, cur_folder)
        except:
            print('fail to copy from', tmp_save_path)
            traceback.print_exc()

        run_info = {
        'episode_max_time':fuzzing_arguments.episode_max_time,
        'ego_car_model':fuzzing_arguments.ego_car_model,
        'route_type':fuzzing_arguments.route_type,
        'call_from_dt':dt_arguments.call_from_dt,
        'town_name':sim_specific_arguments.town_name,
        'scenario':sim_specific_arguments.scenario,
        'direction':sim_specific_arguments.direction,
        'route_str':sim_specific_arguments.route_str,

        'waypoints_num_limit':fuzzing_content.search_space_info.waypoints_num_limit, 'num_of_static_max':fuzzing_content.search_space_info.num_of_static_max, 'num_of_pedestrians_max':fuzzing_content.search_space_info.num_of_pedestrians_max, 'num_of_vehicles_max':fuzzing_content.search_space_info.num_of_vehicles_max,
        # 'xl': problem.xl,
        # 'xu': problem.xu,

        'customized_center_transforms':fuzzing_content.customized_center_transforms,
        'parameters_min_bounds':fuzzing_content.parameters_min_bounds,
        'parameters_max_bounds':fuzzing_content.parameters_max_bounds,
        'labels': fuzzing_content.labels,
        'mask': fuzzing_content.mask,
        'customized_constraints': fuzzing_content.customized_constraints,

        'x': x,
        'objectives': objectives,
        'is_bug': is_bug,
        'bug_type': bug_type,
        'loc': loc,
        'object_type': object_type,
        'route_completion': route_completion,
        'all_final_generated_transforms': all_final_generated_transforms}


    return objectives, run_info


def parse_route_and_scenario(
    location_list, town_name, scenario, direction, route_str, scenario_file
):

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
    waypoint_template = (
        '\t<waypoint pitch="{}" roll="{}" x="{}" y="{}" yaw="{}" z="{}" />\n'
    )
    end_str = "</route>"

    wp_str = ""

    for x, y in location_list:
        wp = waypoint_template.format(pitch, roll, x, y, yaw, z)
        wp_str += wp

    final_str = start_str + wp_str + end_str

    folder = make_hierarchical_dir(
        ["leaderboard/data/customized_routes", town_name, scenario, direction]
    )

    pathlib.Path(folder + "/route_{}.xml".format(route_str)).write_text(
        TEMPLATE % final_str
    )

    # Parse Scenario
    x_0, y_0 = location_list[0]
    parse_scenario(scenario_file, town_name, route_str, x_0, y_0)

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
                                    "z": "0.0",
                                },
                                "transform": {
                                    "pitch": "0.0",
                                    "x": x_0_str,
                                    "y": y_0_str,
                                    "yaw": "270",
                                    "z": "0.0",
                                },
                            }
                        ],
                        "scenario_type": "Scenario12",
                    }
                ]
            }
        ]
    }

    with open(scenario_file, "w") as f_out:
        annotation_dict = json.dump(new_scenario, f_out, indent=4)


def initialize_carla_specific(fuzzing_arguments):

    route_info = customized_routes[fuzzing_arguments.route_type]

    town_name = route_info['town_name']
    scenario = 'Scenario12' # This is only for compatibility purpose
    direction = route_info['direction']
    route = route_info['route_id']
    location_list = route_info['location_list']


    scenario_file = initialize_tmp_scenario_file()


    route_str = str(route)
    if route < 10:
        route_str = '0'+route_str

    parse_route_and_scenario(location_list, town_name, scenario, direction, route_str, scenario_file)

    sim_specific_arguments = emptyobject(
    town_name=town_name,
    scenario=scenario,
    direction=direction,
    route_str=route_str,
    scenario_file=scenario_file,
    location_list=location_list)


    return sim_specific_arguments


def initialize_tmp_scenario_file():
    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    scenario_folder = 'scenario_files'
    if not os.path.exists('scenario_files'):
        os.mkdir(scenario_folder)
    scenario_file = scenario_folder+'/'+'current_scenario_'+time_str+'.json'

    return scenario_file



def estimate_objectives(save_path, default_objectives=np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.]), verbose=True):

    events_path = os.path.join(save_path, "events.txt")
    deviations_path = os.path.join(save_path, "deviations.txt")

    # set thresholds to avoid too large influence
    ego_linear_speed = 0
    min_d = 20
    offroad_d = 7
    wronglane_d = 7
    dev_dist = 0
    d_angle_norm = 1

    ego_linear_speed_max = 7
    dev_dist_max = 7

    is_offroad = 0
    is_wrong_lane = 0
    is_run_red_light = 0
    is_collision = 0

    with open(deviations_path, "r") as f_in:
        for line in f_in:
            type, d = line.split(",")
            d = float(d)
            if type == "min_d":
                min_d = np.min([min_d, d])
            elif type == "offroad_d":
                offroad_d = np.min([offroad_d, d])
            elif type == "wronglane_d":
                wronglane_d = np.min([wronglane_d, d])
            elif type == "dev_dist":
                dev_dist = np.max([dev_dist, d])
            elif type == "d_angle_norm":
                d_angle_norm = np.min([d_angle_norm, d])

    x = None
    y = None
    object_type = None

    infraction_types = [
        "collisions_layout",
        "collisions_pedestrian",
        "collisions_vehicle",
        "red_light",
        "on_sidewalk",
        "outside_lane_infraction",
        "wrong_lane",
        "off_road",
    ]

    try:
        with open(events_path) as json_file:
            events = json.load(json_file)
    except:
        print("events_path", events_path, "is not found")
        return default_objectives, (None, None), None
    infractions = events["_checkpoint"]["records"][0]["infractions"]
    status = events["_checkpoint"]["records"][0]["status"]

    route_completion = float(events["values"][1])

    for infraction_type in infraction_types:
        for infraction in infractions[infraction_type]:
            if "collisions" in infraction_type:
                typ = re.search(".*with type=(.*) and id.*", infraction)
                if verbose:
                    print(infraction, typ)
                if typ:
                    object_type = typ.group(1)
                loc = re.search(
                    ".*x=(.*), y=(.*), z=(.*), ego_linear_speed=(.*), other_actor_linear_speed=(.*)\)",
                    infraction,
                )
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    ego_linear_speed = float(loc.group(4))
                    other_actor_linear_speed = float(loc.group(5))

                    # only record valid collisions to promote valid collision bugs
                    if ego_linear_speed > 0.1:
                        is_collision = 1

            elif infraction_type == "off_road":
                loc = re.search(".*x=(.*), y=(.*), z=(.*)\)", infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    is_offroad = 1
            else:
                if infraction_type == "wrong_lane":
                    is_wrong_lane = 1
                elif infraction_type == "red_light":
                    is_run_red_light = 1
                loc = re.search(".*x=(.*), y=(.*), z=(.*)[\),]", infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))

    # limit impact of too large values
    ego_linear_speed = np.min([ego_linear_speed, ego_linear_speed_max])
    dev_dist = np.min([dev_dist, dev_dist_max])

    return (
        [
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
        ],
        (x, y),
        object_type,
        route_completion,
    )

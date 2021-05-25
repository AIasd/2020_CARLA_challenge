'''
Generate short route files by spliting the original 75 carla challenge routes into >= 50m pieces so we can add agent around the center of those cases.
'''
import sys
sys.path.append("../carla_0994_no_rss/PythonAPI/carla")
sys.path.append("scenario_runner")
sys.path.append("leaderboard")
import xml.etree.ElementTree as ET
import pathlib
from leaderboard.utils.route_parser import RouteParser
import os
import numpy as np

from customized_utils import parse_route_file


import carla
from srunner.scenariomanager.carla_data_provider import *
from leaderboard.utils.route_manipulation import interpolate_trajectory
import xml.etree.ElementTree as ET
from customized_utils import exit_handler
from carla_specific_utils.carla_specific import start_server
from carla_specific_utils.carla_specific_tools import create_transform
import atexit
import pickle







def get_trajectory(route_filename):
    tree = ET.parse(route_filename)
    for route in tree.iter("route"):

        waypoint_list = []  # the list of waypoints that can be found on this route
        for waypoint in route.iter('waypoint'):
            waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']),
                                                y=float(waypoint.attrib['y']),
                                                z=float(waypoint.attrib['z'])))
        trajectory = waypoint_list

    return trajectory

def get_center_transform_of_interpolated_trajectory(world, trajectory, waypoint_ratio=0.5):
    frame_rate = 10
    settings = world.get_settings()
    settings.fixed_delta_seconds = 1.0 / frame_rate
    settings.synchronous_mode = True
    world.apply_settings(settings)

    # spectator = CarlaDataProvider.get_world().get_spectator()
    # spectator.set_transform(carla.Transform(carla.Location(x=0, y=0,z=20), carla.Rotation(pitch=-90)))

    # Wait for the world to be ready
    if world.get_settings().synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick()


    _, route = interpolate_trajectory(world, trajectory)

    ind = np.min([int(len(route)*waypoint_ratio), len(route)-1])
    loc = route[ind][0].location
    center_transform = create_transform(loc.x, loc.y, 0, 0, 0, 0)

    return center_transform



if __name__ == '__main__':
    TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
    <routes>
    %s
    </routes>"""


    route_inds_str = [str(i) if i >= 10 else '0'+str(i) for i in range(76)]
    route_files = ['leaderboard/data/routes/route_'+r_i_str+'.xml' for r_i_str in route_inds_str]


    # run a server to estimate the center transform
    client = None
    port = '2003'
    host = 'localhost'
    # os.environ["DISPLAY"] = ''
    start_server(port)
    while True:
        try:
            client = carla.Client(host, int(port))
            break
        except:
            logging.exception("__init__ error")
            traceback.print_exc()


    atexit.register(exit_handler, [port])
    CarlaDataProvider.set_client(client)

    initialization = False
    route_to_center_d = {}

    for route_id, route_file in enumerate(route_files):
        config_list = parse_route_file(route_file, route_length_lower_bound=50)

        for config_id, config in enumerate(config_list):
            _, town_name, transform_list = config


            # load world when necessary
            if not initialization or CarlaDataProvider.get_map().name != town_name:
                while True:
                    try:
                        world = client.load_world(town_name)
                        break
                    except:
                        logging.exception("_load_and_wait_for_world error")
                        traceback.print_exc()
                        start_server(port)
                        client = carla.Client(host, int(port))
                CarlaDataProvider.set_world(world)

            initialization = False


            # save this center_transform
            start_str = '<route id="{}" town="{}" map="{}">\n'.format(route_id, town_name, town_name)
            waypoints_str = ''
            for transform in transform_list:
                waypoint_template = '    <waypoint x="{}" y="{}" z="{}" pitch="{}" yaw="{}" roll="{}" />\n'
                waypoints_str += waypoint_template.format(*transform)
            end_str = '</route>'
            route_str = start_str+waypoints_str+end_str

            parent_folder = 'leaderboard/data/new_routes'
            if not os.path.exists(parent_folder):
                os.mkdir(parent_folder)

            route_id_str = str(route_id)
            if route_id < 10:
                route_id_str = '0'+route_id_str

            route_name = 'route_{}_{}.xml'.format(route_id_str, config_id)
            route_path = parent_folder + '/' + route_name
            pathlib.Path(route_path).write_text(TEMPLATE % route_str)

            trajectory = get_trajectory(route_path)
            center_transform = get_center_transform_of_interpolated_trajectory(world, trajectory)

            route_to_center_d[route_name] = (center_transform.location.x, center_transform.location.y)
            print(route_name, route_to_center_d[route_name])

    # hack: this filename is hardcoded
    with open('route_to_center_d.pkl', 'wb') as f_out:
        pickle.dump(route_to_center_d, f_out, pickle.HIGHEST_PROTOCOL)

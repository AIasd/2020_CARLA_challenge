'''
Generate short route files by spliting the original 75 carla challenge routes into >= 50m pieces so we can add agent around the center of those cases.
'''
import sys
sys.path.append("../carla_0994_no_rss/PythonAPI/carla")
sys.path.append("scenario_runner")
import xml.etree.ElementTree as ET
import pathlib
from leaderboard.leaderboard.utils.route_parser import RouteParser
import os
import numpy as np

from customized_utils import parse_route_file


TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
%s
</routes>"""
# scenario_file = 'leaderboard/data/customized_scenarios.json'
# world_annotations = RouteParser.parse_annotations_file(scenario_file)







route_inds_str = [str(i) if i >= 10 else '0'+str(i) for i in range(76)]
route_files = ['leaderboard/data/routes/route_'+r_i_str+'.xml' for r_i_str in route_inds_str]

for route_id, route_file in enumerate(route_files):
    config_list = parse_route_file(route_file)

    for config_id, config in enumerate(config_list):
        _, town_name, transform_list = config

        start_str = '<route id="{}" town="{}">\n'.format(route_id, town_name)
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
        pathlib.Path(parent_folder+'/route_{}_{}.xml'.format(route_id_str, config_id)).write_text(TEMPLATE % route_str)

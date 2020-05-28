import os
import json
import re
import numpy as np

eps = 0.001
routes_list = [19, 29, 39, 49, 59, 69]
weather_list = [11, 19]
infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'vehicle_blocked']
for weather_id in weather_list:
    for route in routes_list:
        parent_folder = 'collected_data'+'/'+'route_'+str(route)+'_'+str(weather_id)
        events_path = parent_folder + '/' + 'events.txt'
        measurements_path = parent_folder + '/' + 'measurements.csv'
        measurements_loc_path = parent_folder + '/' + 'measurements_loc.csv'
        new_measurements_path = parent_folder + '/' + 'driving_log.csv'

        events_list = []

        with open(events_path) as json_file:
            events = json.load(json_file)

        infractions = events['_checkpoint']['records'][0]['infractions']
        for infraction_type in infraction_types:
            for infraction in infractions[infraction_type]:
                loc = re.search('.*x=(.*), y=(.*), z=(.*)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    events_list.append((x, y, infraction_type))

        num_of_lines = 0
        with open(measurements_path, 'r') as f_in:
            with open(measurements_loc_path, 'r') as f_loc_in:
                measurements = f_in.read().split('\n')
                locations = f_loc_in.read().split('\n')
        with open(new_measurements_path, 'w') as f_out:
            for i in range(len(measurements)):
                if i == 0:
                    # TBD: include topdown in the title_row
                    title_row = ','.join(['FrameId', 'far_command', 'speed', 'steering', 'throttle', 'brake', 'center', 'left', 'right', 'x', 'y', 'Misbehavior', 'Crashed'])
                    new_line = title_row
                else:
                    # we use crashed to represent all violations
                    crashed = 0

                    m_i = measurements[i].split(',')
                    l_i = locations[i].split(',')
                    if m_i != [''] and l_i != ['']:
                        x, y = float(l_i[0]), float(l_i[1])
                        misbehavior_name = ''

                        for event in events_list:
                            x_e, y_e, event_name = event
                            if np.abs(x_e-x) < eps and np.abs(y_e-y) < eps:
                                misbehavior_name += '_'+event_name
                                crashed = 1
                                print(misbehavior_name, i)
                        data_row = m_i+l_i+[misbehavior_name, str(crashed)]
                        new_line = ','.join(data_row)
                f_out.write(new_line+'\n')

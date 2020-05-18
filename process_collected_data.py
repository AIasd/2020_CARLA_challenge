import os
import json
import re
import numpy as np

eps = 1e-2
id_list = [19]
weather_list = [0]
infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light']
for weather_id in weather_list:
    for id in id_list:
        parent_folder = 'collected_data'+'/'+'route_'+str(id)+'_'+str(weather_id)
        measurements_path = parent_folder + '/' + 'measurements.csv'
        events_path = 'collected_data' + '/' + 'route_'+str(id)+'_'+str(weather_id) + '.txt'
        new_measurements_path = parent_folder + '/' + 'new_measurements.csv'

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
        with open(new_measurements_path, 'w') as f_out:
            with open(measurements_path, 'r') as f_in:
                for i, line in enumerate(f_in):
                    if i == 0:
                        title_row = line.split(',')
                        title_row += ['misbehavior']
                        new_line = ','.join(title_row)
                    else:
                        data_row = line.split(',')
                        _, x, y = data_row[0], float(data_row[1]), float(data_row[2])
                        misbehavior_name = ''

                        for event in events_list:
                            x_e, y_e, event_name = event
                            if np.abs(x_e-x) < eps and np.abs(y_e-y) < eps:
                                misbehavior_name += '_'+event_name
                                print(misbehavior_name, frame)
                        data_row += [misbehavior_name]
                        new_line = ','.join(data_row)
                    f_out.write(new_line+'\n')

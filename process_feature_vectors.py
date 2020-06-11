import pandas as pd
import numpy as np
import os


data_dir = 'collected_data'

infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane']


weather_indexes = [15]
routes = [i for i in range(30)]
routes.remove(13)
model_name = 'SAE'

behaviors_list = []
features_list = []
behaviors_names = []
misbehavior_names = []

for weather_id in weather_indexes:
    for route in routes:
        route_str = str(route)
        if route < 10:
            route_str = '0'+route_str

        data_df = pd.read_csv(os.path.join(data_dir, 'route_'+route_str+'_'+str(weather_id), 'driving_log.csv'))
        behaviors_list.append(data_df['behaviors'])
        behaviors_names.extend(data_df['behaviors_names'])
        features_list.append(np.load(os.path.join(data_dir, 'route_'+route_str+'_'+str(weather_id), model_name+'_features.npy')))
        misbehavior_names.extend(data_df['Misbehavior'])

behaviors = np.concatenate(behaviors_list, axis=0)
features = np.concatenate(features_list, axis=0)
print(behaviors.shape, features.shape, len(behaviors_names), len(misbehavior_names))
np.savez(data_dir+'/'+model_name+'_'+'_'.join([str(w_id) for w_id in weather_indexes]), behaviors=behaviors, features=features, behaviors_names=behaviors_names, misbehavior_names=misbehavior_names)

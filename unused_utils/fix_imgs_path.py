import os
import pandas as pd
folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized/double_npcs_peds_only'


for f in os.listdir(folder):
    subfolder = os.path.join(folder, f)
    if os.path.isdir(subfolder):
        path = os.path.join(subfolder, 'driving_log.csv')
        data_df = pd.read_csv(path)

        x_center = data_df['center'].values
        x_left = data_df['left'].values
        x_right = data_df['right'].values

        for i in range(x_center.shape[0]):
            img_i = x_right[i][1:]

            x_center[i] = 'rgb/'+img_i
            x_left[i] = 'rgb_left/'+img_i
            x_right[i] = 'rgb_right/'+img_i

        data_df['center'] = x_center
        data_df['left'] = x_left
        data_df['right'] = x_right
        # print(data_df['center'].values)
        data_df.to_csv(path)

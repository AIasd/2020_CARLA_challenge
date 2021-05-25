import pandas as pd
import numpy as np
df = pd.read_pickle("initial_observations_one_ped_const_weather.pkl")
print(df)
print(df.loc[0])
print(np.sum(df['is_bug']))

column_labels = [['ego_car_perturbation_x_'+str(i), 'ego_car_perturbation_y_'+str(i)] for i in range(5)]
column_labels = [label for subl in column_labels for label in subl]+['friction', 'num_of_weathers', 'num_of_static', 'num_of_vehicles']
print(column_labels)
df.drop(columns=column_labels, inplace=True)
print(df.loc[0])
df.to_pickle("new_initial_observations_one_ped_const_weather.pkl")

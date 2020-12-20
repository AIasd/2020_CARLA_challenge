'

# ---------------------------------------------------------
# get misbehavior data
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict

infraction_types = ['_'+entry for entry in ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'off_road']]
sub_dir_misbehavior = defaultdict(lambda:{'misbehavior_names':[], 'behaviors':[]})
closest_pedestrian_distance_by_sub_dir = defaultdict(lambda:[])

total_data_dir = 'collected_data_customized/customized_0'

x_center = None
x_left = None
x_right = None
y = None

speed = None
steering = None
throttle = None
brake = None

Misbehavior = []

for sub_dir in sorted(os.listdir(total_data_dir)):
    data_dir = os.path.join(total_data_dir, sub_dir)
    if os.path.isdir(data_dir):
        datafile = os.path.join(data_dir, 'driving_log.csv')
        if os.path.exists(datafile):
            data_df = pd.read_csv(datafile)

            x_center_i = data_df['center'].values
            y_i = data_df['behaviors'].values
            x_left_i = data_df['left'].values
            x_right_i = data_df['right'].values

            speed_i = data_df['speed']
            steering_i = data_df['steering']
            throttle_i = data_df['throttle']
            brake_i = data_df['brake']

            for j in range(x_center_i.shape[0]):
                x_center_i[j] = os.path.join(sub_dir, x_center_i[j])
                x_left_i[j] = os.path.join(sub_dir, x_left_i[j])
                x_right_i[j] = os.path.join(sub_dir, x_right_i[j])


            if x_center is None:
                x_center = x_center_i.copy()
                y = y_i.copy()
                x_left = x_left_i.copy()
                x_right = x_right_i.copy()

                speed = speed_i.copy()
                steering = steering_i.copy()
                throttle = throttle_i.copy()
                brake = brake_i.copy()

            else:
                x_center = np.concatenate((x_center, x_center_i), axis=0)
                y = np.concatenate((y, y_i), axis=0)
                x_left = np.concatenate((x_left, x_left_i), axis=0)
                x_right = np.concatenate((x_right, x_right_i), axis=0)

                speed = np.concatenate((speed, speed_i), axis=0)
                steering = np.concatenate((steering, steering_i), axis=0)
                throttle = np.concatenate((throttle, throttle_i), axis=0)
                brake = np.concatenate((brake, brake_i), axis=0)

            Misbehavior.extend(data_df['Misbehavior'])
            for infraction_type in infraction_types:
                # print(np.array(data_df['Misbehavior']))
                if infraction_type in np.array(data_df['Misbehavior']):
                    sub_dir_misbehavior[sub_dir]['misbehavior_names'].append(infraction_type)
            sub_dir_misbehavior[sub_dir]['behaviors'] = np.array(data_df['behaviors'])

            # if 1 in np.array(data_df['behaviors']):
            #     print(sub_dir)
        else:
            print(datafile, 'does not exist')

Misbehavior = np.array(Misbehavior)
chosen_inds = (y == 0) | (y == 1)


# inds_collision_layout = (Misbehavior=='_collisions_layout')
inds_collisions_pedestrian = (Misbehavior=='_collisions_pedestrian')
# inds_collisions_vehicle = (Misbehavior=='_collisions_vehicle')
# inds_red_light = (Misbehavior=='_red_light')
# inds_wrong_lane = (Misbehavior=='_wrong_lane')
# inds_off_road = (Misbehavior=='_off_road')


# print('collisions_layout', len(x_center[inds_collision_layout]))
print('collisions_pedestrian', len(x_center[inds_collisions_pedestrian]))
# print('collisions_vehicle', len(x_center[inds_collisions_vehicle]))
# print('red_light', len(x_center[inds_red_light]))
# print('wrong_lane', len(x_center[inds_wrong_lane]))
# print('off_road', len(x_center[inds_off_road]))


x_center = x_center[chosen_inds]
y = y[chosen_inds]

print('y==0:', y[y==0].shape)
print('y==1:', y[y==1].shape)


# ---------------------------------------------------------
# get measurements data



closest_pedestrian_distance = []
# closest_vehicle_distance = []
# closest_red_tl_distance = []
# is_pedestrian_hazard = []
# is_vehicle_hazard = []
# is_red_tl_hazard = []
forward_speed_list = []
throttle_list = []
steer_list = []
brake_list = []



for sub_dir in sorted(os.listdir(total_data_dir)):
    data_dir = os.path.join(total_data_dir, sub_dir)
    if os.path.isdir(data_dir):
        measurements_folder = os.path.join(data_dir, 'measurements')
        for measurement_name in os.listdir(measurements_folder):
            measurement_path = os.path.join(measurements_folder, measurement_name)
            with open(measurement_path, 'r') as f_in:
                measurement = json.load(f_in)
                closest_pedestrian_distance.append(measurement['closest_pedestrian_distance'])
                closest_pedestrian_distance_by_sub_dir[sub_dir].append(measurement['closest_pedestrian_distance'])
                # closest_vehicle_distance.append(measurement['closest_vehicle_distance'])
                # closest_red_tl_distance.append(measurement['closest_red_tl_distance'])


                # is_pedestrian_hazard.append(measurement['is_pedestrian_hazard'])
                # is_vehicle_hazard.append(measurement['is_vehicle_hazard'])
                # is_red_tl_hazard.append(measurement['is_red_tl_hazard'])
                forward_speed_list.append(measurement['forward_speed'])
                throttle_list.append(measurement['throttle'])
                steer_list.append(measurement['steer'])
                brake_list.append(measurement['brake'])


closest_pedestrian_distance = np.array(closest_pedestrian_distance)[chosen_inds]
# closest_vehicle_distance = np.array(closest_vehicle_distance)[chosen_inds]
# closest_red_tl_distance = np.array(closest_red_tl_distance)[chosen_inds]

forward_speed = np.array(forward_speed_list)[chosen_inds]
throttle = np.array(throttle_list)[chosen_inds]
steer = np.array(steer_list)[chosen_inds]
brake = np.array(brake_list)[chosen_inds]

forward_speed_0 = forward_speed[y==0]
throttle_0 = throttle[y==0]
steer_0 = steer[y==0]
brake_0 = brake[y==0]
forward_speed_1 = forward_speed[y==1]
throttle_1 = throttle[y==1]
steer_1 = steer[y==1]
brake_1 = brake[y==1]



closest_pedestrian_distance_y_0 = closest_pedestrian_distance[y==0]
closest_pedestrian_distance_y_1 = closest_pedestrian_distance[y==1]

print(np.mean(closest_pedestrian_distance_y_0))
print(np.mean(closest_pedestrian_distance_y_1))


def draw_dist(data_0, data_1, plt_path, xlabel, bin_num, y_max=None):
    plt.hist(data_0, bin_num, density=True, facecolor='g', alpha=0.25, label='normal')

    plt.hist(data_1, bin_num, density=True, facecolor='r', alpha=0.25, label='anomaly')


    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.title('Histogram of '+xlabel, fontsize=20)
    # plt.xlim(0, 50)
    # if y_max:
    #     plt.ylim(0, y_max)
    plt.grid(True)
    plt.legend(prop={'size': 15})
    plt.savefig(plt_path)
    plt.close()


no_50 = False
mode = 'inspect_misbehaviors'
if no_50 == True:
    closest_pedestrian_distance_y_0 = closest_pedestrian_distance_y_0[closest_pedestrian_distance_y_0<49.9]
    closest_pedestrian_distance_y_1 = closest_pedestrian_distance_y_1[closest_pedestrian_distance_y_1<49.9]
# the histogram of the data
if mode == 'draw_dist':
    draw_dist(closest_pedestrian_distance_y_0, closest_pedestrian_distance_y_1, 'collected_data_customized/dist_to_ped_50.png', 'Distance to closest pedestrian', 50, 0.2)
    draw_dist(forward_speed_0, forward_speed_1, 'collected_data_customized/forward_speed_50.png', 'Forward Speed', 50)
    draw_dist(throttle_0, throttle_1, 'collected_data_customized/throttle_50.png', 'Throttle', 50)
    draw_dist(steer_0, steer_1, 'collected_data_customized/steer_50.png', 'Steering Angle', 50)
    draw_dist(brake_0, brake_1, 'collected_data_customized/brake_50.png', 'Brake', 2)


elif mode == 'roc':
    num_normal_total = len(closest_pedestrian_distance_y_0)
    num_anomaly_total = len(closest_pedestrian_distance_y_1)
    # num_both_total = num_normal_total + num_anomaly_total
    tpr_list = []
    fpr_list = []

    for th in np.arange(0, 50.2, 0.2):

        num_normal = np.sum(closest_pedestrian_distance_y_0 <= th)
        num_anomaly = np.sum(closest_pedestrian_distance_y_1 <= th)
        # num_both = num_normal + num_anomaly

        tpr = num_anomaly / num_anomaly_total
        fpr = num_normal / num_normal_total
        # recall = num_anomaly / num_anomaly_total

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        print(th, tpr, fpr, num_anomaly, num_normal)

    print(len(tpr_list), tpr_list[:5])
    print(len(fpr_list), fpr_list[:5])
    plt.plot(fpr_list, tpr_list, label='distance')
    plt.plot(np.arange(0, 1.2, 0.2), np.arange(0, 1.2, 0.2), label='random')
    plt.legend(prop={'size': 15})
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Anomaly Frame Detection using Distance', fontsize=20)
    plt.savefig('collected_data_customized/dist_anomaly_roc_50.png')
    # plt.show()
elif mode=='inspect_misbehaviors':
    normal_list = []
    anomaly_list = []
    for sub_dir, d in sub_dir_misbehavior.items():
        closest_pedestrian_distance_by_sub_dir[sub_dir] = np.array(closest_pedestrian_distance_by_sub_dir[sub_dir])
        if '_collisions_pedestrian' in d['misbehavior_names']:
            assert len(closest_pedestrian_distance_by_sub_dir[sub_dir]) == len(d['behaviors']), str(len(closest_pedestrian_distance_by_sub_dir[sub_dir]))+','+str(len(d['behaviors']))

            ind_normal = d['behaviors'] == 0
            ind_anomaly = d['behaviors'] == 1
            if np.sum(ind_normal) > 0:
                min_normal = np.min(closest_pedestrian_distance_by_sub_dir[sub_dir][ind_normal])
                min_anomaly = np.min(closest_pedestrian_distance_by_sub_dir[sub_dir][ind_anomaly])
                normal_list.append((min_normal, sub_dir))
                anomaly_list.append((min_anomaly, sub_dir))
        else:
            min_normal = np.min(closest_pedestrian_distance_by_sub_dir[sub_dir])
            normal_list.append((min_normal, sub_dir))
    print('normal :', list(enumerate(sorted(normal_list)[:200])))

    # print('anomaly :', list(enumerate(sorted(anomaly_list)[:500])))

    for i, d in enumerate(closest_pedestrian_distance_by_sub_dir['route_13_5']):
        print(i, d)

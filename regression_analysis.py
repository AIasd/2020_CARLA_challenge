'''
tomorrow TBD:
1.try the other 2 scenarios here and think about impact of their variables
# 2.simplify object_type to be more easily to be onehot encoded
3.integrate remove/include into ga_fuzzing and run them to compare with vanilla nsga2
4.improve adv in rerun to reflect remove/include
5.try some heuristic from MTFuzz for adv to potentially produce better performance for this particular problem?
6.try different config for town_05_front to get more out-of-road error? Or maybe improve the objective?






1.carefully design the fields chosen (perturbation, categorical)
2.analyze important features using MtFuzz hotbytes method

3.rethink about search objectives as well as optimization loss

4.redesign/select scenarios to run for potentially better property

5.consider MtFuzz method of leveraging hotbytes for generating new inputs


Methods:
0.better objective function for NN?
1.better NN for better than random performance? figure out training loss decreases but testing loss increases; NN to rank with fine-grained signal?? (right now we only use binary and have not considered different types of bugs)
2.RankNet to rank
3.Adv perturbation on generated cases
4.Inversion Model


Uniqueness
- cases where enough uniqueness exist and consider eliminate cases where duplicates exist??? (rank methods)
- adv with extra projection
- inversion model with extra projection / penalty

show our method is better in previous work def, then show our method works better in generalized cases

Fix:
1.randomness / reproducibility
2.better / increasing(decreasing) objective that more correlated with the bugs
3.continuous red traffic light loss???


Fixing:
1.show finetuning with buggy data from adv will make the model harder to be found bugs than finetuning with buggy data from random


Ex:
1.show bug that happens with 1 vehicle + 1 ped while removing either one won't happen to illustrate the extra bugs found


2.multiple run for statistical significance
3.make a video for better demo



# define new bug score
# if it is a bug, y = 0 + max_i ( f_i / f_i_range over i that is not the bug's cause)
# else: y = 1 + max_i ( f_i / f_i_range over all i)
# maybe consider to improve the quality of score by redefining some such that
# nsga2 can indeed improve these scores / find more bugs over time



'''
import sys
import os
sys.path.append('pymoo')
carla_root = '../carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')
sys.path.append('.')
sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')
sys.path.append('scenario_runner')
sys.path.append('carla_project')
sys.path.append('carla_project/src')


import pickle
import re

import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier


from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression


def reformat(cur_info):
    objectives = cur_info['objectives']
    is_bug = cur_info['is_bug']

    ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, is_collision = objectives
    accident_x, accident_y = cur_info['loc']


    # route_completion = cur_info['route_completion']

    # result_info = [ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, accident_x, accident_y, is_bug, route_completion]


    data, x, xl, xu, mask, labels = cur_info['data'], cur_info['x'][:-1], cur_info['xl'], cur_info['xu'], cur_info['mask'], cur_info['labels']

    assert len(x) == len(xl)



    # town_05_right
    # labels_to_encode = ['num_of_weathers', 'num_of_static', 'num_of_pedestrians', 'num_of_vehicles', 'num_of_pedestrian_types_0', 'num_of_vehicle_types_0', 'vehicle_waypoint_follower_0', 'vehicle_avoid_collision_0', 'num_of_vehicle_colors_0']




    # town_04_front
    labels_to_encode = ['num_of_weathers']

    labels_to_remove = ['num_of_static', 'num_of_pedestrians', 'num_of_vehicles',
     'num_of_pedestrian_types_0', 'num_of_pedestrian_types_1',
     'num_of_pedestrian_types_2', 'num_of_pedestrian_types_3',
     'num_of_pedestrian_types_4', 'num_of_pedestrian_types_5',
     'num_of_pedestrian_types_6', 'num_of_pedestrian_types_7',
     'num_of_pedestrian_types_8', 'num_of_pedestrian_types_9',
     'num_of_vehicle_types_0', 'vehicle_waypoint_follower_0',
     'vehicle_avoid_collision_0', 'num_of_vehicle_colors_0',
     'num_of_vehicle_types_1', 'vehicle_waypoint_follower_1',
     'vehicle_avoid_collision_1', 'num_of_vehicle_colors_1',
     'num_of_vehicle_types_2', 'vehicle_waypoint_follower_2',
     'vehicle_avoid_collision_2', 'num_of_vehicle_colors_2',
     'num_of_vehicle_types_3', 'vehicle_waypoint_follower_3',
     'vehicle_avoid_collision_3', 'num_of_vehicle_colors_3',
     'num_of_vehicle_types_4', 'vehicle_waypoint_follower_4',
     'vehicle_avoid_collision_4', 'num_of_vehicle_colors_4',
     'num_of_vehicle_types_5', 'vehicle_waypoint_follower_5',
     'vehicle_avoid_collision_5', 'num_of_vehicle_colors_5',
     'num_of_vehicle_types_6', 'vehicle_waypoint_follower_6',
     'vehicle_avoid_collision_6', 'num_of_vehicle_colors_6',
     'num_of_vehicle_types_7', 'vehicle_waypoint_follower_7',
     'vehicle_avoid_collision_7', 'num_of_vehicle_colors_7',
     'num_of_vehicle_types_8', 'vehicle_waypoint_follower_8',
     'vehicle_avoid_collision_8', 'num_of_vehicle_colors_8',
     'num_of_vehicle_types_9', 'vehicle_waypoint_follower_9',
     'vehicle_avoid_collision_9', 'num_of_vehicle_colors_9']




    # town_05_front
    # ['num_of_weathers', 'num_of_static', 'num_of_pedestrians', 'num_of_vehicles',
    #  'num_of_pedestrian_types_0', 'num_of_vehicle_types_0',
    #  'vehicle_waypoint_follower_0', 'vehicle_avoid_collision_0',
    #  'num_of_vehicle_colors_0', 'num_of_vehicle_types_1',
    #  'vehicle_waypoint_follower_1', 'vehicle_avoid_collision_1',
    #  'num_of_vehicle_colors_1', 'num_of_vehicle_types_2',
    #  'vehicle_waypoint_follower_2', 'vehicle_avoid_collision_2',
    #  'num_of_vehicle_colors_2']


    # labels_to_encode = ['num_of_weathers', 'num_of_pedestrian_types_0', 'num_of_vehicle_types_0',
    #  'vehicle_waypoint_follower_0', 'vehicle_avoid_collision_0',
    #  'num_of_vehicle_colors_0', 'num_of_vehicle_types_1',
    #  'vehicle_waypoint_follower_1', 'vehicle_avoid_collision_1',
    #  'num_of_vehicle_colors_1', 'num_of_vehicle_types_2',
    #  'vehicle_waypoint_follower_2', 'vehicle_avoid_collision_2',
    #  'num_of_vehicle_colors_2']
    # labels_to_remove = []



    def encode_and_remove_fields(x, mask, labels, labels_to_remove, labels_to_encode):
        from sklearn.preprocessing import OneHotEncoder
        # from object_types import weather_names, vehicle_colors, pedestrian_types, vehicle_types

        weather_names = ['ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset', 'WetNoon', 'WetSunset', 'MidRainyNoon', 'MidRainSunset', 'WetCloudyNoon', 'WetCloudySunset', 'HardRainNoon', 'HardRainSunset', 'SoftRainNoon', 'SoftRainSunset', 'ClearNight', 'CloudyNight', 'WetNight', 'MidRainNight', 'WetCloudyNight', 'HardRainNight', 'SoftRainNight']



        # walker modifiable attributes: speed: float
        pedestrian_types = ['walker.pedestrian.00'+f'{i:02d}' for i in range(1, 14)]


        # vehicle types
        # car
        car_types = ['vehicle.audi.a2',
        'vehicle.audi.tt',
        'vehicle.mercedes-benz.coupe',
        'vehicle.bmw.grandtourer',
        'vehicle.audi.etron',
        'vehicle.nissan.micra',
        'vehicle.lincoln.mkz2017',
        'vehicle.tesla.cybertruck',
        'vehicle.dodge_charger.police',
        'vehicle.tesla.model3',
        'vehicle.toyota.prius',
        'vehicle.seat.leon',
        'vehicle.nissan.patrol',
        'vehicle.mini.cooperst',
        'vehicle.jeep.wrangler_rubicon',
        'vehicle.mustang.mustang',
        'vehicle.volkswagen.t2',
        'vehicle.chevrolet.impala',
        'vehicle.citroen.c3']

        large_car_types = ['vehicle.carlamotors.carlacola']

        # motorcycle
        motorcycle_types = ['vehicle.yamaha.yzf',
        'vehicle.harley-davidson.low_rider',
        'vehicle.kawasaki.ninja']

        # cyclist
        cyclist_types = ['vehicle.bh.crossbike',
        'vehicle.gazelle.omafiets',
        'vehicle.diamondback.century']

        vehicle_types = car_types + large_car_types + motorcycle_types + cyclist_types


        # vehicle colors
        # black, white, gray, silver, blue, red, brown, gold, green, tan, orange
        vehicle_colors = ['(0, 0, 0)',
        '(255, 255, 255)',
        '(220, 220, 220)',
        '(192, 192, 192)',
        '(0, 0, 255)',
        '(255, 0, 0)',
        '(165,42,42)',
        '(255,223,0)',
        '(0,128,0)',
        '(210,180,140)',
        '(255,165,0)']




        keywords = {'num_of_weathers': len(weather_names), 'num_of_vehicle_colors': len(vehicle_colors), 'num_of_pedestrian_types': len(pedestrian_types), 'num_of_vehicle_types': len(vehicle_types)}

        keywords = {'num_of_weathers': len(weather_names)}



        x = np.array(x).astype(np.float)
        inds_to_remove = []
        for label in labels_to_remove:
            ind = labels.index(label)
            inds_to_remove.append(ind)
        inds_to_keep = list(set(range(len(x))) - set(inds_to_remove))
        x = x[inds_to_keep]
        mask = np.array(mask)[inds_to_keep].tolist()
        labels = np.array(labels)[inds_to_keep].tolist()


        encode_fields = []
        inds_to_encode = []
        for label in labels_to_encode:
            for k, v in keywords.items():
                if k in label:
                    ind = labels.index(label)
                    inds_to_encode.append(ind)

                    encode_fields.append(v)
                    break
        inds_non_encode = list(set(range(len(x))) - set(inds_to_encode))

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        m = len(encode_fields)
        data_for_fit_encode = np.zeros((int(np.sum(encode_fields)), m))
        counter = 0
        for i, encode_field in enumerate(encode_fields):
            for j in range(encode_field):
                data_for_fit_encode[counter, i] = j
                counter += 1
        enc.fit(data_for_fit_encode)

        embed = np.array([x[inds_to_encode].astype(np.int)])
        embed = enc.transform(embed)[0]

        x = np.concatenate([embed, x[inds_non_encode]]).astype(np.float)
        return x

    # x = x[mask!='int']
    x = encode_and_remove_fields(x, mask, labels, labels_to_remove, labels_to_encode)

    return x, objectives, int(is_bug)



class NN_EnsembleClassifier:
    def __init__(self, num_of_nets=1):
        self.num_of_nets = num_of_nets
        self.nets = []
        for _ in range(self.num_of_nets):
            net = MLPClassifier(solver='lbfgs', activation='tanh', max_iter=10000)
            self.nets.append(net)
    def fit(self, X_train, y_train):
        for i in range(self.num_of_nets):
            net = self.nets[i]
            net.fit(X_train, y_train)
    def score(self, X_test, y_test):
        s_list = []
        for net in self.nets:
            s = net.score(X_test, y_test)
            s_list.append(s)
        s_np = np.array(s_list)
        return np.mean(s_np)
    def predict(self, X_test):
        s_list = []
        for net in self.nets:
            s = net.predict(X_test)
            s_list.append(s)
        s_np = np.array(s_list)
        # print(s_np)
        prediction = stats.mode(s_np, axis=0)[0][0]
        # print(prediction)
        return prediction
    def predict_proba(self, X_test):
        s_list = []
        for net in self.nets:
            s = net.predict_proba(X_test)
            s_list.append(s)
        s_np = np.array(s_list)
        # print(s_np)
        prediction = np.mean(s_np, axis=0)
        # print(prediction)
        return prediction


def regression_analysis(X, is_bug_list, objective_list, cutoff, cutoff_en, trial_num):

    y = objective_list[:, 1]
    print(X.shape, y.shape)

    X_train, X_test = X[:cutoff], X[cutoff:cutoff_end]
    y_train, y_test = y[:cutoff], y[cutoff:cutoff_end]
    standardize = StandardScaler()
    X_train = standardize.fit_transform(X_train)
    X_test = standardize.transform(X_test)

    ind_0 = y_test==0
    ind_1 = y_test==1

    print(np.sum(y_train<0.5), np.sum(y_train>0.5), np.sum(y_test<0.5), np.sum(y_test>0.5))

    names = ["Neural Net", "Linear Regression"]
    regressors = [
        MLPRegressor(solver='lbfgs', activation='tanh', max_iter=10000),
        LinearRegression()
        ]

    performance = {name:[] for name in names}
    for i in range(trial_num):
        print(i)
        for name, clf in zip(names, regressors):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            y_pred = clf.predict(X_test)
            # print(clf.predict_proba(X_test))

            performance[name].append(score)

            print(name)
            print(y_pred)
            print(y_test)

    for name in names:
        print(name, np.mean(performance[name]), np.std(performance[name]))


def classification_analysis(X, is_bug_list, objective_list, cutoff, cutoff_en, trial_num):

    # from matplotlib import pyplot as plt
    # plt.hist(objective_list[:, 1])
    # plt.show()
    print(np.mean(objective_list[:, 1]), np.median(objective_list[:, 1]))

    y = is_bug_list
    # y = (objective_list[:, 1] < 2.5).astype(np.int)
    # y = (objective_list[:, -1] == 1).astype(np.int)
    print(np.sum(is_bug_list), np.sum(objective_list[:, -1] == 1))
    print(X.shape, y.shape)

    X_train, X_test = X[:cutoff], X[cutoff:cutoff_end]
    y_train, y_test = y[:cutoff], y[cutoff:cutoff_end]
    standardize = StandardScaler()
    X_train = standardize.fit_transform(X_train)
    X_test = standardize.transform(X_test)




    ind_0 = y_test==0
    ind_1 = y_test==1

    print(np.sum(y_train<0.5), np.sum(y_train>0.5), np.sum(y_test<0.5), np.sum(y_test>0.5))

    names = ["Nearest Neighbors", "Neural Net", "AdaBoost", "Random", "NN ensemble"]

    classifiers = [
        KNeighborsClassifier(5),
        MLPClassifier(solver='lbfgs', activation='tanh', max_iter=10000),
        AdaBoostClassifier(),
        DummyClassifier(strategy='stratified'),
        NN_EnsembleClassifier(3),
        ]

    performance = {name:[] for name in names}
    from sklearn.metrics import roc_auc_score

    for i in range(trial_num):
        print(i)
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            # score = clf.score(X_test, y_test)
            y_pred = clf.predict(X_test)
            prob = clf.predict_proba(X_test)[:, 1]


            t = y_test == 1
            p = y_pred == 1
            tp = t & p

            precision = np.sum(tp) / np.sum(p)
            recall = np.sum(tp) / np.sum(t)
            f1 = 2*precision*recall / (precision+recall)
            roc_auc = roc_auc_score(y_test, prob)

            print(f'{name}, roc_auc_score:{roc_auc:.3f}; f1: {f1:.3f}; precision: {precision:.3f}; recall: {recall:.3f}')
            performance[name].append(roc_auc)

            from customized_utils import draw_auc_roc_for_scores
            # draw_auc_roc_for_scores(-1*prob, y_test)


    for name in names:
        print(name, np.mean(performance[name]), np.std(performance[name]))



def load_data(subfolders):
    data_list = []
    is_bug_list = []

    objectives_list = []

    for sub_folder in subfolders:
        if os.path.isdir(sub_folder):
            pickle_filename = os.path.join(sub_folder, 'cur_info.pickle')
            with open(pickle_filename, 'rb') as f_in:
                cur_info = pickle.load(f_in)
                data, objectives, is_bug = reformat(cur_info)
                data_list.append(data)

                is_bug_list.append(is_bug)
                objectives_list.append(objectives)


    return np.array(data_list), np.array(is_bug_list), np.array(objectives_list)


def get_sorted_subfolders(parent_folder):
    bug_folder = os.path.join(parent_folder, 'bugs')
    non_bug_folder = os.path.join(parent_folder, 'non_bugs')
    sub_folders = [os.path.join(bug_folder, sub_name) for sub_name in os.listdir(bug_folder)] + [os.path.join(non_bug_folder, sub_name) for sub_name in os.listdir(non_bug_folder)]

    ind_sub_folder_list = []
    for sub_folder in sub_folders:
        if os.path.isdir(sub_folder):
            ind = int(re.search('.*bugs/([0-9]*)', sub_folder).group(1))
            ind_sub_folder_list.append((ind, sub_folder))

    ind_sub_folder_list_sorted = sorted(ind_sub_folder_list)
    subfolders = [filename for i, filename in ind_sub_folder_list_sorted]
    return subfolders





if __name__ == '__main__':
    mode = 'discrete'
    trial_num = 15
    cutoff = 300
    cutoff_end = 400
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2/town04_front_0/pedestrians_cross_street_town04/lbc/50_8_all'
    # 54.2+-1.5, 400: 218 VS 221, Ada 46
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2/town05_front_0/change_lane_town05_fixed_npc_num/lbc/50_8_all'
    # 51.5+-4, 400: 199 VS 203, 1000: 444 VS 421, Ada 43
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/50_8_all'
    # 55+-3, 400: 206 VS 178, Ada 59
    parent_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2/town04_front_0/pedestrians_cross_street_town04/lbc/50_8_all'

    subfolders = get_sorted_subfolders(parent_folder)
    X, is_bug_list, objective_list  = load_data(subfolders)

    if mode == 'discrete':
        classification_analysis(X, is_bug_list, objective_list, cutoff, cutoff_end, trial_num)
    else:
        regression_analysis(X, is_bug_list, objective_list, cutoff, cutoff_end, trial_num)
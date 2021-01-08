'''
threats to validity:
type of controllers
realism of carla
other type of violations
traffic sign / road shape
other search methods


# tomorrow TBD




should get job done after 1st LR
should use the new base for later iters LR

clip within projection?
debug / test
integrate
uniqueness concern


compare with ConAML in terms of the number of steps to get a better valid optimum point and final performance


adv attack method:
constraints: gradient descent, solve linear equations, VAE, extra head penalty backprop?
diversity: distance from found bugs, distance from start point, avg distance from existing bugs?


1.reread model inversion
2.come up with a generative model that considers bugginess, uniqueness, and constraints



visualization




problem has following characteristics:
expensive to run simulation
no access to derivatives
many local optimum (bugs)
a mixture of discrete variables and continuous variables
configs have constraints (and potentially conditional variables)
noise (simulations are not 100% reproducible)


want to find configs that have the following three properties with few queries:
bugginess:
ga (+nn, +adv nn)
conditional normalizing flow (w/ prior + bugginess as weight of weighted MLE) + adaptive sampling (natural evolutionary search, hamiltonian monte carlo, with objective reward - prob from conditional normalizing flow to trade-off exploration + exploitation)
RL (autoregressive Gaussian to model parameters + REINFORCE to optimize policy network with objective having reward for exploitation and entropy for exploration)
bayes optimization
vae
model inversion


uniqueness:
simple filtering
prob from a modeled probability distribution
self entropy (on the generated action)
curiosity
disagreement of ensemble
information gain


constraints:
simple filtering
extra branch violation penalty





currently:
0.trying pytorch town_05_right without indicator in loss
-- town05 collision
+ python ga_fuzzing.py -p 2021 2024 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2 --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0
nn
+ python ga_fuzzing.py -p 2015 2018 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2 --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --n_offsprings 300 --rank_mode nn --initial_fit_th 300 --dnn_lib pytorch

adv_nn
+ python ga_fuzzing.py -p 2021 2024 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2 --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --n_offsprings 300 --rank_mode adv_nn --initial_fit_th 300 --dnn_lib pytorch

random nn
+ python ga_fuzzing.py -p 2015 2018 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name random --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --n_offsprings 300 --rank_mode nn --initial_fit_th 300 --dnn_lib pytorch


random adv_nn
+ python ga_fuzzing.py -p 2021 2024 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name random --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --n_offsprings 300 --rank_mode adv_nn --initial_fit_th 300 --dnn_lib pytorch


nsga2-un
python ga_fuzzing.py -p 2021 2024 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0

nsga2-un nn
python ga_fuzzing.py -p 2015 2018 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --n_offsprings 300 --rank_mode nn --initial_fit_th 300 --dnn_lib pytorch



-- town05 out-of-road
+ python ga_fuzzing.py -p 2015 2018 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2 --has_run_num 700 --objective_weights 0 0 0 1 1 -1 0 0 0 0


+ python ga_fuzzing.py -p 2015 2018 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2 --has_run_num 700 --objective_weights 0 0 0 1 1 -1 0 0 0 0 --n_offsprings 300 --rank_mode nn --initial_fit_th 300 --dnn_lib pytorch


+ python ga_fuzzing.py -p 2021 2024 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2 --has_run_num 700 --objective_weights 0 0 0 1 1 -1 0 0 0 0 --n_offsprings 300 --rank_mode adv_nn --initial_fit_th 300 --dnn_lib pytorch





2.try nsga2 with no indicator in loss weights for town_03_front

-- town03 collision
+ python ga_fuzzing.py -p 2021 2024 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2 --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --n_offsprings 300


pytorch nn
+ python ga_fuzzing.py -p 2015 2018 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2 --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --n_offsprings 300 --rank_mode nn --initial_fit_th 300 --dnn_lib pytorch


adv_nn
+ python ga_fuzzing.py -p 2015 2018 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2 --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --n_offsprings 300 --rank_mode adv_nn --initial_fit_th 300 --dnn_lib pytorch



tomorrow TBD:


random un baseline


check out their carla visualization repo, consider to limit the parameters to similar to their papers 3/4 such that can be plotted on a plane for visualization and potentially demonstration of our methods


try model inversion + entropy

maybe also some more active search methods



multi-objective VS single-objective

(multi-head) regression adv attack?

uniqueness cretirion







project back into the constraints after each backprop?





The goal of their usage of DNN is different:
1. they keep an archive A of best solutions (since their goal is for best pareto front). they ranke their A and newly generated generation P to reduce the number of simulations on P that has very small chance of being better than A. Instead, since our goal is to generate as many buggy configs as possible, we rank the newly generated generation P and choose the top k ones to run simulations.

2.we additionally use adversarial attack which was rarely used in the setting of mixed categorical and continuous type of data. We project the gradient to categorical fields by taking the ind of the maximum field corresponding category. We also only normalize the continuous fields.

* 3.further, we introduce an extra ensemble DNN fitting and adversarial attack to since we observe that the fitted DNNs tend to be not very stable.





regularization for DNN?


give a nsga2 multi-objective baseline VS current single-objective to show why single-objective makes more sense here.

reproduce DNN-SM baseline
modify DNN-DT as another baseline




2.1 cross-validation ensemble? ensemble using DNNs trained with different subsets?
2.2 study ensemble adv

2.3 use similarity between validation samples and next generation samples (using some embedding distance metric?) to determine the weights of each sub-network in the ensemble?

2.5 study nsga2-un thresholds






maybe a cross-validation run to decide automatically whether to use nn and adv nn or not?


3.try to improve regression_analysis performance of DNN on town_03_front


learn a representation for configs using the objectives signal such that the representation can be further used for interpretation / adv / exploration (a new uniqueness definition is needed)?




learn embed space with supervised (maybe even temporal) signals
+
clustering analysis on embedded space
+
instance-level interpretation via running simulation on perturbed instance (leveraging existing neighbor's running results) (LIME/SHAP) to get hypothesis on most influencing feature(s)
+
causal inference via intervention on most influencing feature(s) during simulation




for other instances we can interpolate explanation of cluster center (multiplied with their distance to each class center) and finetune with small number of extra perturbations




maybe also consider eps for one-hot embed to be 1 or let eps for embed dims and non-embed dims separate (otherwise they cannot be changed), check out effect








consider about (1) sorting and then adv (2) adv and then sorting


t-sne and results change across adv iterations

new scenario that has both types of errors

analyze found bugs distribution

fix some actors are not moving on some parts of some maps

make mutation process more customizable (trade off convergence of exploration; current binary tournament only make use of top elements 2 times which leads to very slow convergence)



1.2 make adv can be tuned to make one type of bug more likely (this needs to make the classification separate classes or 2 DNNs; also may be also consider regression???)


1.5 maybe modify standardization and adv attack ? (i.e. also standardize those one-hot encoded fields but record each field's value to be used during projection?)

2.2 analyze new nsga2 correlation between objective and if bug

2.5 improve nsga2 + DNN over nsga2



2.8 try adv nn for town05_front



3.improve adv in rerun to try to make it work (early stop to avoid overfitting (increasing test loss) -> make this automatic by checking validation loss elbow point) Also early stop adv attack at some point

3.3 improve either search strategy or definition. the current way is extremelly close to sampling randomly for some scenarios due to probability theorem.

3.5 feature importance + causal analysis through intervention for important fields? LIME/SHAP for individual bug config analysis and thus define uniqueness? cluster analysis and calculate+sort std of each field of each cluster to determine uniqueness? neighbor bugness?

4.try some heuristic from MTFuzz for adv to potentially produce better performance for this particular problem?


clustering for error type analysis? (what vectors used for clustering? feature vector of some DNNs?)

maybe consider unique bugs in terms of if the actor is inside the view of the ego car?

think about better way to process the "if condition" for waypoint_follower


5 try regression

5.8 change sklearn in ga_fuzzing to pytorch and integrate adv (applying adv attack on top ranked new configs) (if it shows some early promising results).

6.try different config for town_05_front to get more out-of-road error? Or maybe improve the objective?

6.3 let is_collision to be true only when a real collision bug happens
6.5 dynamic weight adjustment (normalize each objective from previous generation objectives)


7.ensemble idea (use validation for weighted average confidence?)?

modify DNN objective for different types of bugs (i.e. different weights for different bugs assigned)?
compare with random + DNN?

maybe also tsne of learned DNN feature vector?

1.carefully design the fields chosen (perturbation, categorical)
2.analyze important features using MtFuzz hotbytes method

3.rethink about search objectives as well as optimization loss

4.redesign/select scenarios to run for potentially better property

5.consider MtFuzz method of leveraging hotbytes for generating new inputs


Methods:
0.better objective function for NN?
1.better NN for better than random performance? figure out training loss decreases but testing loss increases; NN to rank with fine-grained signal?? (right now we only use binary and have not considered different types of bugs)
2.RankNet / learn to rank to rank???
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

from customized_utils import encode_fields, decode_fields, remove_fields_not_changing, recover_fields_not_changing, get_labels_to_encode, customized_standardize, customized_fit


def reformat(cur_info):
    objectives = cur_info['objectives']
    is_bug = cur_info['is_bug']

    ego_linear_speed, min_d, d_angle_norm, offroad_d, wronglane_d, dev_dist, is_collision, is_offroad, is_wrong_lane, is_run_red_light = objectives
    accident_x, accident_y = cur_info['loc']


    # route_completion = cur_info['route_completion']

    # result_info = [ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, accident_x, accident_y, is_bug, route_completion]


    data, x, xl, xu, mask, labels = cur_info['data'], cur_info['x'][:-1], cur_info['xl'], cur_info['xu'], cur_info['mask'], cur_info['labels']

    assert len(x) == len(xl)


    return x, objectives, int(is_bug), mask, labels



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


def classification_analysis(X, is_bug_list, objective_list, cutoff, cutoff_en, trial_num, encode_fields):

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


    # X_train = standardize.fit_transform(X_train)
    # X_test = standardize.transform(X_test)


    one_hot_fields_len = len(encode_fields)

    customized_fit(X_train, standardize, one_hot_fields_len, partial=True)
    X_train = customized_standardize(X_train, standardize, one_hot_fields_len, partial=True)
    X_test = customized_standardize(X_test, standardize, one_hot_fields_len, partial=True)


    print('y_test', y_test)

    ind_0 = y_test==0
    ind_1 = y_test==1

    print(np.sum(y_train<0.5), np.sum(y_train>0.5), np.sum(y_test<0.5), np.sum(y_test>0.5))

    names = ["Nearest Neighbors", "Neural Net", "AdaBoost", "Random", "NN ensemble"]

    classifiers = [
        KNeighborsClassifier(5),
        MLPClassifier(hidden_layer_sizes=[150], solver='lbfgs', activation='tanh', max_iter=10000),
        AdaBoostClassifier(),
        DummyClassifier(strategy='stratified'),
        NN_EnsembleClassifier(5),
        ]

    performance = {name:[] for name in names}
    from sklearn.metrics import roc_auc_score
    # ['sklearn', 'pytorch']
    dnn_lib = 'pytorch'

    for i in range(trial_num):
        print(i)
        for name, clf in zip(names, classifiers):
            if name == "Neural Net" and dnn_lib == 'pytorch':
                from pgd_attack import train_net
                clf = train_net(X_train, y_train, [], [], model_type='one_output')
                y_pred = clf.predict(X_test)
                prob = clf.predict_proba(X_test)[:, 1]
            else:
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
    mask, labels = None, None
    for sub_folder in subfolders:
        if os.path.isdir(sub_folder):
            pickle_filename = os.path.join(sub_folder, 'cur_info.pickle')
            with open(pickle_filename, 'rb') as f_in:
                cur_info = pickle.load(f_in)
                data, objectives, is_bug, mask, labels = reformat(cur_info)
                data_list.append(data)

                is_bug_list.append(is_bug)
                objectives_list.append(objectives)


    return data_list, np.array(is_bug_list), np.array(objectives_list), mask, labels

def encode_and_remove_x(data_list, mask, labels):
    # town_05_right
    labels_to_encode = get_labels_to_encode(labels)

    x, enc, inds_to_encode, inds_non_encode, encoded_fields = encode_fields(data_list, labels, labels_to_encode)

    one_hot_fields_len = len(encoded_fields)
    x, x_removed, kept_fields, removed_fields = remove_fields_not_changing(x, one_hot_fields_len)

    return x, encoded_fields


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



def analyze_objective_data(X, is_bug_list, objective_list):
    from matplotlib import pyplot as plt


    mode = 'tsne_input'

    if mode == 'hist':
        ind = -2
        ind2 = 1
        print(np.sum(objective_list[:, ind]))



        cond1 = (is_bug_list==1) & (objective_list[:, ind]==1)
        cond2 = (is_bug_list==0) & (objective_list[:, ind]==0)


        print(np.where(cond1 == 1))
        print(objective_list[cond1, ind2])
        print(objective_list[cond2, ind2])

        plt.hist(objective_list[cond1, ind2], label='bug', alpha=0.5, bins=50)
        plt.hist(objective_list[cond2, ind2], label='normal', alpha=0.5, bins=100)
        plt.legend()
        plt.show()
    elif mode == 'tsne_input':
        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=2, perplexity=5, n_iter=3000).fit_transform(X)
        y = np.array(is_bug_list)
        ind0 = y == 0
        ind1 = y == 1
        plt.scatter(X_embedded[ind0, 0], X_embedded[ind0, 1], label='normal', alpha=0.5, s=3)
        plt.scatter(X_embedded[ind1, 0], X_embedded[ind1, 1], label='bug', alpha=0.5, s=5)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    mode = 'discrete'
    trial_num = 15
    cutoff = 300
    cutoff_end = 350

    parent_folder = 'run_results/nsga2/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/2021_01_04_21_08_35,none_pytorch_300'
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2/town03_front_1/change_lane_town03_fixed_npc_num/lbc/50_14_collision_nn_pytorch_300' 51
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2/town03_front_1/change_lane_town03_fixed_npc_num/lbc/50_12_out_of_road' 82.6
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/50_14_collision_none_pytorch_300' 98.7
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/50_14_out_of_road_nn_pytorch_300' 74.5

    subfolders = get_sorted_subfolders(parent_folder)
    X, is_bug_list, objective_list, mask, labels  = load_data(subfolders)

    X, encoded_fields = encode_and_remove_x(X, mask, labels)


    if mode == 'analysis':
        analyze_objective_data(X, is_bug_list, objective_list)
    elif mode == 'discrete':
        classification_analysis(X, is_bug_list, objective_list, cutoff, cutoff_end, trial_num, encoded_fields)
    else:
        regression_analysis(X, is_bug_list, objective_list, cutoff, cutoff_end, trial_num)

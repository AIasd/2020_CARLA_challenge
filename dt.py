'''
sudo -E /home/zhongzzy9/anaconda3/envs/carla99/bin/python dt.py

'''

import os
from sklearn import tree
import graphviz
import numpy as np
import pickle
from ga_fuzzing import run_ga
from datetime import datetime
from customized_utils import make_hierarchical_dir


def filter_critical_regions(X, y):
    print('\n'*20)
    print('+'*100, 'filter_critical_regions', '+'*100)

    min_samples_split = np.max([int(0.1*X.shape[0]), 2])
    estimator = tree.DecisionTreeClassifier(min_samples_split=min_samples_split, min_impurity_decrease=0.01, random_state=0)
    print(X.shape, y.shape, X, y)
    estimator = estimator.fit(X, y)

    leave_ids = estimator.apply(X)
    print('leave_ids', leave_ids)

    unique_leave_ids = np.unique(leave_ids)
    unique_leaves_bug_num = np.zeros(unique_leave_ids.shape[0])
    unique_leaves_normal_num = np.zeros(unique_leave_ids.shape[0])

    for j, unique_leave_id in enumerate(unique_leave_ids):
        for i, leave_id in enumerate(leave_ids):
            if leave_id == unique_leave_id:
                if y[i] == 0:
                    unique_leaves_normal_num[j] += 1
                else:
                    unique_leaves_bug_num[j] += 1

    for i, unique_leave_i in enumerate(unique_leave_ids):
        print('unique_leaves', unique_leave_i, unique_leaves_bug_num[i],  unique_leaves_normal_num[i])

    critical_unique_leaves = unique_leave_ids[unique_leaves_bug_num >= unique_leaves_normal_num]

    print('critical_unique_leaves', critical_unique_leaves)


    inds = np.array([leave_id in critical_unique_leaves for leave_id in leave_ids])
    print('\n'*20)

    return estimator, inds, critical_unique_leaves





def main():
    town_name = 'Town03'
    scenario = 'Scenario12'
    direction = 'front'
    route = 0
    # ['default', 'leading_car_braking', 'vehicles_only']
    scenario_type = 'leading_car_braking'
    ego_car_model = 'lbc'
    # ['generations', 'max_time']
    termination_condition = 'generations'
    max_running_time = 3600*24

    # [5, 7]
    outer_iterations = 2
    # 5
    n_gen = 1
    # 100
    pop_size = 2

    X_filtered = None
    F_filtered = None
    X = None
    y = None
    F = None
    objectives = None
    elapsed_time = None
    bug_num = None
    labels = None
    has_run = []
    estimator = None
    critical_unique_leaves = None


    now = datetime.now()
    dt_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


    for i in range(outer_iterations):
        dt_time_str_i = dt_time_str
        dt = True
        if i == 0 or np.sum(y)==0:
            dt = False
        if i == 1:
            n_gen += 1
        X_new, y_new, F_new, objectives_new, elapsed_time_new, bug_num_new, labels, has_run_new = run_ga(True, dt, X_filtered, F_filtered, estimator, critical_unique_leaves, n_gen, pop_size, dt_time_str_i, i, town_name, scenario, direction, route, scenario_type, ego_car_model)

        if i == 0:
            X = X_new
            y = y_new
            F = F_new
            objectives = objectives_new
            elapsed_time = elapsed_time_new
            bug_num = bug_num_new

        else:
            X = np.concatenate([X, X_new])
            y = np.concatenate([y, y_new])
            F = np.concatenate([F, F_new])
            objectives = np.concatenate([objectives, objectives_new])
            elapsed_time = np.concatenate([elapsed_time, elapsed_time_new + elapsed_time[-1]])
            bug_num = np.concatenate([bug_num, bug_num_new])
        has_run.append(has_run_new)


        estimator, inds, critical_unique_leaves = filter_critical_regions(X, y)
        X_filtered = X[inds]
        F_filtered = F[inds]
        print(len(X_filtered), X.shape)


        if termination_condition == 'max_time' and elapsed_time[-1] > max_running_time:
            break


    # Save data
    dt_save_folder = 'dt_data'
    if not os.path.exists(dt_save_folder):
        os.mkdir(dt_save_folder)
    dt_save_file = '_'.join([town_name, scenario, direction, str(route), scenario_type, str(n_gen), str(pop_size), str(outer_iterations), dt_time_str])

    pth = os.path.join(dt_save_folder, dt_save_file)
    np.savez(pth, X=X, y=y, F=F, objectives=objectives, elapsed_time=elapsed_time, bug_num=bug_num, labels=labels, has_run=has_run)
    print('dt data saved', 'has run', np.sum(has_run))
    os.system('chmod -R 777 '+dt_save_folder)



    return X, y, F, objectives, elapsed_time, bug_num, labels, has_run

def visualization(estimator):
    tree.plot_tree(estimator)
    dot_data = tree.export_graphviz(estimator, out_file=None, filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("tree")

if __name__ == '__main__':
    X, y, F, objectives, elapsed_time, bug_num, labels, has_run = main()


    # d = np.load('dt.npz')
    # X = d['X']
    # y = d['y']
    # estimator = tree.DecisionTreeClassifier(min_samples_split=int(0.1*X.shape[0]), min_impurity_decrease=0.01)
    # estimator = estimator.fit(X, y)
    # visualization(estimator)

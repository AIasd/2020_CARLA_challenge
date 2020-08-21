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
from customized_utils import make_hierarchical_dir, filter_critical_regions








def main():
    end_when_no_critical_region = False
    # ['Town01_left_0', 'Town03_front_0', 'Town05_front_0', 'Town05_right_0']
    route_type = 'town05_right_0'
    # ['default', 'leading_car_braking', 'vehicles_only']
    scenario_type = 'leading_car_braking'
    ego_car_model = 'lbc'
    # ['generations', 'max_time']
    termination_condition = 'generations'
    max_running_time = 3600*24
    objective_weights = np.array([-1, 1, 1, 1, -1])

    # [5, 7]
    outer_iterations = 3
    # 5
    n_gen = 4
    # 100
    pop_size = 100

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
    hv = None
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
        X_new, y_new, F_new, objectives_new, elapsed_time_new, bug_num_new, labels, has_run_new, hv_new = run_ga(True, dt, X_filtered, F_filtered, estimator, critical_unique_leaves, n_gen, pop_size, dt_time_str_i, i, route_type, scenario_type, ego_car_model, objective_weights)

        if i == 0:
            X = X_new
            y = y_new
            F = F_new
            objectives = objectives_new
            elapsed_time = elapsed_time_new
            bug_num = bug_num_new
            hv = hv_new

        else:
            X = np.concatenate([X, X_new])
            y = np.concatenate([y, y_new])
            F = np.concatenate([F, F_new])
            objectives = np.concatenate([objectives, objectives_new])
            elapsed_time = np.concatenate([elapsed_time, elapsed_time_new + elapsed_time[-1]])
            bug_num = np.concatenate([bug_num, bug_num_new])
            hv = np.concatenate([hv, hv_new])

        has_run.append(has_run_new)


        estimator, inds, critical_unique_leaves = filter_critical_regions(X, y)
        X_filtered = X[inds]
        F_filtered = F[inds]
        print(len(X_filtered), X.shape)
        if len(X_filtered) == 0 and end_when_no_critical_region:
            break

        if termination_condition == 'max_time' and elapsed_time[-1] > max_running_time:
            break


    # Save data
    dt_save_folder = 'dt_data'
    if not os.path.exists(dt_save_folder):
        os.mkdir(dt_save_folder)



    dt_save_file = '_'.join([route_type, scenario_type, str(n_gen), str(pop_size), str(outer_iterations), dt_time_str])

    pth = os.path.join(dt_save_folder, dt_save_file)
    np.savez(pth, X=X, y=y, F=F, objectives=objectives, elapsed_time=elapsed_time, bug_num=bug_num, labels=labels, has_run=has_run, hv=hv)
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

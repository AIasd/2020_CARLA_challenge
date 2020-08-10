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
    # [5, 7]
    outer_iterations = 3
    # 5
    n_gen = 2
    # 100
    pop_size = 20

    X_filtered = None
    F_filtered = None
    X = None
    y = None
    F = None
    objectives = None
    time = None
    estimator = None
    critical_unique_leaves = None


    now = datetime.now()
    dt_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


    for i in range(outer_iterations):
        dt = True
        if i == 0 or np.sum(y)==0:
            dt = False
        X_new, y_new, F_new, objectives_new, time_new = run_ga(True, dt, X_filtered, F_filtered, estimator, critical_unique_leaves, n_gen, pop_size, dt_time_str, i)

        if i == 0:
            X = X_new
            y = y_new
            F = F_new
            objectives = objectives_new
            time = time_new
        else:
            X = np.concatenate([X, X_new])
            y = np.concatenate([y, y_new])
            F = np.concatenate([F, F_new])
            objectives = np.concatenate([objectives, objectives_new])
            time.extend(time_new)



        estimator, inds, critical_unique_leaves = filter_critical_regions(X, y)
        X_filtered = X[inds]
        F_filtered = F[inds]
        print(np.sum(y))

    return X, y, F, objectives, time

def visualization(estimator):
    tree.plot_tree(estimator)
    dot_data = tree.export_graphviz(estimator, out_file=None, filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("tree")

if __name__ == '__main__':
    X, y, F, objectives, time = main()


    save_path = []
    np.savez('dt', X=X, y=y, F=F, objectives=objectives, time=time)
    print('dt data saved')


    # d = np.load('dt.npz')
    # X = d['X']
    # y = d['y']
    # estimator = tree.DecisionTreeClassifier(min_samples_split=int(0.1*X.shape[0]), min_impurity_decrease=0.01)
    # estimator = estimator.fit(X, y)
    # visualization(estimator)

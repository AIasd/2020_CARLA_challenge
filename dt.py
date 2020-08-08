import os
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import numpy as np




def filter_critical_regions(X, y):
    estimator = tree.DecisionTreeClassifier(min_samples_split=int(0.1*X.shape[0]), min_impurity_decrease=0.01)
    estimator = estimator.fit(X, y)

    leave_isd = estimator.apply(X)
    print(leave_ids, type(leave_ids))

    unique_leaves = np.unique(leave_ids)
    unique_leaves_bug_num = np.zeros(unique_leaves.shape[0])
    unique_leaves_normal_num = np.zeros(unique_leaves.shape[0])

    for j, leave in enumerate(unique_leaves):
        for i, leave_id in enumerate(leave_ids):
            if leave_id == leave:
                if y[i] == 0:
                    unique_leaves_normal_num[j] += 1
                else:
                    unique_leaves_bug_num[j] += 1
    critical_unique_leaves = unique_leaves[unique_leaves_bug_num > unique_leaves_normal_num]


    inds = np.array([leave_id in critical_unique_leaves for leave_id in leave_ids])

    return estimator, inds

def is_critical_region(X, estimator, critical_unique_leaves):
    leave_id = estimator.apply(x)
    return leave_id in critical_unique_leaves


def main():
    # [5, 7]
    outer_iterations = 3
    # 5
    generations = 5

    X_filtered = None
    F_filtered = None
    X = None
    y = None
    F = None
    estimator = None
    critical_unique_leaves = None

    for i in range(outer_iterations):
        X_new, y_new, F_new = run_ga(True, X_filtered, F_filtered, estimator, critical_unique_leaves, generations)
        run_ga(dt=False, X=None, F=None, estimator=None, critical_unique_leaves=None)

        if i == 0:
            X = X_new
            y = y_new
            F = F_new
        else:
            X = np.concatenate([X, X_new])
            y = np.concatenate([y, y_new])
            F = np.concatenate([F, F_new])



        estimator, inds = filter_critical_regions(X, y)
        X_filtered = X[inds]
        F_filtered = F[inds]
        print(np.sum(y))

    return estimator, X, y, F

def visualization(estimator):
    tree.plot_tree(estimator)
    dot_data = tree.export_graphviz(estimator, out_file=None, filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("tree")

if __name__ == '__main__':
    estimator, X, y, F = main()
    visualization(estimator)

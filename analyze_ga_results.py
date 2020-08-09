import sys
import os
sys.path.append('pymoo')




import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE




def draw_hv(bug_res_path, save_folder):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)
    hv = res['hv']
    n_evals = res['n_evals'].tolist()

    # hv = [0] + hv
    # n_evals = [0] + n_evals


    # visualze the convergence curve
    plt.plot(n_evals, hv, '-o')
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.savefig(os.path.join(save_folder, 'hv_across_generations'))
    plt.close()



def draw_performance(bug_res_path, save_folder):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)

    time_bug_num_list = res['time_bug_num_list']

    t_list = []
    n_list = []
    for t, n in time_bug_num_list:
        t_list.append(t)
        n_list.append(n)
    print(t_list)
    print(n_list)
    plt.plot(t_list, n_list, '-o')
    plt.title("Time VS Number of Bugs")
    plt.xlabel("Time")
    plt.ylabel("Number of Bugs")
    plt.savefig(os.path.join(save_folder, 'bug_num_across_time'))
    plt.close()


def analyze_causes(folder, save_folder, total_num, pop_size):



    avg_f = [0 for _ in range(int(total_num // pop_size))]

    causes_list = []
    counter = 0
    for sub_folder_name in os.listdir(folder):
        sub_folder = os.path.join(folder, sub_folder_name)
        if os.path.isdir(sub_folder):
            for filename in os.listdir(sub_folder):
                if filename.endswith(".npz"):
                    filepath = os.path.join(sub_folder, filename)
                    bug = np.load(filepath, allow_pickle=True)['bug'][()]

                    ego_linear_speed = float(bug['ego_linear_speed'])
                    causes_list.append((sub_folder_name, ego_linear_speed, bug['offroad_dist'], bug['is_wrong_lane'], bug['is_run_red_light'], bug['status'], bug['loc'], bug['object_type']))

                    ind = int(int(sub_folder_name) // pop_size)
                    avg_f[ind] += (ego_linear_speed / pop_size)*-1

    causes_list = sorted(causes_list, key=lambda t: int(t[0]))
    for c in causes_list:
        print(c)
    print(avg_f)

    plt.plot(np.arange(len(avg_f)), avg_f)
    plt.title("average objective value across generations")
    plt.xlabel("Generations")
    plt.ylabel("average objective value")
    plt.savefig(os.path.join(save_folder, 'f_across_generations'))

    plt.close()

def show_gen_f(bug_res_path):
    with open(bug_res_path, 'rb') as f_in:
        res = pickle.load(f_in)

    val = res['val']
    plt.plot(np.arange(len(val)), val)
    plt.show()

def plot_each_bug_num_and_objective_num_over_generations(generation_data_path):
    data = []
    with open(generation_data_path, 'r') as f_in:
        for line in f_in:
            tokens = line.split(',')
            if len(tokens) == 2:
                pass
            else:
                tokens = [float(x.strip('\n')) for x in line.split(',')]
                num, time, bugs, collisions, offroad, wronglane, speed, offroad, wronglane, dev = tokens[:10]
                data.append(np.array([num, time, bugs, collisions, offroad, wronglane, speed, offroad, wronglane, dev]))
    data = np.stack(data)

    fig = plt.figure(figsize=(9, 9))


    plt.suptitle("values over time", fontsize=14)


    info = [(2, 2, 'Bug Numbers'), (4, 3, 'Collision Numbers'), (5, 4, 'Offroad Numbers'), (6, 5, 'Wronglane Numbers'), (7, 6, 'Collision Speed'), (8, 7, 'Offroad Directed Distance'), (9, 8, 'Wronglane Directed Distance'), (11, 9, 'Deviation')]

    for loc, ind, ylabel in info:
        ax = fig.add_subplot(4, 3, loc)
        plt.plot(data[:, 1], data[:, ind])
        plt.xlabel("Time(s)")
        plt.ylabel(ylabel)
    plt.savefig('bug_num_and_objective_num_over_generations')




# list bug types and their run numbers
def list_bug_categories_with_numbers(folder_path):
    l = []
    for sub_folder_name in os.listdir(folder_path):
        sub_folder = os.path.join(folder_path, sub_folder_name)
        if os.path.isdir(sub_folder):
            for filename in os.listdir(sub_folder):
                if filename.endswith(".npz"):
                    filepath = os.path.join(sub_folder, filename)
                    bug = np.load(filepath, allow_pickle=True)['bug'][()]
                    if bug['ego_linear_speed'] > 0:
                        cause_str = 'collision'
                    elif bug['is_offroad']:
                        cause_str = 'offroad'
                    elif bug['is_wrong_lane']:
                        cause_str = 'wronglane'
                    else:
                        cause_str = 'unknown'
                    l.append((sub_folder_name, cause_str))


    for n,s in sorted(l, key=lambda t: int(t[0])):
        print(n,s)



# list pickled data
def analyze_data(pickle_path):
    with open(pickle_path, 'rb') as f_out:
        d = pickle.load(f_out)
        X = d['X']
        y = d['y']
        F = d['F']
        objectives = d['objectives']
        for x in list(X):
            print(x)


# plot two tsne plots for bugs VS normal and data points across generations
def apply_tsne(pickle_path, n_gen, pop_size):
    with open(pickle_path, 'rb') as f_out:
        d = pickle.load(f_out)
        X = d['X']
        y = d['y']
        F = d['F']
        objectives = d['objectives']

    generations = []
    for i in range(n_gen):
        generations += [i for _ in range(pop_size)]


    X_embedded = TSNE(n_components=2).fit_transform(X)
    fig = plt.figure(figsize=(18, 9))


    plt.suptitle("tSNE of sampled/generated data points", fontsize=14)


    ax = fig.add_subplot(121)
    scatter_bug = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap=plt.cm.rainbow)
    plt.title("bugs VS normal")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(handles=scatter_bug.legend_elements()[0], labels=['normal', 'bugs'])

    ax = fig.add_subplot(122)
    scatter_gen = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=generations, cmap=plt.cm.rainbow)
    plt.title("different generations")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(handles=scatter_gen.legend_elements()[0], labels=[str(i) for i in range(n_gen)])

    plt.savefig('tsne')


if __name__ == '__main__':
    # nsga2
    # folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town03/Scenario12/right/01/2020_08_01_01_59_32'
    # bug_res_path = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town03/Scenario12/right/01/2020_08_01_01_59_32/res_0.pkl'
    # folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town01/Scenario12/left/00/2020_08_02_00_47_04'
    # bug_res_path = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town01/Scenario12/left/00/2020_08_02_00_47_04/res_0.pkl'
    # total_num = 1000
    # pop_size = 100

    # random
    # folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town03/Scenario12/right/01/2020_08_01_15_16_26'
    # bug_res_path = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town03/Scenario12/right/01/2020_08_01_15_16_26/res_0.pkl'
    # total_num = 180
    # pop_size = 30

    # save_folder = 'plots'
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    #
    # analyze_causes(folder, save_folder, total_num, pop_size)
    # draw_hv(bug_res_path, save_folder)
    # draw_performance(bug_res_path, save_folder)


    # path = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/True/nsga2/Town05/Scenario12/right/00/2020_08_08_21_36_51/res_0.pkl'
    # analyze_data(path)
    # apply_tsne(path, 5, 20)



    # list_bug_categories_with_numbers('/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/False/nsga2/Town03/Scenario12/front/00/2020_08_08_16_17_53')


    plot_each_bug_num_and_objective_num_over_generations('/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/True/nsga2/Town05/Scenario12/right/00/2020_08_08_21_36_51/mean_objectives_across_generations.txt')

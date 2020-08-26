import sys
import os
sys.path.append('pymoo')




import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
from dt import filter_critical_regions
from customized_utils import  get_distinct_data_points


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

def plot_each_bug_num_and_objective_num_over_generations(generation_data_paths):
    # X=X, y=y, F=F, objectives=objectives, time=time_list, bug_num=bug_num_list, labels=labels, hv=hv
    pop_size = 100
    data_list = []
    for generation_data_path in generation_data_paths:
        data = []
        with open(generation_data_path[1], 'r') as f_in:
            for line in f_in:
                tokens = line.split(',')
                if len(tokens) == 2:
                    pass
                else:
                    tokens = [float(x.strip('\n')) for x in line.split(',')]
                    num, has_run, time, bugs, collisions, offroad_num, wronglane_num, speed, min_d, offroad, wronglane, dev = tokens[:12]
                    out_of_road = offroad_num + wronglane_num
                    data.append(np.array([num/pop_size, has_run, time, bugs, collisions, offroad_num, wronglane_num, out_of_road, speed, min_d, offroad, wronglane, dev]))

        data = np.stack(data)
        data_list.append(data)

    labels = [generation_data_paths[i][0] for i in range(len(data_list))]
    data = np.concatenate([data_list[1], data_list[2]], axis=0)

    for i in range(len(data_list[1]), len(data_list[1])+len(data_list[2])):
        data[i] += data_list[1][-1]
    data_list.append(data)

    labels.append('collision+out-of-road')

    fig = plt.figure(figsize=(15, 9))


    plt.suptitle("values over time", fontsize=14)


    info = [(1, 3, 'Bug Numbers'), (6, 4, 'Collision Numbers'), (7, 5, 'Offroad Numbers'), (8, 6, 'Wronglane Numbers'), (9, 7, 'Out-of-road Numbers'), (11, 8, 'Collision Speed'), (12, 9, 'Min object distance'), (13, 10, 'Offroad Directed Distance'), (14, 11, 'Wronglane Directed Distance'), (15, 12, 'Max Deviation')]

    for loc, ind, ylabel in info:
        ax = fig.add_subplot(3, 5, loc)
        for i in [0, 3, 1, 2]:
            if loc < 11 or i < 3:
                label = labels[i]
                if loc >= 11:
                    y = []
                    for j in range(data_list[i].shape[0]):
                        y.append(np.mean(data_list[i][:j+1, ind]))
                else:
                    y = data_list[i][:, ind]
                ax.plot(data_list[i][:, 0], y, label=label)
        if loc == 1:
            ax.legend()
        plt.xlabel("Generations")
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
        print(np.sum(X[10,:]-X[11,:]))
        filter_critical_regions(X, y)
        # TBD: tree diversity



def unique_bug_num(all_X, all_y, mask, xl, xu, cutoff):
    if cutoff == 0:
        return 0, []
    X = all_X[:cutoff]
    y = all_y[:cutoff]

    bug_inds = np.where(y>0)
    bugs = X[bug_inds]


    p = 0
    c = 0.15
    th = int(len(mask)*0.5)

    # TBD: count different bugs separately
    filtered_bugs, unique_bug_inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)


    print(cutoff, len(filtered_bugs), len(bugs))
    return len(filtered_bugs), np.array(unique_bug_inds), bug_inds

# plot two tsne plots for bugs VS normal and data points across generations
def apply_tsne(path, n_gen, pop_size):
    d = np.load(path, allow_pickle=True)
    X = d['X']
    y = d['y']
    mask = d['mask']
    xl = d['xl']
    xu = d['xu']


    cutoff = n_gen * pop_size
    _, unique_bug_inds, bug_inds = unique_bug_num(X, y, mask, xl, xu, cutoff)

    y[bug_inds] = 1
    y[unique_bug_inds] = 2


    generations = []
    for i in range(n_gen):
        generations += [i for _ in range(pop_size)]


    X_embedded = TSNE(n_components=2).fit_transform(X)

    fig = plt.figure(figsize=(9, 9))
    plt.suptitle("tSNE of bugs and unique bugs", fontsize=14)
    ax = fig.add_subplot(111)
    scatter_bug = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap=plt.cm.rainbow)
    plt.title("bugs VS normal")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(handles=scatter_bug.legend_elements()[0], labels=['normal', 'bugs'])


    # fig = plt.figure(figsize=(18, 9))
    #
    # plt.suptitle("tSNE of sampled/generated data points", fontsize=14)
    #
    #
    # ax = fig.add_subplot(121)
    # scatter_bug = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap=plt.cm.rainbow)
    # plt.title("bugs VS normal")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')
    # plt.legend(handles=scatter_bug.legend_elements()[0], labels=['normal', 'bugs'])
    #
    # ax = fig.add_subplot(122)
    # scatter_gen = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=generations, cmap=plt.cm.rainbow)
    # plt.title("different generations")
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')
    # plt.legend(handles=scatter_gen.legend_elements()[0], labels=[str(i) for i in range(n_gen)])

    plt.savefig('tsne')


def compare_with_dt(file):
    d = np.load(file, allow_pickle=True)
    print(np.sum(d['y']), len(d['y']))
    # from dt import filter_critical_regions
    # filter_critical_regions(d['X'], d['y'])
    # for i in range(d['X'].shape[0]):
    #     for j in range(i+1, d['X'].shape[0]):
    #         if np.abs(np.sum(d['X'][i] - d['X'][j])) < 0.00001:
    #             print(i, j, 'repeat')

    # print(d['X'][0])
    # print(d['X'][1])
    # print(np.sum(d['X'][0] - d['X'][1]))



def check_unique_bug_num(folder, path1, path2):
    d = np.load(folder+'/'+path1, allow_pickle=True)
    xl = d['xl']
    xu = d['xu']
    mask = d['mask']

    d = np.load(folder+'/'+path2, allow_pickle=True)
    all_X = d['X']
    all_y = d['y']
    cutoffs = [100*i for i in range(0, 15)]


    def subroutine(cutoff):
        if cutoff == 0:
            return 0, []
        X = all_X[:cutoff]
        y = all_y[:cutoff]

        bugs = X[y>0]


        p = 0
        c = 0.15
        th = int(len(mask)*0.5)

        filtered_bugs, inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)
        print(cutoff, len(filtered_bugs), len(bugs))
        return len(filtered_bugs), inds


    num_of_unique_bugs = []
    for cutoff in cutoffs:
        num, inds = subroutine(cutoff)
        num_of_unique_bugs.append(num)
    print(inds)
    # print(bug_counters)
    # counter_inds = np.array(bug_counters)[inds] - 1
    # print(all_X[counter_inds[-2]])
    # print(all_X[counter_inds[-1]])

    plt.plot(cutoffs, num_of_unique_bugs)
    plt.xlabel('num of simulations')
    plt.ylabel('num of unique bugs')
    plt.savefig('num_of_unique_bugs')

def calculate_pairwise_dist(path_list):
    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)
        xl = d['xl']
        xu = d['xu']
        mask = d['mask']
        all_X = d['X']
        all_y = d['y']
        cutoffs = [100*i for i in range(0, 16)]

        int_inds = mask == 'int'
        real_inds = mask == 'real'
        eps = 1e-8

        p = 0
        c = 0.15
        th = int(len(mask)*0.5)


        def pair_dist(x_1, x_2):
            int_diff_raw = np.abs(x_1[int_inds] - x_2[int_inds])
            int_diff = np.ones(int_diff_raw.shape) * (int_diff_raw > eps)

            real_diff_raw = np.abs(x_1[real_inds] - x_2[real_inds]) / (np.abs(xu - xl) + eps)[real_inds]

            real_diff = np.ones(real_diff_raw.shape) * (real_diff_raw > c)

            diff = np.concatenate([int_diff, real_diff])

            diff_norm = np.linalg.norm(diff, p)
            print(diff_norm)



        dist_list = []
        for i in range(len(all_X)-1):
            if i % 200 == 0:
                print(i)
            for j in range(i+1, len(all_X)):
                dist_list.append(pair_dist(all_X[i], all_X[j]))
        print(np.mean(dist_list))


def draw_unique_bug_num_over_simulations(path_list):
    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    line_style = ['-', ':', '--', '-.']
    for i, (label, pth) in enumerate(path_list):
        d = np.load(pth, allow_pickle=True)
        xl = d['xl']
        xu = d['xu']
        mask = d['mask']
        all_X = d['X']
        all_y = d['y']
        cutoffs = [100*i for i in range(0, 16)]


        def subroutine(cutoff):
            if cutoff == 0:
                return 0, []
            X = all_X[:cutoff]
            y = all_y[:cutoff]

            bugs = X[y>0]


            p = 0
            c = 0.15
            th = int(len(mask)*0.5)

            filtered_bugs, inds = get_distinct_data_points(bugs, mask, xl, xu, p, c, th)
            print(cutoff, len(filtered_bugs), len(bugs))
            return len(filtered_bugs), inds


        num_of_unique_bugs = []
        for cutoff in cutoffs:
            num, inds = subroutine(cutoff)
            num_of_unique_bugs.append(num)
        print(inds)
        # print(bug_counters)
        # counter_inds = np.array(bug_counters)[inds] - 1
        # print(all_X[counter_inds[-2]])
        # print(all_X[counter_inds[-1]])

        axes.plot(cutoffs, num_of_unique_bugs, label=label, linewidth=5, linestyle=line_style[i])

    axes.legend(loc=2, prop={'size': 18}, fancybox=True, framealpha=0.2)
    axes.set_xlabel('num of simulations', fontsize=23)
    axes.set_ylabel('num of unique bugs', fontsize=23)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.tight_layout()
    fig.savefig('num_of_unique_bugs')

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


    # plot_each_bug_num_and_objective_num_over_generations('/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/True/nsga2/Town05/Scenario12/right/00/2020_08_08_21_36_51/mean_objectives_across_generations.txt')


    # compare_with_dt('dt_data/Town03_Scenario12_front_0_default_1st.npz')
    # compare_with_dt('non_dt_data/Town03_Scenario12_front_0_leading_car_braking_4_50_2020_08_12_20_50_00.npz')

    # generation_data_paths = [('all', 'data_for_analysis/2020_08_15_17_21_03_12_100_leading_car_all_objective/Town05_Scenario12_right_0_leading_car_braking_12_100_all_objectives_2020_08_16_00_53_08.npz'),
    # ('out-of-road', 'data_for_analysis/2020_08_16_18_18_52_6_100_leading_car_out_of_road_objective/Town05_Scenario12_right_0_leading_car_braking_6_100_out_of_road_objective_2020_08_16_22_29_33.npz'),
    # ('collision', 'data_for_analysis/2020_08_17_00_40_54_6_100_leading_car_collision_objective/Town05_Scenario12_right_0_leading_car_braking_6_100_collision_objective_2020_08_17_04_56_20.npz')]


    # generation_data_paths = [('all', 'data_for_analysis/2020_08_15_17_21_03_12_100_leading_car_all_objective/mean_objectives_across_generations.txt'),
    # ('out-of-road', 'data_for_analysis/2020_08_16_18_18_52_6_100_leading_car_out_of_road_objective/mean_objectives_across_generations.txt'),
    # ('collision', 'data_for_analysis/2020_08_17_00_40_54_6_100_leading_car_collision_objective/mean_objectives_across_generations.txt')]
    #
    # plot_each_bug_num_and_objective_num_over_generations(generation_data_paths)


    # check_unique_bug_num('data_for_analysis/2020_08_15_17_21_03_12_100_leading_car_all_objective', 'data_for_analysis/2020_08_15_17_21_03_12_100_leading_car_all_objective/Town05_Scenario12_right_0_leading_car_braking_12_100_all_objectives_2020_08_16_00_53_08.npz')



    # check_unique_bug_num('data_for_analysis/', 'new_town05_right_50_10/2020_08_22_02_59_38_nsga2/bugs/town05_right_0_leading_car_braking_town05_lbc_15_100.npz', 'new_town05_right_50_10/2020_08_22_02_59_38_nsga2/bugs/town05_right_0_leading_car_braking_town05_lbc_15_100.npz')
    #
    # check_unique_bug_num('data_for_analysis/', 'new_town05_right_50_10/2020_08_22_02_59_38_nsga2/bugs/town05_right_0_leading_car_braking_town05_lbc_15_100.npz', 'new_town05_right_50_10/2020_08_22_03_00_00_nsga2_dt/town05_right_0_leading_car_braking_town05_lbc_5_100_3_2020_08_22_03_00_00.npz')


    # check_unique_bug_num('data_for_analysis/', 'new_town07_front_50_10/2020_08_23_03_24_33_nsga2-un/bugs/nsga2-un_town07_front_0_low_traffic_lbc_15_100.npz', 'new_town07_front_50_10/2020_08_23_03_24_33_nsga2-un/bugs/nsga2-un_town07_front_0_low_traffic_lbc_15_100.npz')


    # check_unique_bug_num('data_for_analysis/', 'new_nsga2-un_town07_front_50_10/2020_08_23_03_24_29_random/bugs/random_town07_front_0_low_traffic_lbc_15_100.npz', 'new_nsga2-un_town07_front_50_10/2020_08_23_03_24_29_random/bugs/random_town07_front_0_low_traffic_lbc_15_100.npz')






    # town07
    calculate_pairwise_dist([('random', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_23_03_24_29_random_50_10/bugs/random_town07_front_0_low_traffic_lbc_15_100.npz'), ('NSGA2', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_23_11_58_10_nsga2_50_15/bugs/nsga2_town07_front_0_low_traffic_lbc_15_100.npz'), ('NSGA2-UN', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_23_11_58_24_nsga2-un_50_15/bugs/nsga2-un_town07_front_0_low_traffic_lbc_15_100.npz')])




    # town07
    # draw_unique_bug_num_over_simulations([('random', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_23_03_24_29_random_50_10/bugs/random_town07_front_0_low_traffic_lbc_15_100.npz'), ('NSGA2', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_23_11_58_10_nsga2_50_15/bugs/nsga2_town07_front_0_low_traffic_lbc_15_100.npz'), ('NSGA2-UN', 'data_for_analysis/new_nsga2-un_town07_front_50_15/2020_08_23_11_58_24_nsga2-un_50_15/bugs/nsga2-un_town07_front_0_low_traffic_lbc_15_100.npz')])

    # town 01
    # draw_unique_bug_num_over_simulations([('random', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_23_20_59_53_random/bugs/random_town01_left_0_default_lbc_15_100.npz'), ('NSGA2', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_24_11_37_12_nsga2/bugs/nsga2_town01_left_0_default_lbc_15_100.npz'), ('NSGA2-UN', 'data_for_analysis/new_nsga2-un_town01_left_50_15/2020_08_23_20_59_46_nsga2-un/bugs/nsga2-un_town01_left_0_default_lbc_15_100.npz')])

    # town 05 front
    # draw_unique_bug_num_over_simulations([('random', 'data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_24_01_35_09_random/bugs/random_town05_front_0_change_lane_town05_lbc_15_100.npz'), ('NSGA2', 'data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_24_11_37_14_nsga2/bugs/nsga2_town05_front_0_change_lane_town05_lbc_15_100.npz'), ('NSGA2-UN', 'data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_23_21_42_33_nsga2-un/bugs/nsga2-un_town05_front_0_change_lane_town05_lbc_15_100.npz')])


    # apply_tsne('data_for_analysis/new_nsga2-un_town05_front_50_15/2020_08_23_21_42_33_nsga2-un/bugs/nsga2-un_town05_front_0_change_lane_town05_lbc_15_100.npz', 15, 100)

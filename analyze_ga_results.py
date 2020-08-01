import sys
import os
sys.path.append('pymoo')





import numpy as np
import matplotlib.pyplot as plt
import pickle




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


if __name__ == '__main__':
    # nsga2
    # folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town03/Scenario12/right/01/2020_08_01_01_59_32'
    # bug_res_path = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town03/Scenario12/right/01/2020_08_01_01_59_32/res_0.pkl'
    # total_num = 1000
    # pop_size = 100

    # random
    folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town03/Scenario12/right/01/2020_08_01_15_16_26'
    bug_res_path = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/bugs/Town03/Scenario12/right/01/2020_08_01_15_16_26/res_0.pkl'
    total_num = 180
    pop_size = 30

    save_folder = 'plots'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    analyze_causes(folder, save_folder, total_num, pop_size)
    draw_hv(bug_res_path, save_folder)
    draw_performance(bug_res_path, save_folder)

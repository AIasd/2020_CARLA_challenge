import sys
import shutil
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



import numpy as np
import os

from customized_utils import check_bug

def reformet():
    steps_per_sec = 10
    parent_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized/one_ped_only'

    for filename in sorted(os.listdir(parent_folder)):
        cur_folder = os.path.join(parent_folder, filename)
        new_folder = os.path.join(parent_folder, 'data')
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        file_path = os.path.join(cur_folder, 'other_actor_info.txt')
        new_file_path = os.path.join(new_folder, filename+'_other_actor_info.txt')
        print(filename)
        with open(new_file_path, 'a') as f_out:
            with open(file_path, 'r') as f_in:

                # need to manually change when the number of other agents changes
                f_out.write('steer,throttle,brake,ego_speed,ego_x,ego_y,ego_yaw,ped_speed,ped_x,ped_y,ped_yaw,ped_vis\n')

                prev = []
                for i, line in enumerate(f_in):
                    tokens = line.strip().split(',')
                    f_out.write(','.join(tokens[:7]))
                    for k, j in enumerate(range(7, len(tokens), 4)):
                        if len(prev) == 4:
                            x, y = float(tokens[j]), float(tokens[j+1])
                            prev_x, prev_y = prev[k*2], prev[k*2+1]
                            # print(x, prev_x, y, prev_y)
                            v = np.sqrt((x-prev_x)**2+(y-prev_y)**2) * steps_per_sec
                            prev[k*2], prev[k*2+1] = x, y

                        else:
                            prev_x, prev_y = float(tokens[j]), float(tokens[j+1])
                            v = 0
                            prev.extend([prev_x, prev_y])
                        f_out.write(','+','.join([str(v)]+tokens[j:j+4]))
                    f_out.write('\n')
            data = np.load(os.path.join(cur_folder, 'results.npz'), allow_pickle=True)
            f_out.write(','.join([str(obj) for obj in list(data['objectives'])]))

def inspect_data():
    parent_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized/one_ped_only/data'

    new_bug_parent_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized/one_ped_only/data/collion_bugs'
    new_non_bug_parent_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized/one_ped_only/data/normal_cases'
    if not os.path.exists(new_bug_parent_folder):
        os.mkdir(new_bug_parent_folder)
    if not os.path.exists(new_non_bug_parent_folder):
        os.mkdir(new_non_bug_parent_folder)

    non_bug = 0
    bug = 0
    for filename in os.listdir(parent_folder):
        filepath = os.path.join(parent_folder, filename)
        if not os.path.isdir(filepath):
            with open(filepath, 'r') as f_in:
                objectives = f_in.read().split('\n')[-1].split(',')
                objectives = [float(obj) for obj in objectives]
                if objectives[-1] > 0 and objectives[0] > 0 and bug < 100:
                    new_filepath = os.path.join(new_bug_parent_folder, filename)
                    bug += 1
                    shutil.copyfile(filepath, new_filepath)
                if not check_bug(objectives) and non_bug < 100:
                    new_filepath = os.path.join(new_non_bug_parent_folder, filename)
                    non_bug += 1
                    shutil.copyfile(filepath, new_filepath)

if __name__ == '__main__':
    inspect_data()

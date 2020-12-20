'''
Convert carla_challenge data record folder format to the format of action based repo

set all_folder be equal to the target data folder
run this script: python reformat_data_folder_for_action_based.py
move customized.json to Action-Based-Representation-Learning/carl/database/Corl2020; note that `package name` field in customized.json needs to be set to customized

(only need once) move leaderboard/data/new_routes folder to carl/database/Corl2020/

move customized folder to Action-Based-Representation-Learning/

(may only need once if using customized as the name) add configs/EXP/customized and configs/ENCODER/customized

may want to empty _preloads
may want to empty _logs/ENCODER

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
sys.path.append('carla_project/src')

from shutil import copyfile
from customized_utils import make_hierarchical_dir

all_folder = 'collected_data_customized/customized_auto_pilot_nodebug'
new_all_folder = 'collected_data_customized/customized'
if not os.path.exists(new_all_folder):
    os.mkdir(new_all_folder)


def copyfile_wrapper(original_folder_path, new_folder_path, filename):
    path = os.path.join(original_folder_path, filename)
    new_path = os.path.join(new_folder_path, filename)
    copyfile(path, new_path)

for run_folder_name in os.listdir(all_folder):
    print(run_folder_name)
    run_folder_path = os.path.join(all_folder, run_folder_name)
    if os.path.isdir(run_folder_path):
        parent_new_run_folder_path = os.path.join(new_all_folder, run_folder_name)
        new_run_folder_path = make_hierarchical_dir([parent_new_run_folder_path, '0_NPC', '0'])

        # copy metadata json
        copyfile_wrapper(run_folder_path, parent_new_run_folder_path, 'metadata.json')
        # copy summary json
        copyfile_wrapper(run_folder_path, new_run_folder_path, 'summary.json')

        # copy extra informative files
        for fn in ['driving_log.csv', 'events.txt', 'measurements.csv', 'measurements_loc.csv']:
            copyfile_wrapper(run_folder_path, parent_new_run_folder_path, fn)


        folder_name_pair = [('rgb', 'rgb_central'), ('rgb_left', 'rgb_left'), ('rgb_right', 'rgb_right'), ('measurements', 'measurements_')]

        for folder_name, prefix_name in folder_name_pair:
            subfolder_path = os.path.join(run_folder_path, folder_name)

            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                new_file_path = os.path.join(new_run_folder_path, prefix_name+file_name)
                copyfile(file_path, new_file_path)

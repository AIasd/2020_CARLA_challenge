from ga_fuzzing import run_simulation
from object_types import pedestrian_types, vehicle_types, static_types, vehicle_colors
import random
from datetime import datetime
from customized_utils import make_hierarchical_dir

os.environ['HAS_DISPLAY'] = '0'
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def rerun_simulation(pickle_filename, rerun_save_folder, ind):
    is_bug = False

    # parameters preparation
    if i == 0:
        launch_server = True
    else:
        launch_server = False

    with open(pickle_filename, 'rb') as f_in:
        d = pickle.load(f_in)['info']
        x = d['x']
        waypoints_num_limit = d['waypoints_num_limit']
        max_num_of_static = d['max_num_of_static']
        max_num_of_pedestrians = d['max_num_of_pedestrians']
        max_num_of_vehicles = d['max_num_of_vehicles']

        episode_max_time = d['episode_max_time']
        call_from_dt = d['call_from_dt']
        town_name = d['town_name']
        scenario = d['scenario']
        direction = d['direction']
        route_str = d['route_str']

    ego_car = 'autopilot'

    customized_data = convert_x_to_customized_data(x, waypoints_num_limit, max_num_of_static, max_num_of_pedestrians, max_num_of_vehicles, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms)

    objectives, loc, object_type, info, save_path = run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, ego_car, rerun=True)


    if objectives[0] > 0 or objectives[4] or objectives[5]:
        is_bug = True


    cur_info = {'x':x, 'objectives':objectives, 'loc':loc, 'object_type':object_type}

    cur_folder = rerun_save_folder+'/'+str(ind)
    if not os.path.exists(cur_folder):
        os.mkdir(cur_folder)
    with open(cur_folder+'/'+'cur_info', 'wb') as f_out:
        pickle.dump(bug, f_out)

    # copy data to another place
    try:
        shutil.copytree(save_path, cur_folder)
    except:
        print('fail to copy from', save_path)

    return is_bug, objectives


if __name__ == '__main__':
    random.seed(0)
    datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    rerun_save_folder = make_hierarchical_dir(['rerun', time_str])

    folder = ''
    subfolder_names = [sub_folder_name for sub_folder_name in os.listdir(folder)]
    random.shuffle(subfolder_names)


    assert len(subfolder_names) >= 2
    mid = int(len(subfolder_names)//2)

    train_subfolder_names = subfolder_names[:mid]
    test_subfolder_names = subfolder_names[mid:]

    # ['train', 'test']
    mode = 'train'

    if mode == 'train':
        chosen_subfolder_names = train_subfolder_names
    elif mode == 'test':
        chosen_subfolder_names = test_subfolder_names

    bug_num = 0
    objectives_avg = None

    for ind, sub_folder_name in enumerate(chosen_subfolder_names):
        sub_folder = os.path.join(folder, sub_folder_name)
        if os.path.isdir(sub_folder):
            for filename in os.listdir(sub_folder):
                if filename.endswith(".pickle"):
                    pickle_filename = os.path.join(sub_folder, filename)
                    is_bug, objectives = rerun_simulation(pickle_filename, rerun_save_folder, ind)

                    if not objectives_avg:
                        objectives_avg = objectives
                    else:
                        objectives_avg += objectives

                    if is_bug:
                        bug_num += 1

    print('bug_ratio :', bug_num / len(chosen_subfolder_names))
    print('objectives_avg :', objectives_avg / len(chosen_subfolder_names))

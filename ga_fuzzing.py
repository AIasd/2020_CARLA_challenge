'''
TBD:
* random seed


* tsne, decision tree volume

* another scene






* emcmc
* algorithm selection interface (integrate dt into ga_fuzzing)
* random seed
* record high resolution image

* avoid generation of other objects very close to the ego-car (this can also be achieved by customized constraints)




* save state to continue

* all objective 12 gen VS collision 6 gen + wrong route 6 gen VS all objective dt 3 * 4 gen: check average objectives and error numbers/types distributions, data t-sne visualization across generations and bug VS non-bug and different types of bugs, decision tree volumes
* estimate single thread time


* more exact time for dt
* mating critical region for dt


* route completion bug

* explore generation blocking issue and might consider to pre-run a map and save legit regions
* fix rgb with car high resolution


* more routes


* multi-objective search VS 3 single-objective search and compare results


* need to bound the projection / add_dist process when keep trying to generate an actor to within the bounds
* allow user to specify a region that actors cannot be generated within such that we can avoid spawning of static on route (a variable to control)


* diversity of bugs



* fix x afterwards when original setup cannot generate actors properly


* analyze dt results(visualization, show leaves results)


* hd image saving



* save intermediate results to avoid crash


* analyze visualization across generations

* parametrize in start / end location of each route


* estimate bug diversity via tree diversity
* fix stage2 model training
* debug nsga2-dt

* emcmc
* clustering+tsne(need to label different bugs first), bug category over generation plot

* stage 1 map model


* maybe make the validity checking static before each simulation (this needs extra knowledge of the map)


* limit generation of objects in a certain area




* decision tree feature importance analysis and bug diversity analysis
* seed selection (add constraints to input space) to search for particular pre-crash scene bugs

* continuous objective of wronglane/offroad when violation happens

* record rgb_with_car for pid_controller and auto_pilot

* the birdview window currently does not appear even when os.environ['HAS_DISPLAY'] = '1'
* need to improve diversity of bugs
* maybe not allowed to generate static objects directly on routes
* better way for determining and eliminating duplicates
* remove some very small static objects from the options (collision with them should not be considered as bug)
* evolutionary MCMC


* check diversity of generated scenes


* mutation cannot be picklized
* Traceback (most recent call last):
  File "ga_fuzzing.py", line 1017, in <module>
    main()
  File "ga_fuzzing.py", line 1005, in main
    np.savez(problem.bug_folder+'/'+'res'+'_'+ind, res=res, algorithm_name=algorithm_name, time_bug_num_list=problem.time_bug_num_list)
  File "<__array_function__ internals>", line 6, in savez
  File "/home/zhongzzy9/anaconda3/envs/carla99/lib/python3.7/site-packages/numpy/lib/npyio.py", line 645, in savez
    _savez(file, args, kwds, False)
  File "/home/zhongzzy9/anaconda3/envs/carla99/lib/python3.7/site-packages/numpy/lib/npyio.py", line 754, in _savez
    pickle_kwargs=pickle_kwargs)
  File "/home/zhongzzy9/anaconda3/envs/carla99/lib/python3.7/site-packages/numpy/lib/format.py", line 676, in write_array
    pickle.dump(array, fp, protocol=3, **pickle_kwargs)
_pickle.PicklingError: Can't pickle <class '__main__.MySampling'>: it's not the same object as __main__.MySampling
-------------------------------



* need to consider static objects or avoid generating static objects on the way for autopilot and pid controller
* modify API of PID controller to make it run


* fix OSError: [Errno 24] Too many open files: '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized/Town03/Scenario12/right/route_01_16/events.txt'
RuntimeError: Resource temporarily unavailable
* change to a more reliable controller
* scenario that can run successfully when no other objects are added
* focus on important parameter perturbation (e.g. waypoints perturbation are not very important but take too much dimensions.) If we reduce dimensions to e.g. less than 20, we might consider to apply out-of-box bayes optimization method on github.



* understand n_gen |  n_eval |  n_nds  | delta_ideal  | delta_nadir  |   delta_f in the stdout
* reproduce bug scenario

* offsprings seem to be similar. need to be fixed.
* save and resume_run a training after each generation

* narrow down the range of other actors and limit the time length of each run
* free-view window of the map




check number of opened files:
lsof -p p_id | wc -l
check max number of opened files:
ulimit -n

https://superuser.com/questions/1200539/cannot-increase-open-file-limit-past-4096-ubuntu
su zhongzzy9



Run genertic algorithm for fuzzing:
sudo -E /home/zhongzzy9/anaconda3/envs/carla99/bin/python ga_fuzzing.py

Retrain model from scratch (stage 1):
CUDA_VISIBLE_DEVICES=0 python carla_project/src/map_model.py --dataset_dir '/home/zhongzzy9/Documents/self-driving-car/LBC_data/CARLA_challenge_autopilot' --max_epochs 25

Retrain model from scratch (stage 2):
CUDA_VISIBLE_DEVICES=0 python carla_project/src/image_model.py --dataset_dir '../LBC_data/CARLA_challenge_autopilot' --teacher_path '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/models/stage1_retrain_9_50_leading_car_25_1hz_epoch=18.ckpt' --max_epochs 25

'''

# hack: increase the maximum number of files to open to avoid too many files open error due to leakage.
# import resource
#
# print("getrlimit before:", resource.getrlimit(resource.RLIMIT_NOFILE))
# resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))
# print("getrlimit:", resource.getrlimit(resource.RLIMIT_NOFILE))


import sys
import os
sys.path.append('pymoo')
carla_root = '/home/zhongzzy9/Documents/self-driving-car/carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')
sys.path.append('.')
sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')
sys.path.append('scenario_runner')
sys.path.append('/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/carla_project/src')




from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.duplicate import ElementwiseDuplicateElimination, NoDuplicateElimination

from pymoo.model.population import Population
from pymoo.model.evaluator import Evaluator

from pymoo.algorithms.nsga2 import NSGA2, binary_tournament
from pymoo.algorithms.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.algorithms.random import RandomAlgorithm

from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover

from pymoo.performance_indicator.hv import Hypervolume

import matplotlib.pyplot as plt

from object_types import WEATHERS, pedestrian_types, vehicle_types, static_types, vehicle_colors, car_types, motorcycle_types, cyclist_types

from customized_utils import create_transform, rand_real, specify_args, convert_x_to_customized_data, make_hierarchical_dir, exit_handler, arguments_info, is_critical_region, setup_bounds_mask_labels_distributions_stage1, setup_bounds_mask_labels_distributions_stage2, customize_parameters, customized_bounds_and_distributions, static_general_labels, pedestrian_general_labels, vehicle_general_labels, waypoint_labels, waypoints_num_limit, if_volate_constraints, customized_routes, parse_route_and_scenario, get_distinct_data_points, is_similar


from collections import deque


import numpy as np
import carla

from leaderboard.fuzzing import LeaderboardEvaluator
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.customized.object_params import Static, Pedestrian, Vehicle

import traceback
import json
import re
import time
from datetime import datetime

import pathlib
import shutil
import dill as pickle
# import pickle
import argparse
import atexit
import traceback




import copy

from pymoo.factory import get_termination
from pymoo.model.termination import Termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from pymoo.model.individual import Individual
from pymoo.model.repair import Repair
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_crossover, get_mutation
from pymoo.model.mating import Mating

from dask.distributed import Client, LocalCluster













use_unique_bugs = True
random_seeds = [10, 20, 30]
rng = np.random.default_rng(random_seeds[0])
bug_root_folder = 'bugs'
non_bug_root_folder = 'non_bugs'
# ['town01_left_0', 'town03_front_0', 'town05_front_0', 'town05_right_0']
global_route_type = 'town01_left_0'
# ['default', 'leading_car_braking', 'vehicles_only', 'no_static']
global_scenario_type = 'default'

scenario_file = 'current_scenario.json'

# ['nsga2', 'random']
algorithm_name = 'nsga2'
# ['lbc', 'auto_pilot', 'pid_agent']
global_ego_car_model = 'lbc'
os.environ['HAS_DISPLAY'] = '0'
# This is used to control how this program use GPU
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


parser = argparse.ArgumentParser()
parser.add_argument('--resume-run', help='continue to run', default=False, action='store_true')
parser.add_argument('--ind', help='ind ', default=0)
arguments = parser.parse_args()

# ind = arguments.ind
run_parallelization = True
resume_run = False
save = True
save_path = 'ga_intermediate.pkl'

episode_max_time = 50
global_n_gen = 12
global_pop_size = 100
max_running_time = 3600*24
# [ego_linear_speed, closest_dist, offroad_d, wronglane_d, dev_dist]
# objective_weights = np.array([-1, 1, 1, 1, -1])
global_objective_weights = np.array([-1, 1, 1, 1, -1])
# ['generations', 'max_time']
global_termination_condition = 'generations'

global_scheduler_port = 8788
global_dashboard_address = 8789
global_ports = [2000]
if run_parallelization:
    global_scheduler_port = 8785
    global_dashboard_address = 8786
    global_ports = [2003, 2006]
    # global_ports = [2003]

















'''
for customizing weather choices, static_types, pedestrian_types, vehicle_types, and vehicle_colors, make changes to object_types.py
'''



class MyProblem(Problem):

    def __init__(self, elementwise_evaluation, bug_parent_folder, non_bug_parent_folder, town_name, scenario, direction, route_str, scenario_file, ego_car_model, run_parallelization, scheduler_port, dashboard_address, customized_config, ports=[2000], episode_max_time=10000, customized_parameters_distributions={}, customized_center_transforms={}, call_from_dt=False, dt=False, estimator=None, critical_unique_leaves=None, dt_time_str='', dt_iter=0, objective_weights=np.array([0, 0, 1, 1, -1])):

        customized_parameters_bounds = customized_config['customized_parameters_bounds']
        customized_parameters_distributions = customized_config['customized_parameters_distributions']
        customized_center_transforms = customized_config['customized_center_transforms']
        customized_constraints = customized_config['customized_constraints']


        self.objective_weights = objective_weights
        self.customized_constraints = customized_constraints

        self.call_from_dt = call_from_dt
        self.dt = dt
        self.estimator = estimator
        self.critical_unique_leaves = critical_unique_leaves
        self.has_run = 0

        self.objectives_list = []
        self.x_list = []
        self.y_list = []
        self.F_list = []

        self.run_parallelization = run_parallelization
        self.scheduler_port = scheduler_port
        self.dashboard_address = dashboard_address
        self.ports = ports
        self.episode_max_time = episode_max_time


        if self.call_from_dt:
            time_str = dt_time_str
        else:
            now = datetime.now()
            time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.bug_folder = bug_parent_folder + time_str
        self.non_bug_folder = non_bug_parent_folder + time_str
        if not os.path.exists(self.bug_folder):
            os.mkdir(self.bug_folder)


        self.town_name = town_name
        self.scenario = scenario
        self.direction = direction
        self.route_str = route_str
        self.scenario_file = scenario_file
        self.ego_car_model = ego_car_model

        if self.call_from_dt:
            self.bug_folder += '/' + str(dt_iter)
            if not os.path.exists(self.bug_folder):
                os.mkdir(self.bug_folder)


        self.counter = 0
        self.num_of_bugs = 0
        self.num_of_unique_bugs = 0
        self.num_of_collisions = 0
        self.num_of_offroad = 0
        self.num_of_wronglane = 0

        if resume_run and len(self.time_list) > 0:
            self.start_time = time.time()
        else:
            self.start_time = time.time()
        self.time_elapsed = 0
        self.time_list = []
        self.bug_num_list = []
        self.unique_bug_num_list = []
        self.bugs = []
        self.unique_bugs = []
        self.bugs_inds_list = []
        self.unique_bugs_inds_list = []
        # TBD: add to interface




        fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels = setup_bounds_mask_labels_distributions_stage1()
        customize_parameters(parameters_min_bounds, customized_parameters_bounds)
        customize_parameters(parameters_max_bounds, customized_parameters_bounds)


        fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels, parameters_distributions, n_var = setup_bounds_mask_labels_distributions_stage2(fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds, mask, labels)
        customize_parameters(parameters_min_bounds, customized_parameters_bounds)
        customize_parameters(parameters_max_bounds, customized_parameters_bounds)
        customize_parameters(parameters_distributions, customized_parameters_distributions)




        for d in [fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds]:
            for k, v in d.items():
                assert not hasattr(self, k), k+'should not appear twice.'
                setattr(self, k, v)


        xl = [pair[1] for pair in parameters_min_bounds.items()]
        xu = [pair[1] for pair in parameters_max_bounds.items()]


        self.parameters_min_bounds = parameters_min_bounds
        self.parameters_max_bounds = parameters_max_bounds
        self.mask = mask
        self.labels = labels
        self.parameters_distributions = parameters_distributions
        self.customized_center_transforms = customized_center_transforms


        self.p = 0
        self.c = 1
        self.th = int(len(self.labels) // 2)

        super().__init__(n_var=n_var, n_obj=4, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=elementwise_evaluation)






    def _evaluate(self, X, out, *args, **kwargs):
        objective_weights = self.objective_weights
        customized_center_transforms = self.customized_center_transforms

        waypoints_num_limit = self.waypoints_num_limit
        num_of_static_max = self.num_of_static_max
        num_of_pedestrians_max = self.num_of_pedestrians_max
        num_of_vehicles_max = self.num_of_vehicles_max

        episode_max_time = self.episode_max_time
        call_from_dt = self.call_from_dt
        bug_folder = self.bug_folder
        non_bug_folder = self.non_bug_folder

        parameters_min_bounds = self.parameters_min_bounds
        parameters_max_bounds = self.parameters_max_bounds
        labels = self.labels
        customized_constraints = self.customized_constraints

        dt = self.dt
        estimator = self.estimator
        critical_unique_leaves = self.critical_unique_leaves

        mean_objectives_across_generations_path = os.path.join(self.bug_folder, 'mean_objectives_across_generations.txt')


        town_name = self.town_name
        scenario = self.scenario
        direction = self.direction
        route_str = self.route_str
        scenario_file = self.scenario_file
        ego_car_model = self.ego_car_model

        all_final_generated_transforms_list = []




        def fun(x, launch_server, counter):
            if (dt and not is_critical_region(x[:-1], estimator, critical_unique_leaves)) or if_volate_constraints(x, customized_constraints, labels):
                objectives = [0, 7, 7, 7, 0, 0, 0, 0]
                F = np.array(objectives[:objective_weights.shape[0]]) * objective_weights
                return F, None, None, None, objectives, 0, None

            else:


                # x = denormalize_by_entry(self, x)

                customized_data = convert_x_to_customized_data(x, waypoints_num_limit, num_of_static_max, num_of_pedestrians_max, num_of_vehicles_max, static_types, pedestrian_types, vehicle_types, vehicle_colors, customized_center_transforms, parameters_min_bounds, parameters_max_bounds)


                # run simulation
                objectives, loc, object_type, info, save_path = run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, scenario_file, ego_car_model)

                print(counter, objectives)

                # [ego_linear_speed, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light]
                F = np.array(objectives[:objective_weights.shape[0]]) * objective_weights


                info = {**info, 'x':x, 'waypoints_num_limit':waypoints_num_limit, 'num_of_static_max':num_of_static_max, 'num_of_pedestrians_max':num_of_pedestrians_max, 'num_of_vehicles_max':num_of_vehicles_max, 'customized_center_transforms':customized_center_transforms,
                'parameters_min_bounds':parameters_min_bounds,
                'parameters_max_bounds':parameters_max_bounds}

                cur_info = {'counter':counter, 'x':x, 'objectives':objectives,  'loc':loc, 'object_type':object_type, 'labels':labels, 'info': info}

                is_bug = objectives[0] > 0 or objectives[5] or objectives[6]

                if is_bug:
                    cur_folder = make_hierarchical_dir([bug_folder, str(counter)])
                else:
                    cur_folder = make_hierarchical_dir([non_bug_folder, str(counter)])

                with open(cur_folder+'/'+'cur_info.pickle', 'wb') as f_out:
                    pickle.dump(cur_info, f_out)

                if is_bug:
                    # copy data to current folder if it is a bug
                    try:
                        new_path = os.path.join(cur_folder, 'data')
                        shutil.copytree(save_path, new_path)
                    except:
                        print('fail to copy from', save_path)

                # hack:
                cur_port = int(x[-1])
                filename = 'tmp_folder/'+str(cur_port)+'.pickle'
                with open(filename, 'rb') as f_in:
                    all_final_generated_transforms = pickle.load(f_in)

                return F, loc, object_type, info, objectives, 1, all_final_generated_transforms




        def submit_and_run_jobs(ind_start, ind_end, launch_server, job_results):
            time_elapsed = 0
            jobs = []
            for i in range(ind_start, ind_end):
                j = i % len(self.ports)
                port = self.ports[j]
                worker = workers[j]
                x = np.concatenate([X[i], np.array([port])])
                jobs.append(client.submit(fun, x, launch_server, self.counter, workers=worker))

                self.counter += 1


            for i in range(len(jobs)):
                job = jobs[i]
                F, loc, object_type, info, objectives, has_run, all_final_generated_transforms_i = job.result()
                all_final_generated_transforms_list.append(all_final_generated_transforms_i)


                self.has_run += has_run
                # record bug
                if objectives[0] > 0 or objectives[5] or objectives[6]:
                    bug_str = None
                    if objectives[0] > 0:
                        self.num_of_collisions += 1
                        collision_types = {'pedestrian_collision':pedestrian_types, 'car_collision':car_types, 'motercycle_collision':motorcycle_types, 'cyclist_collision':cyclist_types, 'static_collision':static_types}
                        for k,v in collision_types.items():
                            if object_type in v:
                                bug_str = k
                        if not bug_str:
                            bug_str = 'unknown_collision'+'_'+object_type
                    elif objectives[5]:
                        self.num_of_offroad += 1
                        bug_str = 'offroad'
                    elif objectives[6]:
                        self.num_of_wronglane += 1
                        bug_str = 'wronglane'
                    else:
                        bug_str = 'unknown'
                    with open(mean_objectives_across_generations_path, 'a') as f_out:
                        f_out.write(str(i)+','+bug_str+'\n')

                    self.bugs.append(X[i].astype(float))
                    self.bugs_inds_list.append(self.counter-len(jobs)+i)

                    self.y_list.append(1)
                else:
                    self.y_list.append(0)
                # we don't want to store port number
                self.x_list.append(X[i])
                self.F_list.append(F)
                self.objectives_list.append(np.array(objectives))
                job_results.append(F)

            # print(all_final_generated_transforms_list)

            # hack:
            with open('tmp_folder/total.pickle', 'wb') as f_out:
                pickle.dump(all_final_generated_transforms_list, f_out)


            self.unique_bugs, distinct_inds = get_distinct_data_points(self.bugs, self.mask, self.xl, self.xu, self.p, self.c, self.th)
            self.num_of_bugs = len(self.bugs)
            self.num_of_unique_bugs = len(self.unique_bugs)
            self.unique_bugs_inds_list = list(np.array(self.bugs_inds_list)[distinct_inds])



            # record time elapsed and bug numbers
            self.time_list.append(time_elapsed)
            self.bug_num_list.append(self.num_of_bugs)
            self.unique_bug_num_list.append(self.num_of_unique_bugs)





        job_results = []

        if self.run_parallelization:
            with LocalCluster(scheduler_port=self.scheduler_port, dashboard_address=self.dashboard_address, n_workers=len(self.ports), threads_per_worker=1) as cluster, Client(cluster, connection_limit=8192) as client:
                workers = []
                for k in client.has_what():
                    workers.append(k[len('tcp://'):])

                assert X.shape[0] >= len(self.ports), print(X)

                rng = np.random.default_rng(random_seeds[1])
                submit_and_run_jobs(0, len(self.ports), True, job_results)

                time_elapsed = time.time() - self.start_time


                if X.shape[0] > len(self.ports):
                    rng = np.random.default_rng(random_seeds[2])
                    submit_and_run_jobs(len(self.ports), X.shape[0], False, job_results)


                time_elapsed = time.time() - self.start_time
                print('\n'*10)
                print('+'*100)
                mean_objectives_this_generation = np.mean(np.array(self.objectives_list[-X.shape[0]:]), axis=0)

                print(self.counter, time_elapsed, self.num_of_bugs, self.num_of_unique_bugs, self.num_of_collisions, self.num_of_offroad, self.num_of_wronglane, mean_objectives_this_generation)
                print(self.bugs_inds_list)
                print(self.unique_bugs_inds_list)
                for i in range(X.shape[0]-1):
                    for j in range(i+1, X.shape[0]):
                        if np.sum(X[i]-X[j])==0:
                            print(X.shape[0], i, j, 'same')

                with open(mean_objectives_across_generations_path, 'a') as f_out:
                    f_out.write(','.join([str(x) for x in [self.counter, self.has_run, time_elapsed, self.num_of_bugs, self.num_of_unique_bugs, self.num_of_unique_bugs, self.num_of_collisions, self.num_of_offroad, self.num_of_wronglane]]+[str(x) for x in mean_objectives_this_generation])+'\n')
                    f_out.write(';'.join([str(ind) for ind in self.unique_bugs_inds_list])+'\n')
                print('+'*100)
                print('\n'*10)
                os.system('sudo chmod -R 777 '+self.bug_folder)


        else:
            for i in range(X.shape[0]):
                x = np.concatenate([X[i], np.array(self.ports)])
                if i == 0:
                    launch_server = True
                else:
                    launch_server = False

                F, loc, object_type, info, objectives, has_run, _ = fun(x, launch_server)
                job_results.append(F)

                # record bug
                if objectives[0] > 0 or objectives[5] or objectives[6]:
                    self.num_of_bugs += 1


                # record specs for bugs
                time_elapsed = time.time() - self.start_time
                self.time_list.append(time_elapsed)
                self.bug_num_list.append(self.num_of_bugs)

                print('+'*100)
                print(self.counter, time_elapsed, self.num_of_bugs)
                print('+'*100)

                self.counter += 1




        out["F"] = np.row_stack(job_results)




def run_simulation(customized_data, launch_server, episode_max_time, call_from_dt, town_name, scenario, direction, route_str, scenario_file, ego_car_model, ego_car_model_path=None, rerun=False, rerun_folder=None):
    arguments = arguments_info()
    arguments.port = customized_data['port']
    arguments.debug = 1
    if rerun:
        arguments.debug = 0



    if ego_car_model == 'lbc':
        arguments.agent = 'scenario_runner/team_code/image_agent.py'
        arguments.agent_config = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/models/epoch=24.ckpt'
        # arguments.agent_config = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/models/stage2_retrain_9_50_leading_car_25_epoch=21.ckpt'
        base_save_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized'
    elif ego_car_model == 'auto_pilot':
        arguments.agent = 'leaderboard/team_code/auto_pilot.py'
        arguments.agent_config = ''
        base_save_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_autopilot'
    elif ego_car_model == 'pid_agent':
        arguments.agent = 'scenario_runner/team_code/pid_agent.py'
        arguments.agent_config = ''
        base_save_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_pid_agent'
    elif ego_car_model == 'map_model':
        arguments.agent = 'scenario_runner/team_code/map_agent.py'
        arguments.agent_config = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/models/stage1_default_50_epoch=16.ckpt'
        base_save_folder = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_map_model'
    else:
        print('unknown ego_car_model:', ego_car_model)

    if ego_car_model_path:
        arguments.agent_config = ego_car_model_path


    if rerun:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, '/rerun', str(int(arguments.port)), str(call_from_dt)])
    else:
        os.environ['SAVE_FOLDER'] = make_hierarchical_dir([base_save_folder, str(int(arguments.port)), str(call_from_dt)])



    arguments.scenarios = scenario_file




    statistics_manager = StatisticsManager()


    # Fixed Hyperparameters
    multi_actors_scenarios = ['Scenario12']
    # sample_factor is an integer between [1, 8]
    sample_factor = 5
    weather_index = customized_data['weather_index']


    # Laundry Stuff-------------------------------------------------------------
    arguments.weather_index = weather_index
    os.environ['WEATHER_INDEX'] = str(weather_index)

    town_scenario_direction = town_name + '/' + scenario

    folder_1 = os.environ['SAVE_FOLDER'] + '/' + town_name
    if not os.path.exists(folder_1):
        os.mkdir(folder_1)
    folder_2 = folder_1 + '/' + scenario
    if not os.path.exists(folder_2):
        os.mkdir(folder_2)

    if scenario in multi_actors_scenarios:
        town_scenario_direction += '/' + direction
        folder_2 += '/' + direction
        if not os.path.exists(folder_2):
            os.mkdir(folder_2)

    os.environ['SAVE_FOLDER'] = folder_2
    arguments.save_folder = os.environ['SAVE_FOLDER']

    route_prefix = 'leaderboard/data/customized_routes/' + town_scenario_direction + '/route_'

    arguments.routes = route_prefix + route_str + '.xml'
    os.environ['ROUTES'] = arguments.routes

    save_path = os.path.join(arguments.save_folder, 'route_'+route_str)

    # TBD: for convenience
    arguments.deviations_folder = save_path


    # extract waypoints along route
    import xml.etree.ElementTree as ET
    tree = ET.parse(arguments.routes)
    route_waypoints = []



    # this iteration should only go once since we only keep one route per file
    for route in tree.iter("route"):
        route_id = route.attrib['id']
        route_town = route.attrib['town']

        for waypoint in route.iter('waypoint'):
            route_waypoints.append(create_transform(float(waypoint.attrib['x']), float(waypoint.attrib['y']), float(waypoint.attrib['z']), float(waypoint.attrib['pitch']), float(waypoint.attrib['yaw']), float(waypoint.attrib['roll'])))


    # --------------------------------------------------------------------------



    customized_data['using_customized_route_and_scenario'] = True
    customized_data['destination'] = route_waypoints[-1].location
    customized_data['sample_factor'] = sample_factor
    customized_data['number_of_attempts_to_request_actor'] = 10




    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager, launch_server, episode_max_time)
        leaderboard_evaluator.run(arguments, customized_data)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator
        # collect signals for estimating objectives


        objectives, loc, object_type = estimate_objectives(save_path)




    info = {'episode_max_time':episode_max_time,
    'call_from_dt':call_from_dt,
    'town_name':town_name,
    'scenario':scenario,
    'direction':direction,
    'route_str':route_str,
    'ego_car_model':ego_car_model}


    if rerun:
        is_bug = objectives[0] > 0 or objectives[5] or objectives[6]
        if is_bug:
            print('\n'*3, 'rerun also causes a bug!!! will not save this', '\n'*3)
        else:
            assert rerun_folder
            try:
                # use this version to merge into the existing folder
                from distutils.dir_util import copy_tree
                copy_tree(save_path, rerun_folder)
            except:
                print('fail to copy from', save_path)
                traceback.print_exc()


    return objectives, loc, object_type, info, save_path




def estimate_objectives(save_path):
    events_path = os.path.join(save_path, 'events.txt')
    deviations_path = os.path.join(save_path, 'deviations.txt')

    # hack: threshold to avoid too large influence
    ego_linear_speed = 0
    min_d = 7
    offroad_d = 7
    wronglane_d = 7
    dev_dist = 0

    is_offroad = 0
    is_wrong_lane = 0
    is_run_red_light = 0


    with open(deviations_path, 'r') as f_in:
        for line in f_in:
            type, d = line.split(',')
            d = float(d)
            if type == 'min_d':
                min_d = np.min([min_d, d])
            elif type == 'offroad_d':
                offroad_d = np.min([offroad_d, d])
            elif type == 'wronglane_d':
                wronglane_d = np.min([wronglane_d, d])
            elif type == 'dev_dist':
                dev_dist = np.max([dev_dist, d])





    x = None
    y = None
    object_type = None

    infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'off_road']

    try:
        with open(events_path) as json_file:
            events = json.load(json_file)
    except:
        print('events_path', events_path, 'is not found')
        return [0, 7, 7, 7, 0, 0, 0, 0], (None, None), None

    infractions = events['_checkpoint']['records'][0]['infractions']
    status = events['_checkpoint']['records'][0]['status']

    for infraction_type in infraction_types:
        for infraction in infractions[infraction_type]:
            if 'collisions' in infraction_type:
                typ = re.search('.*with type=(.*) and id.*', infraction)
                print(infraction, typ)
                if typ:
                    object_type = typ.group(1)
                loc = re.search('.*x=(.*), y=(.*), z=(.*), ego_linear_speed=(.*), other_actor_linear_speed=(.*)\)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    ego_linear_speed = float(loc.group(4))
                    other_actor_linear_speed = float(loc.group(5))

            elif infraction_type == 'off_road':
                loc = re.search('.*x=(.*), y=(.*), z=(.*)\)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    is_offroad = 1
            else:
                if infraction_type == 'wrong_lane':
                    is_wrong_lane = 1
                elif infraction_type == 'red_light':
                    is_run_red_light = 1
                loc = re.search('.*x=(.*), y=(.*), z=(.*)[\),]', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))

    # limit impact of too large values
    ego_linear_speed = np.min([ego_linear_speed, 7])
    dev_dist = np.min([dev_dist, 7])

    return [ego_linear_speed, min_d, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light], (x, y), object_type



class MySampling(Sampling):
    '''
    dimension correspondence

    Define:
    n1=problem.waypoints_num_limit
    n2=problem.num_of_static_max
    n3=problem.num_of_pedestrians_max
    n4=problem.num_of_vehicles_max

    global
    0: friction, real, [0, 1].
    1: weather_index, int, [0, problem.num_of_weathers].
    2: num_of_static, int, [0, n2].
    3: num_of_pedestrians, int, [0, n3].
    4: num_of_vehicles, int, [0, n4].

    ego-car
    5 ~ 4+n1*2: waypoints perturbation [(dx_i, dy_i)] with length n1.
    dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].


    static
    5+n1*2 ~ 4+n1*2+n2*4: [(static_type_i, x w.r.t. center, y w.r.t. center, yaw)] with length n2.
    static_type_i, int, [0, problem.num_of_static_types).
    x_i, real, [problem.static_x_min, problem.static_x_max].
    y_i, real, [problem.static_y_min, problem.static_y_max].
    yaw_i, real, [problem.yaw_min, problem.yaw_max).

    pedestrians
    5+n1*2+n2*4 ~ 4+n1*2+n2*4+n3*7: [(pedestrian_type_i, x_i, y_i, yaw_i, trigger_distance_i, speed_i, dist_to_travel_i)] with length n3.
    pedestrian_type_i, int, [0, problem.num_of_static_types)
    x_i, real, [problem.pedestrian_x_min, problem.pedestrian_x_max].
    y_i, real, [problem.pedestrian_y_min, problem.pedestrian_y_max].
    yaw_i, real, [problem.yaw_min, problem.yaw_max).
    trigger_distance_i, real, [problem.pedestrian_trigger_distance_min, problem.pedestrian_trigger_distance_max].
    speed_i, real, [problem.pedestrian_speed_min, problem.pedestrian_speed_max].
    dist_to_travel_i, real, [problem.pedestrian_dist_to_travel_min, problem.pedestrian_dist_to_travel_max].

    vehicles
    5+n1*2+n2*4+n3*7 ~ 4+n1*2+n2*4+n3*7+n4*(14+n1*2): [(vehicle_type_i, x_i, y_i, yaw_i, initial_speed_i, trigger_distance_i, targeted_speed_i, waypoint_follower_i, targeted_x_i, targeted_y_i, avoid_collision_i, dist_to_travel_i, target_yaw_i, color_i, [(dx_i, dy_i)] with length n1)] with length n4.
    vehicle_type_i, int, [0, problem.num_of_vehicle_types)
    x_i, real, [problem.vehicle_x_min, problem.vehicle_x_max].
    y_i, real, [problem.vehicle_y_min, problem.vehicle_y_max].
    yaw_i, real, [problem.yaw_min, problem.yaw_max).
    initial_speed_i, real, [problem.vehicle_initial_speed_min, problem.vehicle_initial_speed_max].
    trigger_distance_i, real, [problem.vehicle_trigger_distance_min, problem.vehicle_trigger_distance_max].
    targeted_speed_i, real, [problem.vehicle_targeted_speed_min, problem.vehicle_targeted_speed_max].
    waypoint_follower_i, boolean, [0, 1]
    targeted_x_i, real, [problem.targeted_x_min, problem.targeted_x_max].
    targeted_y_i, real, [problem.targeted_y_min, problem.targeted_y_max].
    avoid_collision_i, boolean, [0, 1]
    dist_to_travel_i, real, [problem.vehicle_dist_to_travel_min, problem.vehicle_dist_to_travel_max].
    target_yaw_i, real, [problem.yaw_min, problem.yaw_max).
    color_i, int, [0, problem.num_of_vehicle_colors).
    dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].


    '''
    def _do(self, problem, n_samples, **kwargs):
        xl = problem.xl
        xu = problem.xu
        mask = problem.mask
        labels = problem.labels
        parameters_distributions = problem.parameters_distributions
        max_sample_times = 300

        X = []
        print('n_samples', n_samples)
        for i in range(n_samples):
            success = False
            for i in range(max_sample_times):
                x = []
                for i, dist in enumerate(parameters_distributions):
                    typ = mask[i]
                    lower = xl[i]
                    upper = xu[i]
                    label = labels[i]
                    if typ == 'int':
                        val = rng.integers(lower, upper+1)
                    elif typ == 'real':
                        if dist[0] == 'normal':
                            if dist[1] == None:
                                mean = (lower+upper)/2
                            else:
                                mean = dist[1]
                            val = np.clip(rng.normal(mean, dist[2], 1)[0], lower, upper)
                        else: # default is uniform
                            val = rand_real(rng, lower, upper)
                    x.append(val)
                if not if_volate_constraints(x, problem.customized_constraints, problem.labels):
                    success = True
                    break
            if not success:
                raise



            x = np.array(x).astype(float)
            X.append(x)
        X = np.stack(X)
        # X = normalize_by_entry(problem, X)

        return X

# def normalize_by_entry(problem, X):
#     for i in np.where((problem.xu - problem.xl) == 0)[0]:
#         print(i, problem.labels[i])
#     return (X - problem.xl) / (problem.xu - problem.xl)
#
# def denormalize_by_entry(problem, X):
#     return X * (problem.xu - problem.xl) + problem.xl




class NSGA2_DT(NSGA2):
    def __init__(self, dt=False, X=None, F=None, **kwargs):
        self.dt = dt
        self.X = X
        self.F = F

        super().__init__(**kwargs)


    def _initialize(self):
        if self.dt:
            X_list = list(self.X)
            F_list = list(self.F)
            pop = Population(len(X_list), individual=Individual())
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen, "CV", [0 for _ in range(len(X_list))], "feasible", [[True] for _ in range(len(X_list))])

            self.evaluator.eval(self.problem, pop, algorithm=self)

            if self.survival:
                pop = self.survival.do(self.problem, pop, len(pop), algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)

            self.pop, self.off = pop, pop
        else:
            super()._initialize()



class NSGA3_DT(NSGA3):
    def __init__(self, dt=False, X=None, F=None, **kwargs):
        self.dt = dt
        self.X = X
        self.F = F

        super().__init__(**kwargs)


    def _initialize(self):
        if self.dt:
            X_list = list(self.X)
            F_list = list(self.F)
            pop = Population(len(X_list), individual=Individual())
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen, "CV", [0 for _ in range(len(X_list))], "feasible", [[True] for _ in range(len(X_list))])

            self.evaluator.eval(self.problem, pop, algorithm=self)

            if self.survival:
                pop = self.survival.do(self.problem, pop, len(pop), algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)

            self.pop, self.off = pop, pop
        else:
            super()._initialize()





class ClipRepair(Repair):
    """
    A dummy class which can be used to simply do no repair.
    """

    def do(self, problem, pop, **kwargs):
        for i in range(len(pop)):
            pop[i].X = np.clip(pop[i].X, np.array(problem.xl), np.array(problem.xu))
        return pop


class SimpleDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, mask, xu, xl, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask = np.array(mask)
        self.xu = np.array(xu)
        self.xl = np.array(xl)
        self.cmp = lambda a, b: self.is_equal(a, b)

        self.c = 1
        self.p = 0
        self.th = 5

    # def do(self, pop, *args, return_indices=False, to_itself=True):
    #     original = pop
    #
    #     if len(pop) == 0:
    #         return pop
    #
    #     if to_itself:
    #         pop = pop[~self._do(pop, None, np.full(len(pop), False))]
    #
    #     for arg in args:
    #         if len(arg) > 0:
    #
    #             if len(pop) == 0:
    #                 break
    #             elif len(arg) == 0:
    #                 continue
    #             else:
    #                 pop = pop[~self._do(pop, arg, np.full(len(pop), False))]
    #
    #     if return_indices:
    #         no_duplicate, is_duplicate = [], []
    #         H = set(pop)
    #
    #         for i, ind in enumerate(original):
    #             if ind in H:
    #                 no_duplicate.append(i)
    #             else:
    #                 is_duplicate.append(i)
    #
    #         return pop, no_duplicate, is_duplicate
    #     else:
    #         return pop


    def is_equal(self, a, b):
        if type(b).__module__ == np.__name__:
            b_X = b
        else:
            b_X = b.X
        return is_similar(a.X, b_X, self.mask, self.xl, self.xu, self.p, self.c, self.th)


class MyMating(Mating):
    def do(self, problem, pop, n_offsprings, **kwargs):

        # the population object to be used
        off = pop.new()

        # infill counter - counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0

        # iterate until enough offsprings are created
        while len(off) < n_offsprings:

            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off = self._do(problem, pop, n_remaining, **kwargs)

            # repair the individuals if necessary - disabled if repair is NoRepair
            _off = self.repair.do(problem, _off, **kwargs)

            # eliminate the duplicates - disabled if it is NoRepair
            _off = self.eliminate_duplicates.do(_off, problem.unique_bugs)

            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:
                # IMPORTANT: Interestingly, this makes a difference in performance
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]

            # add to the offsprings and increase the mating counter
            off = Population.merge(off, _off)
            n_infills += 1

            # if no new offsprings can be generated within a pre-specified number of generations
            if n_infills > self.n_max_iterations:
                break

        return off





class MyEvaluator(Evaluator):


    def _eval(self, problem, pop, **kwargs):

        super()._eval(problem, pop, **kwargs)
        # print(pop[0].X)
        # hack:
        label_to_id = {label:i for i, label in enumerate(problem.labels)}

        def correct_spawn_locations(all_final_generated_transforms_list_i, i, object_type, keys):
            object_type_plural = object_type
            if object_type in ['pedestrian', 'vehicle']:
                object_type_plural += 's'

            num_of_objects_ind = label_to_id['num_of_'+object_type_plural]
            pop[i].X[num_of_objects_ind] = 0

            empty_slots = deque()
            for j, (x, y, yaw) in enumerate(all_final_generated_transforms_list_i[object_type]):
                if x == None:
                    empty_slots.append(j)
                else:
                    pop[i].X[num_of_objects_ind] += 1
                    x_j_ind = label_to_id[object_type+'_x_'+str(j)]
                    y_j_ind = label_to_id[object_type+'_y_'+str(j)]
                    yaw_j_ind = label_to_id[object_type+'_yaw_'+str(j)]


                    # print(object_type, j)
                    # print('x', pop[i].X[x_j_ind], '->', x)
                    # print('y', pop[i].X[y_j_ind], '->', y)
                    # print('yaw', pop[i].X[yaw_j_ind], '->', yaw)
                    pop[i].X[x_j_ind] = x
                    pop[i].X[y_j_ind] = y
                    pop[i].X[yaw_j_ind] = yaw

                    if len(empty_slots) > 0:
                        q = empty_slots.popleft()
                        print('shift', j, 'to', q)
                        for k in keys:
                            print(k)
                            ind_to = label_to_id[k+'_'+str(q)]
                            ind_from = label_to_id[k+'_'+str(j)]
                            pop[i].X[ind_to] = pop[i].X[ind_from]
                        if object_type == 'vehicle':
                            for p in range(waypoints_num_limit):
                                for waypoint_label in waypoint_labels:
                                    ind_to = label_to_id['_'.join(['vehicle', str(q), waypoint_label, str(p)])]
                                    ind_from = label_to_id['_'.join(['vehicle', str(j), waypoint_label, str(p)])]
                                    pop[i].X[ind_to] = pop[i].X[ind_from]

                        empty_slots.append(j)
            # print()


        with open('tmp_folder/total.pickle', 'rb') as f_in:
            all_final_generated_transforms_list = pickle.load(f_in)

        for i, all_final_generated_transforms_list_i in enumerate(all_final_generated_transforms_list):
            if all_final_generated_transforms_list_i:
                # print(i)
                correct_spawn_locations(all_final_generated_transforms_list_i, i, 'static', static_general_labels)
                correct_spawn_locations(all_final_generated_transforms_list_i, i, 'pedestrian', pedestrian_general_labels)
                correct_spawn_locations(all_final_generated_transforms_list_i, i, 'vehicle', vehicle_general_labels)
                # print('\n'*3)
        # print(pop[0].X)






def customized_minimize(problem,
             algorithm,
             resume_run,
             termination=None,
             **kwargs):
    """

    Minimization of function of one or more variables, objectives and constraints.

    This is used as a convenience function to execute several algorithms with default settings which turned
    out to work for a test single. However, evolutionary computations utilizes the idea of customizing a
    meta-algorithm. Customizing the algorithm using the object oriented interface is recommended to improve the
    convergence.

    Parameters
    ----------

    problem : :class:`~pymoo.model.problem.Problem`
        A problem object which is defined using pymoo.

    algorithm : :class:`~pymoo.model.algorithm.Algorithm`
        The algorithm object that should be used for the optimization.

    termination : :class:`~pymoo.model.termination.Termination` or tuple
        The termination criterion that is used to stop the algorithm.

    seed : integer
        The random seed to be used.

    verbose : bool
        Whether output should be printed or not.

    display : :class:`~pymoo.util.display.Display`
        Each algorithm has a default display object for printouts. However, it can be overwritten if desired.

    callback : :class:`~pymoo.model.callback.Callback`
        A callback object which is called each iteration of the algorithm.

    save_history : bool
        Whether the history should be stored or not.

    Returns
    -------
    res : :class:`~pymoo.model.result.Result`
        The optimization result represented as an object.

    """
    # create a copy of the algorithm object to ensure no side-effects
    algorithm = copy.deepcopy(algorithm)

    # get the termination if provided as a tuple - create an object
    if termination is not None and not isinstance(termination, Termination):
        if isinstance(termination, str):
            termination = get_termination(termination)
        else:
            termination = get_termination(*termination)


    # initialize the algorithm object given a problem
    algorithm.initialize(problem, termination=termination, **kwargs)

    # if no termination could be found add the default termination either for single or multi objective
    if algorithm.termination is None:
        if problem.n_obj > 1:
            algorithm.termination = MultiObjectiveDefaultTermination()
        else:
            algorithm.termination = SingleObjectiveDefaultTermination()

    # actually execute the algorithm
    res = algorithm.solve()

    # store the deep copied algorithm in the result object
    res.algorithm = algorithm

    return res


def run_ga(call_from_dt=False, dt=False, X=None, F=None, estimator=None, critical_unique_leaves=None, n_gen_from_dt=0, pop_size_from_dt=0, dt_time_str=None, dt_iter=None, route_type=None, scenario_type='default', ego_car_model='lbc', objective_weights=np.array([-1, 1, 1, 1, -1])):

    if call_from_dt:
        termination_condition = 'generations'
        n_gen = n_gen_from_dt
        scheduler_port = 8791
        dashboard_address = 8792
        ports = [2021, 2024]
        pop_size = pop_size_from_dt

        if dt and len(list(X)) == 0:
            print('No critical leaves!!! Start from random sampling!!!')
            dt = False

    else:
        termination_condition = global_termination_condition
        n_gen = global_n_gen
        scheduler_port = global_scheduler_port
        dashboard_address = global_dashboard_address
        ports = global_ports
        pop_size = global_pop_size

        route_type = global_route_type

        scenario_type = global_scenario_type
        ego_car_model = global_ego_car_model
        objective_weights = global_objective_weights






    # scenario_type = 'leading_car_braking'
    customized_d = customized_bounds_and_distributions[scenario_type]
    route_info = customized_routes[route_type]

    town_name = route_info['town_name']
    scenario = 'Scenario12' # This is only for compatibility purpose
    direction = route_info['direction']
    route = route_info['route_id']
    location_list = route_info['location_list']

    route_str = str(route)
    if route < 10:
        route_str = '0'+route_str

    parse_route_and_scenario(location_list, town_name, scenario, direction, route_str, scenario_file)




    bug_folder_names = [bug_root_folder, str(call_from_dt), algorithm_name, town_name, scenario, direction, route_str]
    bug_parent_folder = make_hierarchical_dir(bug_folder_names)

    non_bug_folder_names = [non_bug_root_folder, str(call_from_dt), algorithm_name, town_name, scenario, direction, route_str]
    non_bug_parent_folder = make_hierarchical_dir(non_bug_folder_names)




    if resume_run:
        with open(save_path, 'rb') as f_in:
            problem = pickle.load(f_in)

    else:
        problem = MyProblem(elementwise_evaluation=False, bug_parent_folder=bug_parent_folder, non_bug_parent_folder=non_bug_parent_folder, town_name=town_name, scenario=scenario, direction=direction, route_str=route_str, scenario_file=scenario_file, ego_car_model=ego_car_model, run_parallelization=run_parallelization, scheduler_port=scheduler_port, dashboard_address=dashboard_address, customized_config=customized_d, ports=ports, episode_max_time=episode_max_time,
        call_from_dt=call_from_dt, dt=dt, estimator=estimator, critical_unique_leaves=critical_unique_leaves, dt_time_str=dt_time_str, dt_iter=dt_iter, objective_weights=objective_weights)




    # deal with real and int separately
    crossover = MixedVariableCrossover(problem.mask, {
        "real": get_crossover("real_sbx", prob=0.6, eta=20),
        "int": get_crossover("int_sbx", prob=0.6, eta=20)
    })

    mutation = MixedVariableMutation(problem.mask, {
        "real": get_mutation("real_pm", eta=20.0, prob=1/problem.n_var),
        "int": get_mutation("int_pm", eta=20.0, prob=1/problem.n_var)
    })


    if algorithm_name == 'nsga3':
        selection = TournamentSelection(func_comp=comp_by_cv_then_random)
    else: # nsga2
        selection = TournamentSelection(func_comp=binary_tournament)

    repair = ClipRepair()

    if use_unique_bugs:
        eliminate_duplicates = SimpleDuplicateElimination(mask=problem.mask, xu=problem.xu, xl=problem.xl)
        mating = MyMating(selection,
                        crossover,
                        mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates,
                        n_max_iterations=100)
    else:
        eliminate_duplicates = NoDuplicateElimination()
        mating = None





    # TBD: customize mutation and crossover to better fit our problem. e.g.
    # might deal with int and real separately
    if algorithm_name == 'nsga2':
        algorithm = NSGA2_DT(dt=dt, X=X, F=F,
                      pop_size=pop_size,
                      sampling=MySampling(),
                      crossover=crossover,
                      mutation=mutation,
                      eliminate_duplicates=eliminate_duplicates,
                      repair=repair,
                      mating=mating)
    elif algorithm_name == 'nsga3':
        algorithm = NSGA3_DT(dt=dt, X=X, F=F,
                      pop_size=pop_size,
                      sampling=MySampling(),
                      crossover=crossover,
                      mutation=mutation,
                      eliminate_duplicates=eliminate_duplicates,
                      repair=repair,
                      mating=mating)
    elif algorithm_name == 'random':
        algorithm = RandomAlgorithm(pop_size=pop_size,
                                    sampling=MySampling(),
                                    eliminate_duplicates=eliminate_duplicates,
                                    repair=repair,
                                    mating=mating)



    if termination_condition == 'generations':
        termination = ('n_gen', n_gen)
    elif termination_condition == 'max_time':
        termination = ('time', max_running_time)
    else:
        termination = ('n_gen', n_gen)


    # close simulator(s)
    atexit.register(exit_handler, ports, problem.bug_folder, scenario_file)

    # TypeError: can't pickle _asyncio.Task objects when save_history = True
    res = customized_minimize(problem,
                   algorithm,
                   resume_run,
                   termination=termination,
                   seed=0,
                   verbose=True,
                   save_history=False,
                   evaluator=MyEvaluator())

    print('We have found', problem.num_of_bugs, 'bugs in total.')


    # print("Best solution found: %s" % res.X)
    # print("Function value: %s" % res.F)
    # print("Constraint violation: %s" % res.CV)

    # for drawing hv
    # create the performance indicator object with reference point
    metric = Hypervolume(ref_point=np.array([7.0, 7.0, 7.0, 7.0, 7.0]))
    # collect the population in each generation
    pop_each_gen = [a.pop for a in res.history]
    # receive the population in each generation
    obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]
    # calculate for each generation the HV metric
    hv = np.array([metric.calc(f) for f in obj_and_feasible_each_gen])



    # print(problem.x_list, problem.y_list, problem.F_list, problem.objectives_list)

    X = np.stack(problem.x_list)
    y = np.array(problem.y_list)
    F = np.stack(problem.F_list)
    objectives = np.stack(problem.objectives_list)
    time_list = np.array(problem.time_list)
    bug_num_list = np.array(problem.bug_num_list)
    labels = problem.labels
    has_run = problem.has_run

    mask = problem.mask
    xl = problem.xl
    xu = problem.xu
    p = problem.p
    c = problem.c
    th = problem.th


    # with open(os.path.join(problem.bug_folder, 'res_'+str(ind)+'.pkl'), 'wb') as f_out:
    #     pickle.dump({'X':X, 'y':y, 'F':F, 'objectives':objectives, 'n_gen':n_gen, 'pop_size':pop_size, 'hv':hv, 'time_list':time_list, 'bug_num_list':problem.bug_num_list}, f_out)
    #     print('-'*100, 'pickled')


    # save another data npz for easy comparison with dt results
    non_dt_save_folder = 'non_dt_data'
    if not os.path.exists(non_dt_save_folder):
        os.mkdir(non_dt_save_folder)
    now = datetime.now()
    non_dt_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    non_dt_save_file = '_'.join([route_type, scenario_type, str(n_gen), str(pop_size), non_dt_time_str])

    pth = os.path.join(non_dt_save_folder, non_dt_save_file)
    np.savez(pth, X=X, y=y, F=F, objectives=objectives, time=time_list, bug_num=bug_num_list, labels=labels, hv=hv, has_run=has_run, mask=mask, xl=xl, xu=xu, p=p, c=c, th=th, route_type=route_type, scenario_type=scenario_type)
    print('non_dt npz saved')


    if save:
        with open(save_path, 'wb') as f_out:
            pickle.dump(problem, f_out)
            print('-'*100, 'pickled')


    return X, y, F, objectives, time_list, bug_num_list, labels, has_run, hv

if __name__ == '__main__':
    run_ga()

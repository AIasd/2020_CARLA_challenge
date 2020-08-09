'''
TBD:

* fix stage2 model training
* debug nsga2-dt (make sure decision tree code is correct)
* introduction writing

* emcmc
* clustering+tsne(need to label different bugs first), bug category over generation plot
* retraining



* Traceback (most recent call last):
srunner.scenariomanager.carla_data_provider.get_velocity: Actor(id=0, type=static.road) not found!
Traceback (most recent call last):
  File "scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py", line 315, in <lambda>
Traceback (most recent call last):
  File "scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py", line 315, in <lambda>
  File "scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py", line 315, in <lambda>
    self._collision_sensor.listen(lambda event: self._count_collisions(weakref.ref(self), event))
  File "scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py", line 543, in _count_collisions
    'x': actor_location.x,
    self._collision_sensor.listen(lambda event: self._count_collisions(weakref.ref(self), event))
    self._collision_sensor.listen(lambda event: self._count_collisions(weakref.ref(self), event))
  File "scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py", line 543, in _count_collisions
  File "scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py", line 543, in _count_collisions
AttributeError: 'NoneType' object has no attribute 'x'
    'x': actor_location.x,
    'x': actor_location.x,
AttributeError: 'NoneType' object has no attribute 'x'
AttributeError: 'NoneType' object has no attribute 'x'
 waiting for one data reading from sensors...
Traceback (most recent call last):
  File "scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py", line 315, in <lambda>
    self._collision_sensor.listen(lambda event: self._count_collisions(weakref.ref(self), event))
  File "scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py", line 543, in _count_collisions
    'x': actor_location.x,
AttributeError: 'NoneType' object has no attribute 'x'






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
CUDA_VISIBLE_DEVICES=0 python carla_project/src/image_model.py --dataset_dir '../LBC_data/CARLA_challenge_autopilot' --teacher_path '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/checkpoints/32176d62026a4063a2de10e92ebdb03c/wandb/run-20200807_173829-32176d62026a4063a2de10e92ebdb03c/epoch=16.ckpt' --max_epochs 25

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
from pymoo.model.duplicate import ElementwiseDuplicateElimination
from pymoo.model.population import Population

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.random import RandomAlgorithm

from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover

from pymoo.performance_indicator.hv import Hypervolume

import matplotlib.pyplot as plt

from object_types import WEATHERS, pedestrian_types, vehicle_types, static_types, vehicle_colors, car_types, motorcycle_types, cyclist_types

from customized_utils import create_transform, rand_real, specify_args, convert_x_to_customized_data, make_hierarchical_dir, exit_handler, arguments_info, is_critical_region

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




import copy

from pymoo.factory import get_termination
from pymoo.model.termination import Termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from pymoo.model.individual import Individual

from dask.distributed import Client, LocalCluster




rng = np.random.default_rng(20)
bug_root_folder = 'bugs'
town_name = 'Town05'
scenario = 'Scenario12'
direction = 'right'
route = 0

route_str = str(route)
if route < 10:
    route_str = '0'+route_str


# ['nsga2', 'random']
algorithm_name = 'nsga2'
# ['lbc', 'auto_pilot', 'pid_agent']
ego_car_model = 'lbc'
os.environ['HAS_DISPLAY'] = '0'
# This is used to control how this program use GPU
# '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


parser = argparse.ArgumentParser()
parser.add_argument('--resume-run', help='continue to run', default=False, action='store_true')
parser.add_argument('--ind', help='ind ', default=0)
arguments = parser.parse_args()

resume_run = arguments.resume_run
ind = arguments.ind

run_parallelization = True
save = False
save_path = 'ga_intermediate.pkl'
episode_max_time = 10000
global_n_gen = 20
global_pop_size = 100
max_running_time = 3600*24
# [ego_linear_speed, offroad_d, wronglane_d, dev_dist]
objective_weights = np.array([-1/7, 1, 100000, -1])

# ['generations', 'max_time']
global_termination_condition = 'max_time'

global_scheduler_port = 8788
global_dashboard_address = 8789
global_ports = [2000]
if run_parallelization:
    global_scheduler_port = 8785
    global_dashboard_address = 8786
    global_ports = [2003, 2009]








class MyProblem(Problem):

    def __init__(self, elementwise_evaluation, bug_parent_folder, run_parallelization, scheduler_port, dashboard_address, ports=[2000], episode_max_time=10000, call_from_dt=False, dt=False, estimator=None, critical_unique_leaves=None):

        self.call_from_dt = call_from_dt
        self.dt = dt
        self.estimator = estimator
        self.critical_unique_leaves = critical_unique_leaves

        self.objectives_list = []
        self.x_list = []
        self.y_list = []
        self.F_list = []

        self.run_parallelization = run_parallelization
        self.scheduler_port = scheduler_port
        self.dashboard_address = dashboard_address
        self.ports = ports
        self.episode_max_time = episode_max_time


        now = datetime.now()
        self.bug_folder = bug_parent_folder + now.strftime("%Y_%m_%d_%H_%M_%S")
        if not os.path.exists(self.bug_folder):
            os.mkdir(self.bug_folder)


        self.counter = 0
        self.num_of_bugs = 0
        self.num_of_collisions = 0
        self.num_of_offroad = 0
        self.num_of_wronglane = 0

        self.start_time = time.time()
        self.time_elapsed = 0
        self.time_list = []
        self.bug_num_list = []

        # Fixed hyper-parameters
        self.waypoints_num_limit = 5
        self.max_num_of_static = 0
        self.max_num_of_pedestrians = 2
        self.max_num_of_vehicles = 2
        self.num_of_weathers = len(WEATHERS)
        self.num_of_static_types = len(static_types)
        self.num_of_pedestrian_types = len(pedestrian_types)
        self.num_of_vehicle_types = len(vehicle_types)
        self.num_of_vehicle_colors = len(vehicle_colors)

        # general
        self.min_friction = 0.2
        self.max_friction = 0.8
        self.perturbation_min = -1.5
        self.perturbation_max = 1.5
        self.yaw_min = 0
        self.yaw_max = 360

        # static
        self.static_x_min = -20
        self.static_x_max = 20
        self.static_y_min = -20
        self.static_y_max = 20

        # pedestrians
        self.pedestrian_x_min = -20
        self.pedestrian_x_max = 20
        self.pedestrian_y_min = -20
        self.pedestrian_y_max = 20
        self.pedestrian_trigger_distance_min = 2
        self.pedestrian_trigger_distance_max = 50
        self.pedestrian_speed_min = 0
        self.pedestrian_speed_max = 4
        self.pedestrian_dist_to_travel_min = 0
        self.pedestrian_dist_to_travel_max = 50

        # vehicles
        self.vehicle_x_min = -20
        self.vehicle_x_max = 20
        self.vehicle_y_min = -20
        self.vehicle_y_max = 20
        self.vehicle_initial_speed_min = 0
        self.vehicle_initial_speed_max = 15
        self.vehicle_trigger_distance_min = 0
        self.vehicle_trigger_distance_max = 50
        self.vehicle_targeted_speed_min = 0
        self.vehicle_targeted_speed_max = 15
        self.vehicle_targeted_x_min = -20
        self.vehicle_targeted_x_max = 20
        self.vehicle_targeted_y_min = -20
        self.vehicle_targeted_y_max = 20
        self.vehicle_dist_to_travel_min = 0
        self.vehicle_dist_to_travel_max = 50



        # construct xl and xu
        xl = [self.min_friction, 0, 0, 0, 0]
        xu = [self.max_friction,
        self.num_of_weathers-1,
        self.max_num_of_static,
        self.max_num_of_pedestrians,
        self.max_num_of_vehicles
        ]
        mask = ['real', 'int', 'int', 'int', 'int']
        labels = ['friction', 'num_of_weathers', 'max_num_of_static', 'max_num_of_pedestrians', 'max_num_of_vehicles']
        # ego-car
        for i in range(self.waypoints_num_limit):
            xl.extend([self.perturbation_min]*2)
            xu.extend([self.perturbation_max]*2)
            mask.extend(['real']*2)
            labels.extend(['perturbation_x_'+str(i), 'perturbation_y_'+str(i)])
        # static
        for i in range(self.max_num_of_static):
            xl.extend([0, self.static_x_min, self.static_y_min, self.yaw_min])
            xu.extend([self.num_of_static_types-1, self.static_x_max, self.static_y_max, self.yaw_max])
            mask.extend(['int'] + ['real']*3)
            labels.extend(['num_of_static_types_'+str(i), 'static_x_'+str(i), 'static_y_'+str(i), 'yaw_'+str(i)])
        # pedestrians
        for i in range(self.max_num_of_pedestrians):
            xl.extend([0, self.pedestrian_x_min, self.pedestrian_y_min, self.yaw_min, self.pedestrian_trigger_distance_min, self.pedestrian_speed_min, self.pedestrian_dist_to_travel_min])
            xu.extend([self.num_of_pedestrian_types-1, self.pedestrian_x_max, self.pedestrian_y_max, self.yaw_max, self.pedestrian_trigger_distance_max, self.pedestrian_speed_max, self.pedestrian_dist_to_travel_max])
            mask.extend(['int'] + ['real']*6)
            labels.extend(['num_of_pedestrian_types_'+str(i), 'pedestrian_x_'+str(i), 'pedestrian_y_'+str(i), 'yaw_'+str(i), 'pedestrian_trigger_distance_'+str(i), 'pedestrian_speed_'+str(i), 'pedestrian_dist_to_travel_'+str(i)])
        # vehicles
        for i in range(self.max_num_of_vehicles):
            xl.extend([0, self.vehicle_x_min, self.vehicle_y_min, self.yaw_min, self.vehicle_initial_speed_min, self.vehicle_trigger_distance_min, self.vehicle_targeted_speed_min, 0, self.vehicle_targeted_x_min, self.vehicle_targeted_y_min, 0, self.vehicle_dist_to_travel_min, self.yaw_min, 0])

            xu.extend([self.num_of_vehicle_types-1, self.vehicle_x_max, self.vehicle_y_max, self.yaw_max, self.vehicle_initial_speed_max, self.vehicle_trigger_distance_max, self.vehicle_targeted_speed_max, 1, self.vehicle_targeted_x_max, self.vehicle_targeted_y_max, 1, self.vehicle_dist_to_travel_max, self.yaw_max, self.num_of_vehicle_colors-1])
            mask.extend(['int'] + ['real']*6 + ['int'] + ['real']*2 + ['int'] + ['real']*2 + ['int'])
            labels.extend(['num_of_vehicle_types_'+str(i), 'vehicle_x_'+str(i), 'vehicle_y_'+str(i), 'yaw_'+str(i), 'vehicle_initial_speed_'+str(i), 'vehicle_trigger_distance_'+str(i), 'vehicle_targeted_speed_'+str(i), 'waypoint_follower_'+str(i), 'vehicle_targeted_x_'+str(i), 'vehicle_targeted_y_'+str(i), 'avoid_collision_'+str(i), 'vehicle_dist_to_travel_'+str(i), 'yaw_'+str(i), 'num_of_vehicle_colors_'+str(i)])

            for j in range(self.waypoints_num_limit):
                xl.extend([self.perturbation_min]*2)
                xu.extend([self.perturbation_max]*2)
                mask.extend(['real']*2)
                labels.extend(['perturbation_x_'+str(i)+'_'+str(j), 'perturbation_y_'+str(i)+'_'+str(j)])

        n_var = 5+self.waypoints_num_limit*2+self.max_num_of_static*4+self.max_num_of_pedestrians*7+self.max_num_of_vehicles*(14+self.waypoints_num_limit*2)

        self.mask = mask
        self.labels = labels

        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=elementwise_evaluation)






    def _evaluate(self, X, out, *args, **kwargs):



        waypoints_num_limit = self.waypoints_num_limit
        max_num_of_static = self.max_num_of_static
        max_num_of_pedestrians = self.max_num_of_pedestrians
        max_num_of_vehicles = self.max_num_of_vehicles
        episode_max_time = self.episode_max_time
        call_from_dt = self.call_from_dt
        bug_folder = self.bug_folder

        xl = self.xl
        xu = self.xu

        dt = self.dt
        estimator = self.estimator
        critical_unique_leaves = self.critical_unique_leaves

        mean_objectives_across_generations_path = os.path.join(self.bug_folder, 'mean_objectives_across_generations.txt')






        def fun(x, launch_server, counter):
            if dt and not is_critical_region(x, estimator, critical_unique_leaves):
                objectives = [-1, 10000, 10000, 0, 0, 0, 0]
                F = np.array(objectives[:4]) * objective_weights
                return F, None, None, None, objectives

            else:

                x[:-1] = np.clip(x[:-1], np.array(xl), np.array(xu))

                # x = denormalize_by_entry(self, x)

                customized_data = convert_x_to_customized_data(x, waypoints_num_limit, max_num_of_static, max_num_of_pedestrians, max_num_of_vehicles, static_types, pedestrian_types, vehicle_types, vehicle_colors)


                # run simulation
                objectives, loc, object_type, info, save_path = run_simulation(customized_data, launch_server, episode_max_time, call_from_dt)

                # [ego_linear_speed, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light]


                F = np.array(objectives[:4]) * objective_weights


                if objectives[0] > 0 or objectives[4] or objectives[5]:
                    bug = {'counter':counter, 'x':x, 'ego_linear_speed':objectives[0], 'offroad_d':objectives[1], 'wronglane_d':objectives[2], 'dev_dist':objectives[3], 'is_offroad':objectives[4], 'is_wrong_lane':objectives[5], 'is_run_red_light':objectives[6], 'loc':loc, 'object_type':object_type, 'info': info}
                    cur_folder = bug_folder+'/'+str(counter)
                    if not os.path.exists(cur_folder):
                        os.mkdir(cur_folder)
                    np.savez(cur_folder+'/'+'bug_info', bug=bug)
                    # copy data to another place
                    try:
                        new_path = os.path.join(cur_folder, 'data')
                        shutil.copytree(save_path, new_path)
                    except:
                        print('fail to copy from', save_path)


                return F, loc, object_type, info, objectives




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
                F, loc, object_type, info, objectives = job.result()



                # record bug
                if objectives[0] > 0 or objectives[4] or objectives[5]:
                    if objectives[0] > 0:
                        self.num_of_collisions += 1
                        collision_types = {'pedestrian_collision':pedestrian_types, 'car_collision':car_types, 'motercycle_collision':motorcycle_types, 'cyclist_collision':cyclist_types, 'static_collision':static_types}
                        for k,v in collision_types.items():
                            if object_type in v:
                                bug_str = k
                    elif objectives[4]:
                        self.num_of_offroad += 1
                        bug_str = 'offroad'
                    elif objectives[5]:
                        self.num_of_wronglane += 1
                        bug_str = 'wronglane'
                    with open(mean_objectives_across_generations_path, 'a') as f_out:
                        f_out.write(str(i)+','+bug_str+'\n')

                    self.num_of_bugs += 1
                    self.y_list.append(1)
                else:
                    self.y_list.append(0)
                # we don't want to store port number
                self.x_list.append(x[:-1])
                self.F_list.append(F)
                self.objectives_list.append(np.array(objectives))
                job_results.append(F)




            # record time elapsed and bug numbers
            self.time_list.append(time_elapsed)
            self.bug_num_list.append(self.num_of_bugs)





        job_results = []

        if self.run_parallelization:
            with LocalCluster(scheduler_port=self.scheduler_port, dashboard_address=self.dashboard_address, n_workers=len(self.ports), threads_per_worker=1) as cluster, Client(cluster, connection_limit=8192) as client:
                workers = []
                for k in client.has_what():
                    workers.append(k[len('tcp://'):])

                assert X.shape[0] >= len(self.ports), print(X)


                submit_and_run_jobs(0, len(self.ports), True, job_results)

                time_elapsed = time.time() - self.start_time
                print('\n'*10)
                print('+'*100)
                print(self.counter, time_elapsed, self.num_of_bugs)
                print('+'*100)
                print('\n'*10)

                if X.shape[0] > len(self.ports):
                    submit_and_run_jobs(len(self.ports), X.shape[0], False, job_results)


                time_elapsed = time.time() - self.start_time
                print('\n'*10)
                print('+'*100)
                mean_objectives_this_generation = np.mean(np.array(self.objectives_list[-X.shape[0]:]), axis=0)

                print(self.counter, time_elapsed, self.num_of_bugs, self.num_of_collisions, self.num_of_offroad, self.num_of_wronglane, mean_objectives_this_generation)

                with open(mean_objectives_across_generations_path, 'a') as f_out:
                    f_out.write(','.join([str(x) for x in [self.counter, time_elapsed, self.num_of_bugs, self.num_of_collisions, self.num_of_offroad, self.num_of_wronglane]]+[str(x) for x in mean_objectives_this_generation])+'\n')

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

                F, loc, object_type, info, objectives = fun(x, launch_server)
                job_results.append(F)

                # record bug
                if objectives[0] > 0 or objectives[4] or objectives[5]:
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




def run_simulation(customized_data, launch_server, episode_max_time, call_from_dt):
    arguments = arguments_info()
    arguments.port = customized_data['port']
    arguments.debug = 1







    if ego_car_model == 'lbc':
        arguments.agent='scenario_runner/team_code/image_agent.py'
        arguments.agent_config='/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/models/epoch=24.ckpt'
        os.environ['SAVE_FOLDER'] = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_customized'
    elif ego_car_model == 'auto_pilot':
        arguments.agent = 'leaderboard/team_code/auto_pilot.py'
        arguments.agent_config = ''
        os.environ['SAVE_FOLDER'] = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_autopilot'
    elif ego_car_model == 'pid_agent':
        arguments.agent = 'scenario_runner/team_code/pid_agent.py'
        arguments.agent_config = ''
        os.environ['SAVE_FOLDER'] = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/collected_data_pid_agent'



    arguments.scenarios = 'leaderboard/data/fuzzing_scenarios.json'
    if not os.path.exists(os.environ['SAVE_FOLDER']):
        os.mkdir(os.environ['SAVE_FOLDER'])
    os.environ['SAVE_FOLDER'] += '/'+str(int(arguments.port))
    if not os.path.exists(os.environ['SAVE_FOLDER']):
        os.mkdir(os.environ['SAVE_FOLDER'])
    os.environ['SAVE_FOLDER'] += '/'+str(call_from_dt)
    if not os.path.exists(os.environ['SAVE_FOLDER']):
        os.mkdir(os.environ['SAVE_FOLDER'])


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

    # extract waypoints for the scenario
    # world_annotations = RouteParser.parse_annotations_file(arguments.scenarios)
    # info = world_annotations[town_name][0]["available_event_configurations"][0]
    #
    # center = info["center"]
    # RouteParser.convert_waypoint_float(center)
    # center_location = carla.Location(float(center['x']), float(center['y']), float(center['z']))
    # center_rotation = carla.Rotation(float(center['pitch']), float(center['yaw']), 0.0)
    # center_transform = carla.Transform(center_location, center_rotation)

    # use the intermediate waypoint as the center transform

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


    info = [arguments.scenarios, town_name, scenario, direction, route, sample_factor, customized_data['center_transform'].location.x, customized_data['center_transform'].location.y]

    return objectives, loc, object_type, info, save_path




def estimate_objectives(save_path):
    events_path = os.path.join(save_path, 'events.txt')
    deviations_path = os.path.join(save_path, 'deviations.txt')

    offroad_d = 10000
    wronglane_d = 10000
    dev_dist = 0

    with open(deviations_path, 'r') as f_in:
        for line in f_in:
            type, d = line.split(',')
            d = float(d)
            if type == 'offroad_d':
                offroad_d = np.min([offroad_d, d])
            elif type == 'wronglane_d':
                wronglane_d = np.min([wronglane_d, d])
            elif type == 'dev_dist':
                dev_dist = np.max([dev_dist, d])



    ego_linear_speed = -1
    is_offroad = 0
    is_wrong_lane = 0
    is_run_red_light = 0

    x = None
    y = None
    object_type = None

    infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'red_light', 'on_sidewalk', 'outside_lane_infraction', 'wrong_lane', 'off_road']

    try:
        with open(events_path) as json_file:
            events = json.load(json_file)
    except:
        print('events_path', events_path, 'is not found')
        return [ego_linear_speed, offroad_dist, is_wrong_lane, is_run_red_light]

    infractions = events['_checkpoint']['records'][0]['infractions']
    status = events['_checkpoint']['records'][0]['status']

    for infraction_type in infraction_types:
        for infraction in infractions[infraction_type]:
            if 'collisions' in infraction_type:
                typ = re.search('.*with type=(.*) and.*', infraction)
                if typ:
                    object_type = typ.group(1)
                loc = re.search('.*x=(.*), y=(.*), z=(.*), ego_linear_speed=(.*), other_actor_linear_speed=(.*)\)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    ego_linear_speed = float(loc.group(4))
                    other_actor_linear_speed = float(loc.group(5))
            elif infraction_type == 'off_road':
                loc = re.search('.*x=(.*), y=(.*), z=(.*), offroad distance=(.*)\)', infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    offroad_dist = float(loc.group(4))
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


    return [ego_linear_speed, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light], (x, y), object_type



class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = []
        print('n_samples', n_samples)
        for i in range(n_samples):
            '''
            dimension correspondence

            Define:
            n1=problem.waypoints_num_limit
            n2=problem.max_num_of_static
            n3=problem.max_num_of_pedestrians
            n4=problem.max_num_of_vehicles

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



            d = 4+problem.waypoints_num_limit*2+problem.max_num_of_static*4+problem.max_num_of_pedestrians*7+problem.max_num_of_vehicles*(12+problem.waypoints_num_limit*2)

            x = []

            # global
            friction = rng.random()
            weather_index = rng.integers(problem.num_of_weathers)
            num_of_static = rng.integers(problem.max_num_of_static+1)
            num_of_pedestrians = rng.integers(problem.max_num_of_pedestrians+1)
            num_of_vehicles = rng.integers(problem.max_num_of_vehicles+1)
            x.extend([friction, weather_index, num_of_static, num_of_pedestrians, num_of_vehicles])

            # ego car
            for _ in range(problem.waypoints_num_limit):
                dx = np.clip(rng.normal(0, 2, 1)[0], problem.perturbation_min, problem.perturbation_max)
                dy = np.clip(rng.normal(0, 2, 1)[0], problem.perturbation_min, problem.perturbation_max)
                x.extend([dx, dy])
            # static
            for i in range(problem.max_num_of_static):
                static_type_i = rng.integers(problem.num_of_static_types)
                static_x_i = rand_real(rng, problem.static_x_min, problem.static_x_max)
                static_y_i = rand_real(rng, problem.static_y_min, problem.static_y_max)
                static_yaw_i = rand_real(rng, problem.yaw_min, problem.yaw_max)
                x.extend([static_type_i, static_x_i, static_y_i, static_yaw_i])
            # pedestrians
            for i in range(problem.max_num_of_pedestrians):
                pedestrian_type_i = rng.integers(problem.num_of_pedestrian_types)
                pedestrian_x_i = rand_real(rng, problem.pedestrian_x_min, problem.pedestrian_x_max)
                pedestrian_y_i = rand_real(rng, problem.pedestrian_x_min, problem.pedestrian_x_max)
                pedestrian_yaw_i = rand_real(rng, problem.yaw_min, problem.yaw_max)
                pedestrian_trigger_distance_i = rand_real(rng, problem.pedestrian_trigger_distance_min, problem.pedestrian_trigger_distance_max)
                speed_i = rand_real(rng, problem.pedestrian_speed_min, problem.pedestrian_speed_max)
                dist_to_travel_i = rand_real(rng, problem.pedestrian_dist_to_travel_min, problem.pedestrian_dist_to_travel_max)
                x.extend([pedestrian_type_i, pedestrian_x_i, pedestrian_y_i, pedestrian_yaw_i, pedestrian_trigger_distance_i, speed_i, dist_to_travel_i])
            # vehicles
            for i in range(problem.max_num_of_vehicles):
                vehicle_type_i = rand_real(rng, 0, problem.num_of_vehicle_types)
                vehicle_x_i = rand_real(rng, problem.vehicle_x_min, problem.vehicle_x_max)
                vehicle_y_i = rand_real(rng, problem.vehicle_x_min, problem.vehicle_x_max)
                vehicle_yaw_i = rand_real(rng, problem.yaw_min, problem.yaw_max)
                vehicle_initial_speed_i = rand_real(rng, problem.vehicle_initial_speed_min, problem.vehicle_initial_speed_max)
                vehicle_trigger_distance_i = rand_real(rng, problem.vehicle_trigger_distance_min, problem.vehicle_trigger_distance_max)

                targeted_speed_i = rand_real(rng, problem.vehicle_targeted_speed_min, problem.vehicle_targeted_speed_max)
                waypoint_follower_i = rng.integers(2)

                vehicle_targeted_x_i = rand_real(rng, problem.vehicle_targeted_x_min, problem.vehicle_targeted_x_max)
                vehicle_targeted_y_i = rand_real(rng, problem.vehicle_targeted_x_min, problem.vehicle_targeted_x_max)

                vehicle_avoid_collision_i = rng.integers(2)
                vehicle_dist_to_travel_i = rand_real(rng, problem.vehicle_dist_to_travel_min, problem.vehicle_dist_to_travel_max)
                vehicle_target_yaw_i = rand_real(rng, problem.yaw_min, problem.yaw_max)
                vehicle_color_i = rng.integers(0, problem.num_of_vehicle_colors)

                x.extend([vehicle_type_i, vehicle_x_i, vehicle_y_i, vehicle_yaw_i, vehicle_initial_speed_i, vehicle_trigger_distance_i, targeted_speed_i, waypoint_follower_i, vehicle_targeted_x_i, vehicle_targeted_y_i, vehicle_avoid_collision_i, vehicle_dist_to_travel_i, vehicle_target_yaw_i, vehicle_color_i])

                for _ in range(problem.waypoints_num_limit):
                    dx = np.clip(rng.normal(0, 2, 1)[0], problem.perturbation_min, problem.perturbation_max)
                    dy = np.clip(rng.normal(0, 2, 1)[0], problem.perturbation_min, problem.perturbation_max)
                    x.extend([dx, dy])
            x = np.array(x).astype(float)
            X.append(x)
        X = np.stack(X)
        # X = normalize_by_entry(problem, X)

        return X

def normalize_by_entry(problem, X):
    for i in np.where((problem.xu - problem.xl) == 0)[0]:
        print(i, problem.labels[i])
    return (X - problem.xl) / (problem.xu - problem.xl)

def denormalize_by_entry(problem, X):
    return X * (problem.xu - problem.xl) + problem.xl




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
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen)

            self.evaluator.eval(self.problem, pop, algorithm=self)



            if self.survival:
                pop = self.survival.do(self.problem, pop, len(pop), algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)

            self.pop, self.off = pop, pop
        else:
            super()._initialize()






class SimpleDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, mask, xu, xl, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask = np.array(mask)
        self.xu = np.array(xu)
        self.xl = np.array(xl)
        self.cmp = lambda a, b: self.is_equal(a, b)
    def is_equal(self, a, b):
        int_inds = self.mask == 'int'
        real_inds = self.mask == 'real'
        int_diff = np.sum(np.abs(a.X[int_inds] - b.X[int_inds])) == 0
        real_diff = np.sum(np.abs(a.X[real_inds] - b.X[real_inds]) - 0.05 * np.abs(self.xu[real_inds] - self.xl[real_inds])) == 0
        return int_diff and real_diff




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

    if resume_run:
        res = algorithm.solve()
    else:
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


def run_ga(call_from_dt=False, dt=False, X=None, F=None, estimator=None, critical_unique_leaves=None, n_gen_from_dt=0, pop_size_from_dt=0):

    if call_from_dt:
        termination_condition = 'generations'
        n_gen = n_gen_from_dt
        scheduler_port = 8791
        dashboard_address = 8792
        ports = [2021, 2027]
        pop_size = pop_size_from_dt
    else:
        termination_condition = global_termination_condition
        n_gen = global_n_gen
        scheduler_port = global_scheduler_port
        dashboard_address = global_dashboard_address
        ports = global_ports
        pop_size = global_pop_size









    folder_names = [bug_root_folder, str(call_from_dt), algorithm_name, town_name, scenario, direction, route_str]
    bug_parent_folder = make_hierarchical_dir(folder_names)




    if resume_run:
        with open(save_path, 'rb') as f_in:
            algorithm = pickle.load(f_in)

        algorithm.launch_cluster = True
        problem = algorithm.problem
    else:
        problem = MyProblem(elementwise_evaluation=False, bug_parent_folder=bug_parent_folder, run_parallelization=run_parallelization, scheduler_port=scheduler_port, dashboard_address=dashboard_address, ports=ports, episode_max_time=episode_max_time, call_from_dt=call_from_dt, dt=dt, estimator=estimator, critical_unique_leaves=critical_unique_leaves)


        from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
        from pymoo.factory import get_crossover, get_mutation





        # deal with real and int separately
        crossover = MixedVariableCrossover(problem.mask, {
            "real": get_crossover("real_sbx", prob=0.6, eta=20),
            "int": get_crossover("int_sbx", prob=0.6, eta=20)
        })

        mutation = MixedVariableMutation(problem.mask, {
            "real": get_mutation("real_pm", eta=20.0, prob=1/problem.n_var),
            "int": get_mutation("int_pm", eta=20.0, prob=1/problem.n_var)
        })


        # TBD: customize mutation and crossover to better fit our problem. e.g.
        # might deal with int and real separately
        if algorithm_name == 'nsga2':
            algorithm = NSGA2_DT(dt=dt, X=X, F=F,
                          pop_size=pop_size,
                          sampling=MySampling(),
                          crossover=crossover,
                          mutation=mutation,
                          eliminate_duplicates=SimpleDuplicateElimination(mask=problem.mask, xu=problem.xu, xl=problem.xl))
        elif algorithm_name == 'random':
            algorithm = RandomAlgorithm(pop_size=pop_size,
                                        sampling=MySampling(),
                                        eliminate_duplicates=SimpleDuplicateElimination(mask=problem.mask, xu=problem.xu, xl=problem.xl))



    if termination_condition == 'generations':
        termination = ('n_gen', n_gen)
    elif termination_condition == 'max_time':
        termination = ('time', max_running_time)
    else:
        termination = ('n_gen', n_gen)


    # close simulator(s)
    atexit.register(exit_handler, ports, problem.bug_folder)

    # TypeError: can't pickle _asyncio.Task objects when save_history = True
    res = customized_minimize(problem,
                   algorithm,
                   resume_run,
                   termination=termination,
                   seed=0,
                   verbose=True,
                   save_history=False)

    print('We have found', problem.num_of_bugs, 'bugs in total.')


    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)

    # for drawing hv
    # create the performance indicator object with reference point
    metric = Hypervolume(ref_point=np.array([1.0, 1.0, 1.0, 1.0]))
    # collect the population in each generation
    pop_each_gen = [a.pop for a in res.history]
    # receive the population in each generation
    obj_and_feasible_each_gen = [pop[pop.get("feasible")[:,0]].get("F") for pop in pop_each_gen]
    # calculate for each generation the HV metric
    hv = [metric.calc(f) for f in obj_and_feasible_each_gen]



    print(problem.x_list, problem.y_list, problem.F_list, problem.objectives_list)

    X = np.stack(problem.x_list)
    y = np.array(problem.y_list)
    F = np.stack(problem.F_list)
    objectives = np.stack(problem.objectives_list)


    with open(os.path.join(problem.bug_folder, 'res_'+str(ind)+'.pkl'), 'wb') as f_out:
        pickle.dump({'X':X, 'y':y, 'F':F, 'objectives':objectives, 'n_gen':n_gen, 'pop_size':pop_size, 'hv':hv, 'time_list':problem.time_list, 'bug_num_list':problem.bug_num_list}, f_out)
        print('-'*100, 'pickled')



    if save:
        with open(save_path, 'wb') as f_out:
            pickle.dump(res.algorithm, f_out)
            print('-'*100, 'pickled')


    return X, y, F, objectives

if __name__ == '__main__':
    run_ga()

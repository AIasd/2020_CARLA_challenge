# python ga_fuzzing.py -p 2015 -s 8791 -d 8792 --n_gen 2 --pop_size 2 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 300 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.2 0.5
import sys
import os
sys.path.append('pymoo')
carla_root = '../carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')

sys.path.append('leaderboard')
sys.path.append('leaderboard/team_code')
sys.path.append('scenario_runner')
sys.path.append('carla_project')
sys.path.append('carla_project/src')

sys.path.append('fuzzing_utils')
sys.path.append('carla_specific_utils')
os.system('export PYTHONPATH=/home/zhongzzy9/anaconda3/envs/carla99/bin/python')

sys.path.append('..')

from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.duplicate import ElementwiseDuplicateElimination, NoDuplicateElimination
from pymoo.model.population import Population, pop_from_array_or_individual
from pymoo.model.evaluator import Evaluator
from pymoo.algorithms.nsga2 import NSGA2, binary_tournament
from pymoo.algorithms.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.algorithms.random import RandomAlgorithm
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.performance_indicator.hv import Hypervolume

import matplotlib.pyplot as plt

from object_types import WEATHERS, pedestrian_types, vehicle_types, static_types, vehicle_colors, car_types, motorcycle_types, cyclist_types

from customized_utils import rand_real,  make_hierarchical_dir, exit_handler, is_critical_region, if_violate_constraints, check_bug, filter_critical_regions, encode_fields, remove_fields_not_changing, get_labels_to_encode, customized_fit, customized_standardize, customized_inverse_standardize, decode_fields, encode_bounds, recover_fields_not_changing, process_X, inverse_process_X, determine_y_upon_weights, calculate_rep_d, select_batch_max_d_greedy, if_violate_constraints_vectorized, is_distinct_vectorized, eliminate_repetitive_vectorized, get_sorted_subfolders, load_data, choose_weight_inds, get_F, get_unique_bugs, set_general_seed, emptyobject



from collections import deque


import numpy as np
import carla


from leaderboard.utils.route_parser import RouteParser


import traceback
import json
import re
import time
from datetime import datetime

import pathlib
from distutils.dir_util import copy_tree
import dill as pickle
# import pickle
import argparse
import atexit
import traceback
import math



import copy
from distutils.dir_util import copy_tree

from dask.distributed import Client, LocalCluster

from pymoo.factory import get_termination
from pymoo.model.termination import Termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from pymoo.model.individual import Individual
from pymoo.model.repair import Repair
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_crossover, get_mutation
from pymoo.model.mating import Mating
from pymoo.model.initialization import Initialization
from pymoo.model.duplicate import NoDuplicateElimination
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.model.survival import Survival
from pymoo.model.individual import Individual




from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.stats import rankdata



from pgd_attack import pgd_attack, train_net, train_regression_net, VanillaDataset
from acquisition import map_acquisition








# [ego_linear_speed, min_d, d_angle_norm, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, is_collision]
default_objective_weights = np.array([-1., 1., 1., 1., 1., -1., 0., 0., 0., 0.])
default_objectives = np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.])
default_check_unique_coeff = [0, 0.1, 0.5]


parser = argparse.ArgumentParser()

# general
parser.add_argument("-r", "--route_type", type=str, default='town05_right_0')
parser.add_argument("-c", "--scenario_type", type=str, default='default')
parser.add_argument("-m", "--ego_car_model", type=str, default='lbc')
parser.add_argument('-a','--algorithm_name', type=str, default='nsga2')

parser.add_argument('-p','--ports', nargs='+', type=int, default=[2003], help='TCP port(s) to listen to (default: 2003)')
parser.add_argument("-s", "--scheduler_port", type=int, default=8785)
parser.add_argument("-d", "--dashboard_address", type=int, default=8786)

parser.add_argument('--simulator', type=str, default='carla')

# carla specific
parser.add_argument("--has_display", type=str, default='0')
parser.add_argument("--debug", type=int, default=1, help="whether using the debug mode: planned paths will be visualized.")
parser.add_argument('--correct_spawn_locations_after_run', type=int, default=0)



# logistic
parser.add_argument("--root_folder", type=str, default='run_results')
parser.add_argument("--parent_folder", type=str, default='') # will be automatically created
parser.add_argument("--mean_objectives_across_generations_path", type=str, default='') # will be automatically created
parser.add_argument("--episode_max_time", type=int, default=60)
parser.add_argument('--record_every_n_step', type=int, default=2000)
parser.add_argument('--gpus', type=str, default='0,1')


# algorithm related
parser.add_argument("--n_gen", type=int, default=2)
parser.add_argument("--pop_size", type=int, default=50)
parser.add_argument("--survival_multiplier", type=int, default=1)
parser.add_argument("--n_offsprings", type=int, default=300)
parser.add_argument("--has_run_num", type=int, default=1000)
parser.add_argument('--sample_multiplier', type=int, default=200)
parser.add_argument('--mating_max_iterations', type=int, default=200)
parser.add_argument('--only_run_unique_cases', type=int, default=1)
parser.add_argument('--consider_interested_bugs', type=int, default=1)

parser.add_argument("--outer_iterations", type=int, default=3)
parser.add_argument('--objective_weights', nargs='+', type=float, default=default_objective_weights)
parser.add_argument('--check_unique_coeff', nargs='+', type=float, default=default_check_unique_coeff)
parser.add_argument('--use_single_objective', type=int, default=1)
parser.add_argument('--rank_mode', type=str, default='none')
parser.add_argument('--ranking_model', type=str, default='nn_pytorch')
parser.add_argument('--initial_fit_th', type=int, default=100, help='minimum number of instances needed to train a DNN.')
parser.add_argument('--min_bug_num_to_fit_dnn', type=int, default=10, help='minimum number of bug instances needed to train a DNN.')

parser.add_argument('--pgd_eps', type=float, default=1.01)
parser.add_argument('--adv_conf_th', type=float, default=-4)
parser.add_argument('--attack_stop_conf', type=float, default=0.9)
parser.add_argument('--use_single_nn', type=int, default=1)

parser.add_argument('--warm_up_path', type=str, default=None)
parser.add_argument('--warm_up_len', type=int, default=-1)
parser.add_argument('--regression_nn_use_running_data', type=int, default=1)



parser.add_argument('--uncertainty', type=str, default='')
parser.add_argument('--model_type', type=str, default='one_output')

parser.add_argument('--explore_iter_num', type=int, default=2)
parser.add_argument('--exploit_iter_num', type=int, default=1)
parser.add_argument('--high_conf_num', type=int, default=60)
parser.add_argument('--low_conf_num', type=int, default=60)

parser.add_argument('--use_alternate_nn', type=int, default=0)
parser.add_argument('--diversity_mode', type=str, default='none')
parser.add_argument('--adv_exploitation_only', type=int, default=0)
parser.add_argument('--uncertainty_exploration', type=str, default='confidence')

parser.add_argument('--termination_condition', type=str, default='generations')
parser.add_argument('--max_running_time', type=int, default=3600*24)

parser.add_argument('--emcmc', type=int, default=0)
parser.add_argument('--use_unique_bugs', type=int, default=1)
parser.add_argument('--finish_after_has_run', type=int, default=1)

fuzzing_arguments = parser.parse_args()


os.environ['HAS_DISPLAY'] = fuzzing_arguments.has_display
os.environ['CUDA_VISIBLE_DEVICES'] = fuzzing_arguments.gpus
fuzzing_arguments.objective_weights = np.array(fuzzing_arguments.objective_weights)
# ['BNN', 'one_output']
# BALD and BatchBALD only support BNN
if fuzzing_arguments.uncertainty.split('_')[0] in ['BALD', 'BatchBALD']:
    fuzzing_arguments.model_type = 'BNN'

if 'un' in fuzzing_arguments.algorithm_name:
    fuzzing_arguments.use_unique_bugs = 1
else:
    fuzzing_arguments.use_unique_bugs = 0

if fuzzing_arguments.algorithm_name in ['nsga2-emcmc', 'nsga2-un-emcmc']:
    fuzzing_arguments.emcmc = 1
else:
    fuzzing_arguments.emcmc = 0

# eliminate some randomness
set_general_seed(seed=0)
random_seeds = [10, 20, 30]
rng = np.random.default_rng(random_seeds[0])





class MyProblem(Problem):

    def __init__(self, fuzzing_arguments, sim_specific_arguments, fuzzing_content, run_simulation, dt_arguments):

        self.fuzzing_arguments = fuzzing_arguments
        self.sim_specific_arguments = sim_specific_arguments
        self.fuzzing_content = fuzzing_content
        self.run_simulation = run_simulation
        self.dt_arguments = dt_arguments


        self.ego_car_model = fuzzing_arguments.ego_car_model
        self.scheduler_port = fuzzing_arguments.scheduler_port
        self.dashboard_address = fuzzing_arguments.dashboard_address
        self.ports = fuzzing_arguments.ports
        self.episode_max_time = fuzzing_arguments.episode_max_time
        self.objective_weights = fuzzing_arguments.objective_weights
        self.check_unique_coeff = fuzzing_arguments.check_unique_coeff
        self.consider_interested_bugs = fuzzing_arguments.consider_interested_bugs
        self.record_every_n_step = fuzzing_arguments.record_every_n_step
        self.use_single_objective = fuzzing_arguments.use_single_objective
        self.simulator = fuzzing_arguments.simulator


        self.call_from_dt = dt_arguments.call_from_dt
        self.dt = dt_arguments.dt
        self.estimator = dt_arguments.estimator
        self.critical_unique_leaves = dt_arguments.critical_unique_leaves
        self.cumulative_info = dt_arguments.cumulative_info
        cumulative_info = dt_arguments.cumulative_info

        if cumulative_info:
            self.counter = cumulative_info['counter']
            self.has_run = cumulative_info['has_run']
            self.start_time = cumulative_info['start_time']
            self.time_list = cumulative_info['time_list']
            self.bugs = cumulative_info['bugs']
            self.unique_bugs = cumulative_info['unique_bugs']
            self.interested_unique_bugs = cumulative_info['interested_unique_bugs']
            self.bugs_type_list = cumulative_info['bugs_type_list']
            self.bugs_inds_list = cumulative_info['bugs_inds_list']
            self.bugs_num_list = cumulative_info['bugs_num_list']
            self.unique_bugs_num_list = cumulative_info['unique_bugs_num_list']
            self.has_run_list = cumulative_info['has_run_list']
        else:
            self.counter = 0
            self.has_run = 0
            self.start_time = time.time()
            self.time_list = []
            self.bugs = []
            self.unique_bugs = []
            self.interested_unique_bugs = []
            self.bugs_type_list = []
            self.bugs_inds_list = []
            self.bugs_num_list = []
            self.unique_bugs_num_list = []
            self.has_run_list = []




        self.labels = fuzzing_content.labels
        self.mask = fuzzing_content.mask
        self.parameters_min_bounds = fuzzing_content.parameters_min_bounds
        self.parameters_max_bounds = fuzzing_content.parameters_max_bounds
        self.parameters_distributions = fuzzing_content.parameters_distributions
        self.customized_constraints = fuzzing_content.customized_constraints
        self.customized_center_transforms = fuzzing_content.customized_center_transforms
        xl = [pair[1] for pair in self.parameters_min_bounds.items()]
        xu = [pair[1] for pair in self.parameters_max_bounds.items()]
        n_var = fuzzing_content.n_var



        self.p, self.c, self.th = self.check_unique_coeff
        self.launch_server = True
        self.objectives_list = []
        self.x_list = []
        self.y_list = []
        self.F_list = []



        super().__init__(n_var=n_var, n_obj=4, n_constr=0, xl=xl, xu=xu)



    def _evaluate(self, X, out, *args, **kwargs):
        objective_weights = self.objective_weights
        customized_center_transforms = self.customized_center_transforms

        episode_max_time = self.episode_max_time

        parameters_min_bounds = self.parameters_min_bounds
        parameters_max_bounds = self.parameters_max_bounds
        labels = self.labels
        mask = self.mask
        xl = self.xl
        xu = self.xu
        customized_constraints = self.customized_constraints

        dt = self.dt
        estimator = self.estimator
        critical_unique_leaves = self.critical_unique_leaves


        run_simulation = self.run_simulation
        fuzzing_content = self.fuzzing_content
        sim_specific_arguments = self.sim_specific_arguments
        dt_arguments = self.dt_arguments



        all_final_generated_transforms_list = []



        def fun(x, launch_server, counter, port):
            not_critical_region = dt and not is_critical_region(x, estimator, critical_unique_leaves)
            violate_constraints, _ = if_violate_constraints(x, customized_constraints, labels, verbose=True)
            if not_critical_region or violate_constraints:
                objectives = default_objectives
                return objectives, None, 0
            else:
                objectives, run_info  = run_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port)

                print('\n'*3)
                print("counter, run_info['is_bug'], run_info['bug_type'], objectives", counter, run_info['is_bug'], run_info['bug_type'], objectives)
                print('\n'*3)

                # correct_travel_dist(x, labels, customized_data['tmp_travel_dist_file'])


                return objectives, run_info, 1




        def submit_and_run_jobs(ind_start, ind_end, launch_server, job_results):
            time_elapsed = 0
            jobs = []
            for i in range(ind_start, ind_end):
                j = i % len(self.ports)
                port = self.ports[j]
                worker = workers[j]
                x = X[i]
                jobs.append(client.submit(fun, x, launch_server, self.counter, port, workers=worker))

                print('submit job', i, self.counter)
                self.counter += 1


            for i in range(len(jobs)):
                job = jobs[i]
                cur_i = i + ind_start
                total_i = i + (self.counter-len(jobs))
                objectives, run_info, has_run  = job.result()
                print('get job result for', i)
                if run_info and 'all_final_generated_transforms' in run_info:
                    all_final_generated_transforms_list.append(run_info['all_final_generated_transforms'])
                else:
                    all_final_generated_transforms_list.append(None)

                self.has_run_list.append(has_run)
                self.has_run += has_run


                # record bug
                if run_info and run_info['is_bug']:
                    self.bugs.append(X[cur_i].astype(float))
                    self.bugs_inds_list.append(total_i)
                    self.bugs_type_list.append(run_info['bug_type'])

                    self.y_list.append(run_info['bug_type'])
                else:
                    self.y_list.append(0)


                self.x_list.append(X[cur_i])
                self.objectives_list.append(np.array(objectives))
                job_results.append(np.array(objectives))


            # hack:
            with open('tmp_folder/total.pickle', 'wb') as f_out:
                pickle.dump(all_final_generated_transforms_list, f_out)


            # record time elapsed and bug numbers
            self.time_list.append(time_elapsed)





        with LocalCluster(scheduler_port=self.scheduler_port, dashboard_address=self.dashboard_address, n_workers=len(self.ports), threads_per_worker=1) as cluster, Client(cluster) as client:
            job_results = []
            workers = []
            for k in client.has_what():
                workers.append(k[len('tcp://'):])

            end_ind = np.min([len(self.ports), X.shape[0]])
            rng = np.random.default_rng(random_seeds[1])
            submit_and_run_jobs(0, end_ind, self.launch_server, job_results)
            self.launch_server = False
            time_elapsed = time.time() - self.start_time

            if X.shape[0] > len(self.ports):
                rng = np.random.default_rng(random_seeds[2])
                submit_and_run_jobs(end_ind, X.shape[0], self.launch_server, job_results)

            current_F = get_F(job_results, self.objectives_list, objective_weights, self.use_single_objective)

            out["F"] = current_F
            self.F_list.append(current_F)



            print('\n'*10, '+'*100)



            bugs_type_list_tmp = self.bugs_type_list
            bugs_tmp = self.bugs
            bugs_inds_list_tmp = self.bugs_inds_list

            self.unique_bugs, unique_bugs_inds_list, self.interested_unique_bugs, bugcounts = get_unique_bugs(self.x_list, self.objectives_list, self.mask, self.xl, self.xu, self.check_unique_coeff, objective_weights, return_mode='unique_inds_and_interested_and_bugcounts', consider_interested_bugs=1, bugs_type_list=bugs_type_list_tmp, bugs=bugs_tmp, bugs_inds_list=bugs_inds_list_tmp)


            time_elapsed = time.time() - self.start_time
            num_of_bugs = len(self.bugs)
            num_of_unique_bugs = len(self.unique_bugs)
            num_of_interested_unique_bugs = len(self.interested_unique_bugs)

            self.bugs_num_list.append(num_of_bugs)
            self.unique_bugs_num_list.append(num_of_unique_bugs)
            mean_objectives_this_generation = np.mean(np.array(self.objectives_list[-X.shape[0]:]), axis=0)

            with open(self.fuzzing_arguments.mean_objectives_across_generations_path, 'a') as f_out:

                combined_list = [self.counter, self.has_run, time_elapsed, num_of_bugs, num_of_unique_bugs, num_of_interested_unique_bugs
                ]+['bugcounts']+bugcounts+['mean_objectives_this_generation']+mean_objectives_this_generation.tolist()+['\n']
                info_str = ','.join([str(x) for x in combined_list])
                f_out.write(info_str)
                f_out.write(';'.join([str(ind) for ind in unique_bugs_inds_list])+' objective_weights : '+str(self.objective_weights)+'\n')
            print(info_str)
            print('+'*100, '\n'*10)















# class MySampling(Sampling):
#     '''
#     dimension correspondence
#
#     Define:
#     n1=problem.waypoints_num_limit
#     n2=problem.num_of_static_max
#     n3=problem.num_of_pedestrians_max
#     n4=problem.num_of_vehicles_max
#
#     global
#     0: friction, real, [0, 1].
#     1: weather_index, int, [0, problem.num_of_weathers].
#     2: num_of_static, int, [0, n2].
#     3: num_of_pedestrians, int, [0, n3].
#     4: num_of_vehicles, int, [0, n4].
#
#     ego-car
#     5 ~ 4+n1*2: waypoints perturbation [(dx_i, dy_i)] with length n1.
#     dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].
#
#
#     static
#     5+n1*2 ~ 4+n1*2+n2*4: [(static_type_i, x w.r.t. center, y w.r.t. center, yaw)] with length n2.
#     static_type_i, int, [0, problem.num_of_static_types).
#     x_i, real, [problem.static_x_min, problem.static_x_max].
#     y_i, real, [problem.static_y_min, problem.static_y_max].
#     yaw_i, real, [problem.yaw_min, problem.yaw_max).
#
#     pedestrians
#     5+n1*2+n2*4 ~ 4+n1*2+n2*4+n3*7: [(pedestrian_type_i, x_i, y_i, yaw_i, trigger_distance_i, speed_i, dist_to_travel_i)] with length n3.
#     pedestrian_type_i, int, [0, problem.num_of_static_types)
#     x_i, real, [problem.pedestrian_x_min, problem.pedestrian_x_max].
#     y_i, real, [problem.pedestrian_y_min, problem.pedestrian_y_max].
#     yaw_i, real, [problem.yaw_min, problem.yaw_max).
#     trigger_distance_i, real, [problem.pedestrian_trigger_distance_min, problem.pedestrian_trigger_distance_max].
#     speed_i, real, [problem.pedestrian_speed_min, problem.pedestrian_speed_max].
#     dist_to_travel_i, real, [problem.pedestrian_dist_to_travel_min, problem.pedestrian_dist_to_travel_max].
#
#     vehicles
#     5+n1*2+n2*4+n3*7 ~ 4+n1*2+n2*4+n3*7+n4*(14+n1*2): [(vehicle_type_i, x_i, y_i, yaw_i, initial_speed_i, trigger_distance_i, targeted_speed_i, waypoint_follower_i, targeted_x_i, targeted_y_i, avoid_collision_i, dist_to_travel_i, target_yaw_i, color_i, [(dx_i, dy_i)] with length n1)] with length n4.
#     vehicle_type_i, int, [0, problem.num_of_vehicle_types)
#     x_i, real, [problem.vehicle_x_min, problem.vehicle_x_max].
#     y_i, real, [problem.vehicle_y_min, problem.vehicle_y_max].
#     yaw_i, real, [problem.yaw_min, problem.yaw_max).
#     initial_speed_i, real, [problem.vehicle_initial_speed_min, problem.vehicle_initial_speed_max].
#     trigger_distance_i, real, [problem.vehicle_trigger_distance_min, problem.vehicle_trigger_distance_max].
#     targeted_speed_i, real, [problem.vehicle_targeted_speed_min, problem.vehicle_targeted_speed_max].
#     waypoint_follower_i, boolean, [0, 1]
#     targeted_x_i, real, [problem.targeted_x_min, problem.targeted_x_max].
#     targeted_y_i, real, [problem.targeted_y_min, problem.targeted_y_max].
#     avoid_collision_i, boolean, [0, 1]
#     dist_to_travel_i, real, [problem.vehicle_dist_to_travel_min, problem.vehicle_dist_to_travel_max].
#     target_yaw_i, real, [problem.yaw_min, problem.yaw_max).
#     color_i, int, [0, problem.num_of_vehicle_colors).
#     dx_i, dy_i, real, ~ [problem.perturbation_min, problem.perturbation_max].
#
#
#     '''
#     def __init__(self, use_unique_bugs, check_unique_coeff, sample_multiplier=500):
#         self.use_unique_bugs = use_unique_bugs
#         self.check_unique_coeff = check_unique_coeff
#         self.sample_multiplier = sample_multiplier
#         assert len(self.check_unique_coeff) == 3
#     def _do(self, problem, n_samples, **kwargs):
#         p, c, th = self.check_unique_coeff
#         xl = problem.xl
#         xu = problem.xu
#         mask = np.array(problem.mask)
#         labels = problem.labels
#         parameters_distributions = problem.parameters_distributions
#         max_sample_times = n_samples * self.sample_multiplier
#
#         algorithm = kwargs['algorithm']
#
#         tmp_off = algorithm.tmp_off
#
#         # print(tmp_off)
#         tmp_off_and_X = []
#         if len(tmp_off) > 0:
#             tmp_off = [off.X for off in tmp_off]
#             tmp_off_and_X = tmp_off
#         # print(tmp_off)
#
#
#         def subroutine(X, tmp_off_and_X):
#             def sample_one_feature(typ, lower, upper, dist, label):
#                 assert lower <= upper, label+','+str(lower)+'>'+str(upper)
#                 if typ == 'int':
#                     val = rng.integers(lower, upper+1)
#                 elif typ == 'real':
#                     if dist[0] == 'normal':
#                         if dist[1] == None:
#                             mean = (lower+upper)/2
#                         else:
#                             mean = dist[1]
#                         val = rng.normal(mean, dist[2], 1)[0]
#                     else: # default is uniform
#                         val = rand_real(rng, lower, upper)
#                     val = np.clip(val, lower, upper)
#                 return val
#
#             sample_time = 0
#             while sample_time < max_sample_times and len(X) < n_samples:
#                 sample_time += 1
#                 x = []
#                 for i, dist in enumerate(parameters_distributions):
#                     typ = mask[i]
#                     lower = xl[i]
#                     upper = xu[i]
#                     label = labels[i]
#                     val = sample_one_feature(typ, lower, upper, dist, label)
#                     x.append(val)
#
#
#                 if not if_violate_constraints(x, problem.customized_constraints, problem.labels)[0]:
#                     if not self.use_unique_bugs or (is_distinct(x, tmp_off_and_X, mask, xl, xu, p, c, th) and is_distinct(x, problem.interested_unique_bugs, mask, xl, xu, p, c, th)):
#                         x = np.array(x).astype(float)
#                         X.append(x)
#                         if len(tmp_off) > 0:
#                             tmp_off_and_X = tmp_off + X
#                         else:
#                             tmp_off_and_X = X
#                         # if self.use_unique_bugs:
#                         #     if disable_unique_x_for_X:
#                         #         X = eliminate_duplicates_for_list(mask, xl, xu, p, c, th, X, problem.unique_bugs)
#                         #     else:
#                         #         X = eliminate_duplicates_for_list(mask, xl, xu, p, c, th, X, problem.unique_bugs, tmp_off=tmp_off)
#
#             return X, sample_time
#
#
#         X = []
#         X, sample_time_1 = subroutine(X, tmp_off_and_X)
#
#         if len(X) > 0:
#             X = np.stack(X)
#         else:
#             X = np.array([])
#         print('\n'*3, 'We sampled', X.shape[0], '/', n_samples, 'samples', 'by sampling', sample_time_1, 'times' '\n'*3)
#
#         return X

class MySamplingVectorized(Sampling):

    def __init__(self, use_unique_bugs, check_unique_coeff, sample_multiplier=500):
        self.use_unique_bugs = use_unique_bugs
        self.check_unique_coeff = check_unique_coeff
        self.sample_multiplier = sample_multiplier
        assert len(self.check_unique_coeff) == 3
    def _do(self, problem, n_samples, **kwargs):
        p, c, th = self.check_unique_coeff
        xl = problem.xl
        xu = problem.xu
        mask = np.array(problem.mask)
        labels = problem.labels
        parameters_distributions = problem.parameters_distributions


        if self.sample_multiplier >= 50:
            max_sample_times = self.sample_multiplier // 50
            n_samples_sampling = n_samples * 50
        else:
            max_sample_times = self.sample_multiplier
            n_samples_sampling = n_samples

        algorithm = kwargs['algorithm']

        tmp_off = algorithm.tmp_off

        # print(tmp_off)
        tmp_off_and_X = []
        if len(tmp_off) > 0:
            tmp_off = [off.X for off in tmp_off]
            tmp_off_and_X = tmp_off
        # print(tmp_off)


        def subroutine(X, tmp_off_and_X):
            def sample_one_feature(typ, lower, upper, dist, label, size=1):
                assert lower <= upper, label+','+str(lower)+'>'+str(upper)
                if typ == 'int':
                    val = rng.integers(lower, upper+1, size=size)
                elif typ == 'real':
                    if dist[0] == 'normal':
                        if dist[1] == None:
                            mean = (lower+upper)/2
                        else:
                            mean = dist[1]
                        val = rng.normal(mean, dist[2], size=size)
                    else: # default is uniform
                        val = rng.random(size=size) * (upper - lower) + lower
                    val = np.clip(val, lower, upper)
                return val

            # TBD: temporary
            sample_time = 0
            while sample_time < max_sample_times and len(X) < n_samples:
                print('sample_time / max_sample_times', sample_time, '/', max_sample_times, 'len(X)', len(X))
                sample_time += 1
                cur_X = []
                for i, dist in enumerate(parameters_distributions):
                    typ = mask[i]
                    lower = xl[i]
                    upper = xu[i]
                    label = labels[i]
                    val = sample_one_feature(typ, lower, upper, dist, label, size=n_samples_sampling)
                    cur_X.append(val)
                cur_X = np.swapaxes(np.stack(cur_X),0,1)


                remaining_inds = if_violate_constraints_vectorized(cur_X, problem.customized_constraints, problem.labels, verbose=False)
                if len(remaining_inds) == 0:
                    continue

                cur_X = cur_X[remaining_inds]

                if not self.use_unique_bugs:
                    X.extend(cur_X)
                    if len(X) > n_samples:
                        X = X[:n_samples]
                else:
                    if len(tmp_off_and_X) > 0 and len(problem.interested_unique_bugs) > 0:
                        prev_X = np.concatenate([problem.interested_unique_bugs, tmp_off_and_X])
                    elif len(tmp_off_and_X) > 0:
                        prev_X = tmp_off_and_X
                    else:
                        prev_X = problem.interested_unique_bugs
                    # print('prev_X.shape', prev_X.shape)
                    remaining_inds = is_distinct_vectorized(cur_X, prev_X, mask, xl, xu, p, c, th, verbose=False)

                    if len(remaining_inds) == 0:
                        continue
                    else:
                        cur_X = cur_X[remaining_inds]
                        X.extend(cur_X)
                        if len(X) > n_samples:
                            X = X[:n_samples]
                        if len(tmp_off) > 0:
                            tmp_off_and_X = tmp_off + X
                        else:
                            tmp_off_and_X = X
            return X, sample_time


        X = []
        X, sample_time_1 = subroutine(X, tmp_off_and_X)

        if len(X) > 0:
            X = np.stack(X)
        else:
            X = np.array([])
        print('\n'*3, 'We sampled', X.shape[0], '/', n_samples, 'samples', 'by sampling', sample_time_1, 'times' '\n'*3)

        return X



def do_emcmc(parents, off, n_gen, objective_weights):
    base_val = np.sum(np.array(default_objectives[:len(objective_weights)])*np.array(objective_weights))
    filtered_off = []
    F_list = []
    for i in off:
        for p in parents:
            print(i.F, p.F)

            i_val = np.sum(np.array(i.F) * np.array(objective_weights))
            p_val = np.sum(np.array(p.F) * np.array(objective_weights))

            print('1', base_val, i_val, p_val)
            i_val = np.abs(base_val-i_val)
            p_val = np.abs(base_val-p_val)
            prob = np.min([i_val / p_val, 1])
            print('2', base_val, i_val, p_val, prob)

            if np.random.uniform() < prob:
                filtered_off.append(i.X)
                F_list.append(i.F)

    pop = Population(len(filtered_off), individual=Individual())
    pop.set("X", filtered_off, "F", F_list, "n_gen", n_gen, "CV", [0 for _ in range(len(filtered_off))], "feasible", [[True] for _ in range(len(filtered_off))])

    return Population.merge(parents, off)


# class MyMating(Mating):
#     def __init__(self,
#                  selection,
#                  crossover,
#                  mutation,
#                  use_unique_bugs,
#                  emcmc,
#                  **kwargs):
#
#         super().__init__(selection, crossover, mutation, **kwargs)
#         self.use_unique_bugs = use_unique_bugs
#         self.mating_max_iterations = mating_max_iterations
#         self.emcmc = emcmc
#
#     def do(self, problem, pop, n_offsprings, **kwargs):
#
#         # the population object to be used
#         off = pop.new()
#         parents = pop.new()
#
#         # infill counter - counts how often the mating needs to be done to fill up n_offsprings
#         n_infills = 0
#
#         # iterate until enough offsprings are created
#         while len(off) < n_offsprings:
#             # how many offsprings are remaining to be created
#             n_remaining = n_offsprings - len(off)
#
#             # do the mating
#             _off, _parents = self._do(problem, pop, n_remaining, **kwargs)
#
#
#             # repair the individuals if necessary - disabled if repair is NoRepair
#             _off_first = self.repair.do(problem, _off, **kwargs)
#
#             # Previous
#             _off = []
#             for x in _off_first:
#                 if not if_violate_constraints(x.X, problem.customized_constraints, problem.labels)[0]:
#                     _off.append(x.X)
#
#             _off = pop.new("X", _off)
#
#             # Previous
#             # eliminate the duplicates - disabled if it is NoRepair
#             if self.use_unique_bugs and len(_off) > 0:
#                 _off, no_duplicate, _ = self.eliminate_duplicates.do(_off, problem.unique_bugs, off, return_indices=True, to_itself=True)
#                 _parents = _parents[no_duplicate]
#                 assert len(_parents)==len(_off)
#
#
#
#
#             # if more offsprings than necessary - truncate them randomly
#             if len(off) + len(_off) > n_offsprings:
#                 # IMPORTANT: Interestingly, this makes a difference in performance
#                 n_remaining = n_offsprings - len(off)
#                 _off = _off[:n_remaining]
#                 _parents = _parents[:n_remaining]
#
#
#             # add to the offsprings and increase the mating counter
#             off = Population.merge(off, _off)
#             parents = Population.merge(parents, _parents)
#             n_infills += 1
#
#             # if no new offsprings can be generated within a pre-specified number of generations
#             if n_infills > self.mating_max_iterations:
#                 break
#
#         # assert len(parents)==len(off)
#         print('Mating finds', len(off), 'offsprings after doing', n_infills-1, '/', self.mating_max_iterations, 'mating iterations')
#         return off, parents
#
#
#
#     # only to get parents
#     def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
#
#         # if the parents for the mating are not provided directly - usually selection will be used
#         if parents is None:
#             # how many parents need to be select for the mating - depending on number of offsprings remaining
#             n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)
#             # select the parents for the mating - just an index array
#             parents = self.selection.do(pop, n_select, self.crossover.n_parents, **kwargs)
#             parents_obj = pop[parents].reshape([-1, 1]).squeeze()
#         else:
#             parents_obj = parents
#
#
#         # do the crossover using the parents index and the population - additional data provided if necessary
#         _off = self.crossover.do(problem, pop, parents, **kwargs)
#         # do the mutation on the offsprings created through crossover
#         _off = self.mutation.do(problem, _off, **kwargs)
#
#         return _off, parents_obj

class MyMatingVectorized(Mating):
    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 use_unique_bugs,
                 emcmc,
                 mating_max_iterations,
                 **kwargs):

        super().__init__(selection, crossover, mutation, **kwargs)
        self.use_unique_bugs = use_unique_bugs
        self.mating_max_iterations = mating_max_iterations
        self.emcmc = emcmc


    def do(self, problem, pop, n_offsprings, **kwargs):

        if self.mating_max_iterations >= 5:
            mating_max_iterations = self.mating_max_iterations // 5
            n_offsprings_sampling = n_offsprings * 5
        else:
            mating_max_iterations = self.mating_max_iterations
            n_offsprings_sampling = n_offsprings

        # the population object to be used
        off = pop.new()
        parents = pop.new()

        # infill counter - counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0

        # iterate until enough offsprings are created
        while len(off) < n_offsprings:
            n_infills += 1
            print('n_infills / mating_max_iterations', n_infills, '/', mating_max_iterations, 'len(off)', len(off))
            # if no new offsprings can be generated within a pre-specified number of generations
            if n_infills >= mating_max_iterations:
                break

            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off, _parents = self._do(problem, pop, n_offsprings_sampling, **kwargs)


            # repair the individuals if necessary - disabled if repair is NoRepair
            _off_first = self.repair.do(problem, _off, **kwargs)


            # Vectorized
            _off_X = np.array([x.X for x in _off_first])
            remaining_inds = if_violate_constraints_vectorized(_off_X, problem.customized_constraints, problem.labels, verbose=False)
            _off_X = _off_X[remaining_inds]

            _off = _off_first[remaining_inds]
            _parents = _parents[remaining_inds]

            # Vectorized
            if self.use_unique_bugs:
                if len(_off) == 0:
                    continue
                elif len(off) > 0 and len(problem.interested_unique_bugs) > 0:
                    prev_X = np.concatenate([problem.interested_unique_bugs, np.array([x.X for x in off])])
                elif len(off) > 0:
                    prev_X = np.array([x.X for x in off])
                else:
                    prev_X = problem.interested_unique_bugs


                remaining_inds = is_distinct_vectorized(_off_X, prev_X, problem.mask, problem.xl, problem.xu, problem.p, problem.c, problem.th, verbose=False)

                if len(remaining_inds) == 0:
                    continue

                _off = _off[remaining_inds]
                _parents = _parents[remaining_inds]
                assert len(_parents)==len(_off)



            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:
                # IMPORTANT: Interestingly, this makes a difference in performance
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]
                _parents = _parents[:n_remaining]


            # add to the offsprings and increase the mating counter
            off = Population.merge(off, _off)
            parents = Population.merge(parents, _parents)




        # assert len(parents)==len(off)
        print('Mating finds', len(off), 'offsprings after doing', n_infills, '/', mating_max_iterations, 'mating iterations')
        return off, parents



    # only to get parents
    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:
            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)
            # select the parents for the mating - just an index array
            parents = self.selection.do(pop, n_select, self.crossover.n_parents, **kwargs)
            parents_obj = pop[parents].reshape([-1, 1]).squeeze()
        else:
            parents_obj = parents


        # do the crossover using the parents index and the population - additional data provided if necessary
        _off = self.crossover.do(problem, pop, parents, **kwargs)
        # do the mutation on the offsprings created through crossover
        _off = self.mutation.do(problem, _off, **kwargs)

        return _off, parents_obj


class NSGA2_DT(NSGA2):
    def __init__(self, dt=False, X=None, F=None, fuzzing_arguments=None, plain_sampling=None, **kwargs):
        self.dt = dt
        self.X = X
        self.F = F
        self.plain_sampling = plain_sampling

        self.sampling = kwargs['sampling']
        self.pop_size = fuzzing_arguments.pop_size
        self.n_offsprings = fuzzing_arguments.n_offsprings

        self.survival_multiplier = fuzzing_arguments.survival_multiplier
        self.algorithm_name = fuzzing_arguments.algorithm_name
        self.emcmc = fuzzing_arguments.emcmc
        self.initial_fit_th = fuzzing_arguments.initial_fit_th
        self.rank_mode = fuzzing_arguments.rank_mode
        self.min_bug_num_to_fit_dnn = fuzzing_arguments.min_bug_num_to_fit_dnn
        self.ranking_model = fuzzing_arguments.ranking_model
        self.use_unique_bugs = fuzzing_arguments.use_unique_bugs
        self.pgd_eps = fuzzing_arguments.pgd_eps
        self.adv_conf_th = fuzzing_arguments.adv_conf_th
        self.attack_stop_conf = fuzzing_arguments.attack_stop_conf
        self.use_single_nn = fuzzing_arguments.use_single_nn
        self.uncertainty = fuzzing_arguments.uncertainty
        self.model_type = fuzzing_arguments.model_type
        self.explore_iter_num = fuzzing_arguments.explore_iter_num
        self.exploit_iter_num = fuzzing_arguments.exploit_iter_num
        self.high_conf_num = fuzzing_arguments.high_conf_num
        self.low_conf_num = fuzzing_arguments.low_conf_num
        self.warm_up_path = fuzzing_arguments.warm_up_path
        self.warm_up_len = fuzzing_arguments.warm_up_len
        self.use_alternate_nn = fuzzing_arguments.use_alternate_nn
        self.diversity_mode = fuzzing_arguments.diversity_mode
        self.regression_nn_use_running_data = fuzzing_arguments.regression_nn_use_running_data
        self.adv_exploitation_only = fuzzing_arguments.adv_exploitation_only
        self.uncertainty_exploration = fuzzing_arguments.uncertainty_exploration
        self.only_run_unique_cases = fuzzing_arguments.only_run_unique_cases


        super().__init__(pop_size=self.pop_size, n_offsprings=self.n_offsprings, **kwargs)

        self.plain_initialization = Initialization(self.plain_sampling, individual=Individual(), repair=self.repair, eliminate_duplicates= NoDuplicateElimination())


        # heuristic: we keep up about 2 times of each generation's population
        self.survival_size = self.pop_size * self.survival_multiplier

        self.all_pop_run_X = []

        # hack: defined separately w.r.t. MyMating
        self.mating_max_iterations = 1

        self.tmp_off = []
        self.tmp_off_type_1_len = 0
        # self.tmp_off_type_1and2_len = 0

        self.high_conf_configs_stack = []
        self.high_conf_configs_ori_stack = []



    def set_off(self):
        self.tmp_off = []
        if self.algorithm_name == 'random':
            self.tmp_off = self.plain_initialization.do(self.problem, self.n_offsprings, algorithm=self)
        else:
            if self.algorithm_name == 'random-un':
                self.tmp_off, parents = [], []

            else:
                print('len(self.pop)', len(self.pop))
                # do the mating using the current population
                if len(self.pop) > 0:
                    self.tmp_off, parents = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

            print('\n'*3, 'after mating len 0', len(self.tmp_off), 'self.n_offsprings', self.n_offsprings, '\n'*3)


            if len(self.tmp_off) < self.n_offsprings:
                remaining_num = self.n_offsprings - len(self.tmp_off)
                remaining_off = self.initialization.do(self.problem, remaining_num, algorithm=self)
                remaining_parrents = remaining_off
                if len(self.tmp_off) == 0:
                    self.tmp_off = remaining_off
                    parents = remaining_parrents
                else:
                    self.tmp_off = Population.merge(self.tmp_off, remaining_off)
                    parents = Population.merge(parents, remaining_parrents)

                print('\n'*3, 'unique after random generation len 1', len(self.tmp_off), '\n'*3)

            self.tmp_off_type_1_len = len(self.tmp_off)

            if len(self.tmp_off) < self.n_offsprings:
                remaining_num = self.n_offsprings - len(self.tmp_off)
                remaining_off = self.plain_initialization.do(self.problem, remaining_num, algorithm=self)
                remaining_parrents = remaining_off

                self.tmp_off = Population.merge(self.tmp_off, remaining_off)
                parents = Population.merge(parents, remaining_parrents)

                print('\n'*3, 'random generation len 2', len(self.tmp_off), '\n'*3)




        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.tmp_off) == 0 or (not self.problem.call_from_dt and self.problem.fuzzing_arguments.finish_after_has_run and self.problem.has_run >= self.problem.fuzzing_arguments.has_run_num):
            self.termination.force_termination = True
            print("Mating cannot generate new springs, terminate earlier.")
            print('self.tmp_off', len(self.tmp_off), self.tmp_off)
            return

        # if not the desired number of offspring could be created
        elif len(self.tmp_off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")


        if len(self.all_pop_run_X) > 0:
            print('self.all_pop_run_X.shape', self.all_pop_run_X.shape)

        # additional step to rank and select self.off after gathering initial population
        if self.rank_mode != 'none':
            # print(self.rank_mode in ['nn', 'adv_nn'])
            # print(len(self.problem.objectives_list), self.initial_fit_th)
            # print(np.sum(determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)), self.min_bug_num_to_fit_dnn)
            if (self.rank_mode in ['nn', 'adv_nn', 'alternate_nn'] and len(self.problem.objectives_list) >= self.initial_fit_th and  np.sum(determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)) >= self.min_bug_num_to_fit_dnn) or (self.rank_mode in ['regression_nn'] and len(self.problem.objectives_list) >= self.pop_size):
                if self.rank_mode in ['regression_nn']:
                    # only consider collision case for now
                    from customized_utils import pretrain_regression_nets

                    if self.regression_nn_use_running_data:
                        initial_X = self.all_pop_run_X
                        initial_objectives_list = self.problem.objectives_list
                        cutoff = len(initial_X)
                        cutoff_end = cutoff
                    else:
                        subfolders = get_sorted_subfolders(self.warm_up_path)
                        initial_X, _, initial_objectives_list, _, _, _ = load_data(subfolders)

                        cutoff = self.warm_up_len
                        cutoff_end = self.warm_up_len + 100

                        if cutoff == 0:
                            cutoff = len(initial_X)
                        if cutoff_end > len(initial_X):
                            cutoff_end = len(initial_X)


                    # print('self.problem.labels', self.problem.labels)
                    clfs, confs, chosen_weights, standardize_prev = pretrain_regression_nets(initial_X, initial_objectives_list, self.problem.objective_weights, self.problem.xl, self.problem.xu, self.problem.labels, self.problem.customized_constraints, cutoff, cutoff_end)
                else:
                    standardize_prev = None

                X_train_ori = self.all_pop_run_X
                print('X_train_ori.shape[0]', X_train_ori.shape[0])
                X_test_ori = self.tmp_off.get("X")
                # print(np.array(X_train_ori).shape, np.array(X_test_ori).shape)

                initial_X = np.concatenate([X_train_ori, X_test_ori])
                cutoff = X_train_ori.shape[0]
                cutoff_end = initial_X.shape[0]
                partial = True
                # print('initial_X.shape', np.array(initial_X).shape, cutoff, cutoff_end)
                # print('len(self.problem.labels)', len(self.problem.labels))
                X_train, X_test, xl, xu, labels_used, standardize, one_hot_fields_len, param_for_recover_and_decode = process_X(initial_X, self.problem.labels, self.problem.xl, self.problem.xu, cutoff, cutoff_end, partial, len(self.problem.interested_unique_bugs), standardize_prev=standardize_prev)
                # print('X_train.shape[0]', X_train.shape[0])
                # print('labels_used', labels_used)
                # print('process_X X_train.shape, X_test.shape', X_train.shape, X_test.shape)
                (X_removed, kept_fields, removed_fields, enc, inds_to_encode, inds_non_encode, encoded_fields, _, _, unique_bugs_len) = param_for_recover_and_decode

                print('process_X finished')
                if self.rank_mode in ['regression_nn']:
                    # only consider collision case for now

                    # print('X_test.shape', X_test.shape)
                    weight_inds = choose_weight_inds(self.problem.objective_weights)
                    obj_preds = []
                    for clf in clfs:
                        obj_preds.append(clf.predict(X_test))

                    tmp_objectives = np.concatenate(obj_preds, axis=1)
                    # print('tmp_objectives', tmp_objectives)
                    # when using unique bugs give preference to unique inputs

                    if self.use_unique_bugs:
                        tmp_objectives[:self.tmp_off_type_1_len] -= 100*chosen_weights

                    # print(len(tmp_objectives), self.tmp_off_type_1_len)
                    # print('tmp_objectives after use_unique_bugs', tmp_objectives)

                    tmp_objectives_minus = tmp_objectives - confs
                    tmp_objectives_plus = tmp_objectives + confs


                    tmp_pop_minus = Population(X_train.shape[0]+X_test.shape[0], individual=Individual())
                    # print(X_train.shape)
                    # print(X_test.shape)
                    tmp_X_minus = np.concatenate([X_train, X_test])

                    # print(np.array(self.problem.objectives_list)[:, :3])
                    # print(tmp_objectives)
                    # print(np.array(default_objective_weights[:3]))
                    tmp_objectives_minus = np.concatenate([np.array(self.problem.objectives_list)[:, weight_inds], tmp_objectives_minus]) * np.array(default_objective_weights[weight_inds])

                    tmp_pop_minus.set("X", tmp_X_minus)
                    tmp_pop_minus.set("F", tmp_objectives_minus)

                    print('len(tmp_objectives_minus)', len(tmp_objectives_minus))
                    inds_minus_top = np.array(self.survival.do(self.problem, tmp_pop_minus, self.pop_size, return_indices=True))
                    print('inds_minus_top', inds_minus_top, 'len(X_train)', len(X_train), np.sum(inds_minus_top<len(X_train)))
                    # print('inds_minus_top', inds_minus_top)
                    num_of_top_already_run = np.sum(inds_minus_top<len(X_train))
                    num_to_run = self.pop_size - num_of_top_already_run

                    if num_to_run > 0:
                        tmp_pop_plus = Population(X_test.shape[0], individual=Individual())

                        tmp_X_plus = X_test
                        tmp_objectives_plus = tmp_objectives_plus * np.array(default_objective_weights[weight_inds])

                        tmp_pop_plus.set("X", tmp_X_plus)
                        tmp_pop_plus.set("F", tmp_objectives_plus)

                        print('tmp_objectives_plus', tmp_objectives_plus)
                        inds_plus_top = np.array(self.survival.do(self.problem, tmp_pop_plus, num_to_run, return_indices=True))

                        print('inds_plus_top', inds_plus_top)
                        self.off = self.tmp_off[inds_plus_top]
                    else:
                        print('no more offsprings to run (regression nn)')
                        self.off = Population(0, individual=Individual())
                else:

                    if self.uncertainty:

                        # [None, 'BUGCONF', 'Random', 'BALD', 'BatchBALD']
                        print('uncertainty', self.uncertainty)
                        uncertainty_key, uncertainty_conf = self.uncertainty.split('_')

                        acquisition_strategy = map_acquisition(uncertainty_key)
                        acquirer = acquisition_strategy(self.pop_size)



                        if uncertainty_conf == 'conf':
                            uncertainty_conf = True
                        else:
                            uncertainty_conf = False

                        pool_data = VanillaDataset(X_test, np.zeros(X_test.shape[0]), to_tensor=True)
                        pool_data = torch.utils.data.Subset(pool_data, np.arange(len(pool_data)))

                        y_train = determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)
                        clf = train_net(X_train, y_train, [], [], batch_train=60, model_type=self.model_type)

                        if self.use_unique_bugs:
                            unique_len = self.tmp_off_type_1_len
                        else:
                            unique_len = 0
                        inds = acquirer.select_batch(clf, pool_data, unique_len=unique_len, uncertainty_conf=uncertainty_conf)
                        print('chosen indices', inds)
                    else:
                        one_clf = True
                        adv_conf_th = self.adv_conf_th
                        attack_stop_conf = self.attack_stop_conf

                        print('self.use_single_nn', self.use_single_nn)
                        if self.use_single_nn:
                            y_train = determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)
                            print('self.ranking_model', self.ranking_model)
                            if self.ranking_model == 'nn_sklearn':
                                clf = MLPClassifier(solver='lbfgs', activation='tanh', max_iter=10000)
                                clf.fit(X_train, y_train)
                            elif self.ranking_model == 'nn_pytorch':
                                print(X_train.shape, y_train.shape)
                                clf = train_net(X_train, y_train, [], [], batch_train=200)
                            elif self.ranking_model == 'adaboost':
                                from sklearn.ensemble import AdaBoostClassifier
                                clf = AdaBoostClassifier()
                                clf = clf.fit(X_train, y_train)
                            elif self.ranking_model == 'regression':
                                # only support collision for now
                                y_train = np.array([obj[1] for obj in self.problem.objectives_list])
                                # print('y_train', y_train)
                                clf = train_net(X_train, y_train, [], [], batch_train=64, model_type=ranking_model)
                            else:
                                raise ValueError('invalid ranking model', ranking_model)
                            print('X_train', X_train.shape)
                            print('clf.predict_proba(X_train)', clf.predict_proba(X_train).shape)
                            if self.ranking_model == 'adaboost':
                                prob_train = clf.predict_proba(X_train)[:, 0].squeeze()
                            else:
                                prob_train = clf.predict_proba(X_train)[:, 1].squeeze()
                            cur_y = y_train

                            if self.adv_conf_th < 0 and self.rank_mode in ['adv_nn']:
                                print(sorted(prob_train, reverse=True))
                                print('cur_y', cur_y)
                                print('np.abs(self.adv_conf_th)', np.abs(self.adv_conf_th))
                                print(int(np.sum(cur_y)//np.abs(self.adv_conf_th)))
                                adv_conf_th = sorted(prob_train, reverse=True)[int(np.sum(cur_y)//np.abs(self.adv_conf_th))]
                                attack_stop_conf = np.max([self.attack_stop_conf, adv_conf_th])
                            if self.adv_conf_th > attack_stop_conf:
                                self.adv_conf_th = attack_stop_conf

                        else:
                            from customized_utils import get_all_y

                            y_list = get_all_y(self.problem.objectives_list, self.problem.objective_weights)
                            clf_list = []
                            bug_type_nn_activated = []
                            print('self.problem.objectives_list', self.problem.objectives_list)
                            print('self.problem.objective_weights', self.problem.objective_weights)
                            print('y_list', y_list)
                            for i, y_train in enumerate(y_list):
                                print('np.sum(y_train)', np.sum(y_train), 'self.min_bug_num_to_fit_dnn', self.min_bug_num_to_fit_dnn)
                                if np.sum(y_train) >= self.min_bug_num_to_fit_dnn:
                                    if self.ranking_model == 'nn_sklearn':
                                        clf = MLPClassifier(solver='lbfgs', activation='tanh', max_iter=10000)
                                        clf.fit(X_train, y_train)
                                    elif ranking_model == 'nn_pytorch':
                                        clf = train_net(X_train, y_train, [], [], batch_train=200)
                                    else:
                                        raise
                                    clf_list.append(clf)
                                    bug_type_nn_activated.append(i)

                            if len(clf_list) > 1:
                                if self.adv_conf_th < 0:
                                    adv_conf_th = []
                                    attack_stop_conf = []

                                from scipy import stats
                                one_clf = False
                                scores_on_all_nn = np.zeros([X_test.shape[0], len(clf_list)])
                                for j, clf in enumerate(clf_list):
                                    prob_test = clf.predict_proba(X_test)[:, 1].squeeze()
                                    prob_train = clf.predict_proba(X_train)[:, 1].squeeze()
                                    bug_type = bug_type_nn_activated[j]
                                    cur_y = y_list[bug_type]
                                    print('np.sum(cur_y)', np.sum(cur_y), 'np.abs(self.adv_conf_th)', np.abs(self.adv_conf_th), 'np.sum(cur_y)//np.abs(self.adv_conf_th)', np.sum(cur_y)//np.abs(self.adv_conf_th))

                                    th_conf = sorted(prob_train, reverse=True)[int(np.sum(cur_y)//np.abs(self.adv_conf_th))]
                                    adv_conf_th.append(th_conf)
                                    attack_stop_conf.append(np.max([th_conf, self.attack_stop_conf]))
                                    print('adv_conf_th', adv_conf_th)
                                    print('attack_stop_conf', attack_stop_conf)

                                    y_j_bug_perc = np.mean(cur_y)*100
                                    scores_on_all_nn[:, j] = [(stats.percentileofscore(prob_train, prob_test_i) - (100 - y_j_bug_perc)) / y_j_bug_perc for prob_test_i in prob_test]

                                    print('-'*50)
                                    print(j)
                                    print('y_j_bug_perc', y_j_bug_perc)
                                    print('prob_train', prob_train)
                                    print('prob_test', prob_test)
                                    print('scores_on_all_nn[:, j]', scores_on_all_nn[:, j])
                                    print('-'*50)

                                print(scores_on_all_nn)
                                associated_clf_id = np.argmax(scores_on_all_nn, axis=1)

                                print(associated_clf_id)

                                # TBD: change the name to plural for less confusion
                                clf = clf_list
                            else:
                                clf = clf_list[0]




                        print('\n', 'adv_conf_th', adv_conf_th, '\n')
                        if one_clf == True:
                            pred = clf.predict_proba(X_test)
                            if len(pred.shape) == 1:
                                pred = np.expand_dims(pred, axis=0)
                            scores = pred[:, 1]
                        else:
                            scores = np.max(scores_on_all_nn, axis=1)
                        print('initial scores', scores)
                        # when using unique bugs give preference to unique inputs

                        if self.rank_mode == 'adv_nn':
                            X_test_pgd_ori = None
                            X_test_pgd = None

                        if not self.use_alternate_nn:
                            if self.use_unique_bugs:
                                print('self.tmp_off_type_1_len', self.tmp_off_type_1_len)
                                scores[:self.tmp_off_type_1_len] += np.max(scores)
                                # scores[:self.tmp_off_type_1and2_len] += 100
                            scores *= -1

                            inds = np.argsort(scores)[:self.pop_size]
                            print('scores', scores)
                            print('sorted(scores)', sorted(scores))
                            print('chosen indices', inds)

                        else:
                            # if self.warm_up_path:
                            #     cur_gen = (len(self.problem.objectives_list) - self.warm_up_len) // self.pop_size
                            # else:
                            #     cur_gen = (len(self.problem.objectives_list) - self.initial_fit_th) // self.pop_size
                            cur_gen = self.n_gen
                            cycle_num = self.explore_iter_num+self.exploit_iter_num
                            print('cur_gen', cur_gen, 'cycle_num', cycle_num)
                            if (cur_gen-1) % cycle_num < self.explore_iter_num:
                                current_stage = 'exploration'
                                print('\n', 'exploration', '\n')
                                scores *= -1

                                if self.uncertainty_exploration == 'confidence':
                                    inds_used = np.argsort(scores)
                                else: # random
                                    inds_used = np.arange(len(scores))

                                high_inds = inds_used[:self.high_conf_num]
                                mid_inds = inds_used[self.high_conf_num:len(scores)-self.low_conf_num]

                                print(len(mid_inds), self.high_conf_num, len(scores)-self.low_conf_num)

                                if diversity_mode == 'nn_rep':
                                    d_list = calculate_rep_d(clf, X_train, X_test[mid_inds])
                                    if self.use_unique_bugs:
                                        unique_inds = is_distinct_vectorized(X_test_ori[mid_inds], self.problem.interested_unique_bugs, self.problem.mask, self.problem.xl, self.problem.xu, self.problem.p, self.problem.c, self.problem.th, verbose=False)

                                        print('len(unique_inds)', len(unique_inds))
                                        d_list[unique_inds] += np.max(d_list)
                                    print('X_train.shape[0]', X_train.shape[0])
                                    mid_inds_top_inds = select_batch_max_d_greedy(d_list, X_train.shape[0], self.pop_size)
                                    print('mid_inds_top_inds', mid_inds_top_inds)
                                    inds = mid_inds[mid_inds_top_inds]
                                    print('inds', inds)
                                else:
                                    inds = np.random.choice(mid_inds, self.pop_size, replace=False)

                                if self.rank_mode == 'adv_nn' and not self.adv_exploitation_only:
                                    X_test_pgd_ori = X_test_ori[inds]
                                    X_test_pgd = X_test[inds]
                                else:
                                    self.off = self.tmp_off[inds]

                                print('sorted(scores)', sorted(scores))
                                scores_rank = rankdata(scores)
                                print('chosen indices (rank)', scores_rank[inds])
                                print('chosen indices', inds)

                                if diversity_mode == 'nn_rep':
                                    print('all min distance', np.sort(d_list, axis=1)[:, 1])
                                    print('chosen max min distance', np.sort(d_list[mid_inds_top_inds], axis=1)[:, 1])

                                self.high_conf_configs_stack.append(X_test[high_inds])
                                self.high_conf_configs_ori_stack.append(X_test_ori[high_inds])
                            else:
                                current_stage = 'exploitation'
                                print('\n', 'exploitation', '\n')
                                if len(self.high_conf_configs_stack) > 0:
                                    high_conf_configs_stack_np = np.concatenate(self.high_conf_configs_stack)
                                    high_conf_configs_ori_stack_np = np.concatenate(self.high_conf_configs_ori_stack)

                                    print('len(high_conf_configs_ori_stack_np) before filtering', len(high_conf_configs_ori_stack_np))
                                    # high_conf_configs_ori_stack_np, distinct_inds = get_distinct_data_points(high_conf_configs_ori_stack_np, self.problem.mask, self.problem.xl, self.problem.xu, self.problem.p, self.problem.c, self.problem.th)
                                    # high_conf_configs_ori_stack_np = np.array(high_conf_configs_ori_stack_np)
                                    # high_conf_configs_stack_np = high_conf_configs_stack_np[distinct_inds]
                                    # print('len(high_conf_configs_ori_stack_np) after filtering', len(high_conf_configs_ori_stack_np))

                                    scores = clf.predict_proba(high_conf_configs_stack_np)[:, 1]
                                    if self.use_unique_bugs:
                                        unique_inds = is_distinct_vectorized(high_conf_configs_ori_stack_np, self.problem.interested_unique_bugs, self.problem.mask, self.problem.xl, self.problem.xu, self.problem.p, self.problem.c, self.problem.th, verbose=False)

                                        print('len(unique_inds)', len(unique_inds))
                                        if len(unique_inds) > 0:
                                            scores[unique_inds] += np.max(scores)

                                    scores *= -1
                                    inds = np.argsort(scores)[:self.pop_size]

                                    print('sorted(scores)', sorted(scores))
                                    scores_rank = rankdata(scores)
                                    print('chosen indices (rank)', scores_rank[inds])

                                else:
                                    print('\n'*2, 'len(self.high_conf_configs_stack)', len(self.high_conf_configs_stack), '\n'*2)
                                    inds = []


                                print('chosen indices', inds)

                                if self.rank_mode == 'adv_nn':
                                    X_test_pgd_ori = high_conf_configs_ori_stack_np[inds]
                                    X_test_pgd = high_conf_configs_stack_np[inds]
                                else:
                                    X_test_pgd_ori = high_conf_configs_ori_stack_np[inds]
                                    pop = Population(X_test_pgd_ori.shape[0], individual=Individual())
                                    pop.set("X", X_test_pgd_ori)
                                    pop.set("F", [None for _ in range(X_test_pgd_ori.shape[0])])
                                    self.off = pop

                                self.high_conf_configs_stack = []
                                self.high_conf_configs_ori_stack = []




                    if self.use_alternate_nn and self.rank_mode != 'adv_nn' or (self.use_alternate_nn and self.rank_mode == 'adv_nn' and self.adv_exploitation_only and current_stage == 'exploration'):
                        pass
                    elif self.rank_mode == 'nn':
                        self.off = self.tmp_off[inds]
                    elif self.rank_mode == 'adv_nn':
                        if not self.use_alternate_nn:
                            X_test_pgd_ori = X_test_ori[inds]
                            X_test_pgd = X_test[inds]

                        if one_clf == True:
                            associated_clf_id = []
                        else:
                            associated_clf_id = associated_clf_id[inds]


                        # conduct pgd with constraints differently for different types of inputs
                        if self.use_unique_bugs:
                            unique_coeff = (self.problem.p, self.problem.c, self.problem.th)
                            mask = self.problem.mask

                            y_zeros = np.zeros(X_test_pgd.shape[0])
                            X_test_adv, new_bug_pred_prob_list, initial_bug_pred_prob_list = pgd_attack(clf, X_test_pgd, y_zeros, xl, xu, encoded_fields, labels_used, self.problem.customized_constraints, standardize, prev_X=self.problem.interested_unique_bugs, base_ind=0, unique_coeff=unique_coeff, mask=mask, param_for_recover_and_decode=param_for_recover_and_decode, eps=self.pgd_eps, adv_conf_th=adv_conf_th, attack_stop_conf=attack_stop_conf, associated_clf_id=associated_clf_id, X_test_pgd_ori=X_test_pgd_ori, consider_uniqueness=True)


                        else:
                            y_zeros = np.zeros(X_test_pgd.shape[0])
                            X_test_adv, new_bug_pred_prob_list, initial_bug_pred_prob_list = pgd_attack(clf, X_test_pgd, y_zeros, xl, xu, encoded_fields, labels_used, self.problem.customized_constraints, standardize, eps=self.pgd_eps, adv_conf_th=adv_conf_th, attack_stop_conf=attack_stop_conf, associated_clf_id=associated_clf_id, X_test_pgd_ori=X_test_pgd_ori)


                        X_test_adv_processed = inverse_process_X(X_test_adv, standardize, one_hot_fields_len, partial, X_removed, kept_fields, removed_fields, enc, inds_to_encode, inds_non_encode, encoded_fields)


                        # X_test_adv_processed = customized_inverse_standardize(X_test_adv, standardize, one_hot_fields_len, partial)
                        # X_test_adv_processed = recover_fields_not_changing(X_test_adv_processed, X_removed, kept_fields, removed_fields)
                        # X_test_adv_processed = decode_fields(X_test_adv_processed, enc, inds_to_encode, inds_non_encode, encoded_fields, adv=True)


                        use_combined = False
                        if use_combined:
                            X_combined = np.concatenate([X_test_pgd_ori, X_test_adv_processed], axis=0)
                            X_combined_processed = np.concatenate([X_test_pgd, X_test_adv], axis=0)

                            print('before considering constraints', X_combined.shape[0])
                            chosen_inds = []
                            for i, x in enumerate(X_combined):
                                if not if_violate_constraints(x, self.problem.customized_constraints, self.problem.labels)[0]:
                                    chosen_inds.append(i)
                            chosen_inds = np.array(chosen_inds)

                            X_combined = X_combined[chosen_inds]
                            X_combined_processed = X_combined_processed[chosen_inds]
                            print('after considering constraints', X_combined.shape[0])

                            scores = -1*clf.predict_proba(X_combined_processed)[:, 1]
                            inds = np.argsort(scores)[:self.pop_size]
                            print('scores', scores)
                            print('chosen indices', inds)
                            X_off = X_combined[inds]

                        else:
                            X_off = X_test_adv_processed


                        pop = Population(X_off.shape[0], individual=Individual())
                        pop.set("X", X_off)
                        pop.set("F", [None for _ in range(X_off.shape[0])])
                        self.off = pop


            else:
                self.off = self.tmp_off[:self.pop_size]
        else:
            self.off = self.tmp_off[:self.pop_size]

        if self.only_run_unique_cases:
            X_off = [off_i.X for off_i in self.off]
            remaining_inds = is_distinct_vectorized(X_off, self.problem.interested_unique_bugs, self.problem.mask, self.problem.xl, self.problem.xu, self.problem.p, self.problem.c, self.problem.th, verbose=False)
            self.off = self.off[remaining_inds]

        self.off.set("n_gen", self.n_gen)

        print('\n'*2, 'self.n_gen', self.n_gen, '\n'*2)

        if len(self.all_pop_run_X) == 0:
            self.all_pop_run_X = self.off.get("X")
        else:
            if len(self.off.get("X")) > 0:
                self.all_pop_run_X = np.concatenate([self.all_pop_run_X, self.off.get("X")])

    # mainly used to modify survival
    def _next(self):

        # set self.off
        self.set_off()
        # evaluate the offspring
        # print('start evaluator', 'pop', self.off)
        if len(self.off) > 0:
            self.evaluator.eval(self.problem, self.off, algorithm=self)
        # print('end evaluator')


        if self.algorithm_name == 'random':
            self.pop = self.off
        elif self.emcmc:
            new_pop = do_emcmc(parents, self.off, self.n_gen, self.problem.objective_weights)

            self.pop = Population.merge(self.pop, new_pop)

            if self.survival:
                self.pop = self.survival.do(self.problem, self.pop, self.survival_size, algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)
        else:
            # merge the offsprings with the current population
            self.pop = Population.merge(self.pop, self.off)

            # the do survival selection
            if self.survival:
                print('\n'*3)
                print('len(self.pop) before', len(self.pop))
                print('survival')
                self.pop = self.survival.do(self.problem, self.pop, self.survival_size, algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)
                print('len(self.pop) after', len(self.pop))
                print(self.pop_size, self.survival_size)
                print('\n'*3)



    def _initialize(self):
        if self.warm_up_path and ((self.dt and not self.problem.cumulative_info) or (not self.dt)):
            subfolders = get_sorted_subfolders(self.warm_up_path)
            X, _, objectives_list, mask, _, _ = load_data(subfolders)

            if self.warm_up_len > 0:
                X = X[:self.warm_up_len]
                objectives_list = objectives_list[:self.warm_up_len]
            else:
                self.warm_up_len = len(X)

            xl = self.problem.xl
            xu = self.problem.xu
            p, c, th = self.problem.p, self.problem.c, self.problem.th
            unique_coeff = (p, c, th)


            self.problem.unique_bugs, (self.problem.bugs, self.problem.bugs_type_list, self.problem.bugs_inds_list, self.problem.interested_unique_bugs) = get_unique_bugs(
                X, objectives_list, mask, xl, xu, unique_coeff, self.problem.objective_weights, return_mode='return_bug_info', consider_interested_bugs=self.problem.consider_interested_bugs
            )

            print('\n'*10)
            print('self.problem.bugs', len(self.problem.bugs))
            print('self.problem.unique_bugs', len(self.problem.unique_bugs))
            print('\n'*10)

            self.all_pop_run_X = np.array(X)
            self.problem.objectives_list = objectives_list.tolist()

        if self.dt:
            X_list = list(self.X)
            F_list = list(self.F)
            pop = Population(len(X_list), individual=Individual())
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen, "CV", [0 for _ in range(len(X_list))], "feasible", [[True] for _ in range(len(X_list))])
            self.pop = pop
            self.set_off()
            pop = self.off

        elif self.warm_up_path:
            X_list = X[-self.pop_size:]
            current_objectives = objectives_list[-self.pop_size:]

            # current_objectives = np.dot(current_objectives, np.expand_dims(np.array(objective_weights), axis=1))
            # F_list = current_objectives.tolist()

            F_list = get_F(current_objectives, objectives_list, self.problem.objective_weights, self.problem.use_single_objective)

            # print('F_list', sorted(F_list))
            # print('len(self.all_pop_run_X)', len(self.all_pop_run_X))

            pop = Population(len(X_list), individual=Individual())
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen, "CV", [0 for _ in range(len(X_list))], "feasible", [[True] for _ in range(len(X_list))])

            self.pop = pop
            self.set_off()
            pop = self.off

        else:
            # create the initial population
            # pop = Population(0, individual=self.individual)
            # pop = self.sampling.do(self.problem, pop_size, pop=pop, algorithm=self)
            # pop = self.repair.do(self.problem, pop, algorithm=self)
            #
            # if len(pop) < self.pop_size:
            #     remaining_num = self.pop_size - len(pop)
            #     remaining_pop = self.plain_initialization.do(self.problem, remaining_num, algorithm=self)
            #     pop = Population.merge(pop, remaining_pop)

            if self.use_unique_bugs:
                pop = self.initialization.do(self.problem, self.problem.fuzzing_arguments.pop_size, algorithm=self)
            else:
                pop = self.plain_initialization.do(self.problem, self.pop_size, algorithm=self)
            pop.set("n_gen", self.n_gen)


        if len(pop) > 0:
            self.evaluator.eval(self.problem, pop, algorithm=self)
        print('\n'*5, 'after initialize evaluator', '\n'*5)
        print('len(self.all_pop_run_X)', len(self.all_pop_run_X))
        # if not self.warm_up_path:
        #     # then evaluate using the objective function
        #     self.evaluator.eval(self.problem, pop, algorithm=self)

        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)

        self.pop, self.off = pop, pop

        # print('\n'*5)
        # print(self.pop)
        # print(self.pop.individual)
        # print(self.pop.individual.X)
        # print(self.pop.get("X"))
        # print('\n'*5)









class ClipRepair(Repair):
    """
    A dummy class which can be used to simply do no repair.
    """

    def do(self, problem, pop, **kwargs):
        for i in range(len(pop)):
            pop[i].X = np.clip(pop[i].X, np.array(problem.xl), np.array(problem.xu))
        return pop


class SimpleDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, mask, xu, xl, check_unique_coeff, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask = np.array(mask)
        self.xu = np.array(xu)
        self.xl = np.array(xl)
        self.cmp = lambda a, b: self.is_equal(a, b)
        self.check_unique_coeff = check_unique_coeff
        assert len(self.check_unique_coeff) == 3

    def is_equal(self, a, b):
        if type(b).__module__ == np.__name__:
            b_X = b
        else:
            b_X = b.X
        if type(a).__module__ == np.__name__:
            a_X = a
        else:
            a_X = a.X
        p, c, th = self.check_unique_coeff
        return is_similar(a_X, b_X, self.mask, self.xl, self.xu, p, c, th)




class MyEvaluator(Evaluator):
    def __init__(self, correct_spawn_locations_after_run=0, correct_spawn_locations=None, **kwargs):
        super().__init__()
        self.correct_spawn_locations_after_run = correct_spawn_locations_after_run
        self.correct_spawn_locations = correct_spawn_locations
    def _eval(self, problem, pop, **kwargs):

        super()._eval(problem, pop, **kwargs)


        if self.correct_spawn_locations_after_run:
            correct_spawn_locations_all(pop[i].X, problem.labels)


        # print(pop[0].X)



def customized_minimize(problem,
             algorithm,
             termination=None,
             **kwargs):
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



def run_nsga2_dt(fuzzing_arguments, sim_specific_arguments, fuzzing_content, run_simulation):

    end_when_no_critical_region = True
    cumulative_info = None

    X_filtered = None
    F_filtered = None
    X = None
    y = None
    F = None
    labels = None
    estimator = None
    critical_unique_leaves = None


    now = datetime.now()
    dt_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    if fuzzing_arguments.warm_up_path:
        subfolders = get_sorted_subfolders(fuzzing_arguments.warm_up_path)
        X, _, objectives_list, _, _, _ = load_data(subfolders)

        if fuzzing_arguments.warm_up_len > 0:
            X = X[:fuzzing_arguments.warm_up_len]
            objectives_list = objectives_list[:fuzzing_arguments.warm_up_len]

        y = determine_y_upon_weights(objectives_list, fuzzing_arguments.objective_weights)
        F = get_F(objectives_list, objectives_list, fuzzing_arguments.objective_weights, fuzzing_arguments.use_single_objective)

        estimator, inds, critical_unique_leaves = filter_critical_regions(np.array(X), y)
        X_filtered = np.array(X)[inds]
        F_filtered = F[inds]



    for i in range(fuzzing_arguments.outer_iterations):
        dt_time_str_i = dt_time_str
        dt = True
        if (i == 0 and not fuzzing_arguments.warm_up_path) or np.sum(y)==0:
            dt = False


        dt_arguments = emptyobject(
            call_from_dt=True,
            dt=dt,
            X=X_filtered,
            F=F_filtered,
            estimator=estimator,
            critical_unique_leaves=critical_unique_leaves,
            dt_time_str=dt_time_str_i, dt_iter=i, cumulative_info=cumulative_info)


        X_new, y_new, F_new, _, labels, parent_folder, cumulative_info, all_pop_run_X, objective_list, objective_weights = run_ga(fuzzing_arguments, sim_specific_arguments, fuzzing_content, run_simulation, dt_arguments=dt_arguments)



        if fuzzing_arguments.finish_after_has_run and cumulative_info['has_run'] > fuzzing_arguments.has_run_num:
            break

        if len(X_new) == 0:
            break

        if i == 0 and not fuzzing_arguments.warm_up_path:
            X = X_new
            y = y_new
            F = F_new
        else:
            X = np.concatenate([X, X_new])
            y = np.concatenate([y, y_new])
            F = np.concatenate([F, F_new])


        estimator, inds, critical_unique_leaves = filter_critical_regions(X, y)
        # print(X, F, inds)
        X_filtered = X[inds]
        F_filtered = F[inds]

        if len(X_filtered) == 0 and end_when_no_critical_region:
            break



def run_ga(fuzzing_arguments, sim_specific_arguments, fuzzing_content, run_simulation, dt_arguments=None):

    if not dt_arguments:
        dt_arguments = emptyobject(
            call_from_dt=False,
            dt=False,
            X=None,
            F=None,
            estimator=None,
            critical_unique_leaves=None,
            dt_time_str=None, dt_iter=None, cumulative_info=None)

    if dt_arguments.call_from_dt:
        fuzzing_arguments.termination_condition = 'generations'
        if dt_arguments.dt and len(list(dt_arguments.X)) == 0:
            print('No critical leaves!!! Start from random sampling!!!')
            dt_arguments.dt = False

        time_str = dt_arguments.dt_time_str

    else:
        now = datetime.now()
        p, c, th = fuzzing_arguments.check_unique_coeff
        time_str = now.strftime("%Y_%m_%d_%H_%M_%S")+','+'_'.join([str(fuzzing_arguments.pop_size), str(fuzzing_arguments.n_gen), fuzzing_arguments.rank_mode, str(fuzzing_arguments.has_run_num), str(fuzzing_arguments.initial_fit_th), str(fuzzing_arguments.pgd_eps), str(fuzzing_arguments.adv_conf_th), str(fuzzing_arguments.attack_stop_conf), 'coeff', str(p), str(c), str(th), fuzzing_arguments.uncertainty, fuzzing_arguments.model_type, 'n_offsprings', str(fuzzing_arguments.n_offsprings), str(fuzzing_arguments.mating_max_iterations), str(fuzzing_arguments.sample_multiplier), 'only_unique', str(fuzzing_arguments.only_run_unique_cases), 'eps', str(fuzzing_arguments.pgd_eps)])

    cur_parent_folder = make_hierarchical_dir([fuzzing_arguments.root_folder, fuzzing_arguments.algorithm_name, fuzzing_arguments.route_type, fuzzing_arguments.scenario_type, fuzzing_arguments.ego_car_model, time_str])

    if dt_arguments.call_from_dt:
        parent_folder = make_hierarchical_dir([cur_parent_folder, str(dt_arguments.dt_iter)])
    else:
        parent_folder = cur_parent_folder

    fuzzing_arguments.parent_folder = parent_folder
    fuzzing_arguments.mean_objectives_across_generations_path = os.path.join(parent_folder, 'mean_objectives_across_generations.txt')

    problem = MyProblem(fuzzing_arguments, sim_specific_arguments, fuzzing_content, run_simulation, dt_arguments)


    # deal with real and int separately
    crossover = MixedVariableCrossover(problem.mask, {
        "real": get_crossover("real_sbx", prob=0.8, eta=5),
        "int": get_crossover("int_sbx", prob=0.8, eta=5)
    })

    mutation = MixedVariableMutation(problem.mask, {
        "real": get_mutation("real_pm", eta=5, prob=int(0.05*problem.n_var)),
        "int": get_mutation("int_pm", eta=5, prob=int(0.05*problem.n_var))
    })


    selection = TournamentSelection(func_comp=binary_tournament)
    repair = ClipRepair()

    eliminate_duplicates = NoDuplicateElimination()

    mating = MyMatingVectorized(selection,
                    crossover,
                    mutation,
                    fuzzing_arguments.use_unique_bugs,
                    fuzzing_arguments.emcmc,
                    fuzzing_arguments.mating_max_iterations,
                    repair=repair,
                    eliminate_duplicates=eliminate_duplicates)


    sampling = MySamplingVectorized(use_unique_bugs=fuzzing_arguments.use_unique_bugs, check_unique_coeff=problem.check_unique_coeff, sample_multiplier=fuzzing_arguments.sample_multiplier)

    plain_sampling = MySamplingVectorized(use_unique_bugs=False, check_unique_coeff=problem.check_unique_coeff, sample_multiplier=fuzzing_arguments.sample_multiplier)

    # TBD: customize mutation and crossover to better fit our problem. e.g.
    # might deal with int and real separately
    algorithm = NSGA2_DT(dt=dt_arguments.dt, X=dt_arguments.X, F=dt_arguments.F, fuzzing_arguments=fuzzing_arguments, plain_sampling=plain_sampling, sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=eliminate_duplicates,
    repair=repair,
    mating=mating)


    # close simulator(s)
    atexit.register(exit_handler, fuzzing_arguments.ports)

    if fuzzing_arguments.termination_condition == 'generations':
        termination = ('n_gen', fuzzing_arguments.n_gen)
    elif fuzzing_arguments.termination_condition == 'max_time':
        termination = ('time', fuzzing_arguments.max_running_time)
    else:
        termination = ('n_gen', fuzzing_arguments.n_gen)

    if hasattr(sim_specific_arguments, 'correct_spawn_locations_after_run'):
        correct_spawn_locations_after_run = sim_specific_arguments.correct_spawn_locations_after_run
        correct_spawn_locations = sim_specific_arguments.correct_spawn_locations
    else:
        correct_spawn_locations_after_run = False
        correct_spawn_locations = None

    res = customized_minimize(problem,
                   algorithm,
                   termination=termination,
                   seed=0,
                   verbose=False,
                   save_history=False,
                   evaluator=MyEvaluator(correct_spawn_locations_after_run=correct_spawn_locations_after_run, correct_spawn_locations=correct_spawn_locations))

    print('We have found', len(problem.bugs), 'bugs in total.')



    if len(problem.x_list) > 0:
        X = np.stack(problem.x_list)
        F = np.concatenate(problem.F_list)
        objectives = np.stack(problem.objectives_list)
    else:
        X = []
        F = []
        objectives = []

    y = np.array(problem.y_list)
    time_list = np.array(problem.time_list)
    bugs_num_list = np.array(problem.bugs_num_list)
    unique_bugs_num_list = np.array(problem.unique_bugs_num_list)
    labels = problem.labels
    has_run = problem.has_run
    has_run_list = problem.has_run_list

    mask = problem.mask
    xl = problem.xl
    xu = problem.xu
    p = problem.p
    c = problem.c
    th = problem.th



    cumulative_info = {
        'has_run': problem.has_run,
        'start_time': problem.start_time,
        'counter': problem.counter,
        'time_list': problem.time_list,
        'bugs': problem.bugs,
        'unique_bugs': problem.unique_bugs,
        'interested_unique_bugs': problem.interested_unique_bugs,
        'bugs_type_list': problem.bugs_type_list,
        'bugs_inds_list': problem.bugs_inds_list,
        'bugs_num_list': problem.bugs_num_list,
        'unique_bugs_num_list': problem.unique_bugs_num_list,
        'has_run_list': problem.has_run_list
    }


    return X, y, F, objectives, labels, cur_parent_folder, cumulative_info, algorithm.all_pop_run_X, problem.objectives_list, problem.objective_weights



def run_ga_general(fuzzing_arguments, sim_specific_arguments, fuzzing_content, run_simulation):
    if fuzzing_arguments.algorithm_name in ['nsga2-un-dt', 'nsga2-dt']:
        run_nsga2_dt(fuzzing_arguments, sim_specific_arguments, fuzzing_content, run_simulation)
    else:
        run_ga(fuzzing_arguments, sim_specific_arguments, fuzzing_content, run_simulation)

if __name__ == '__main__':
    '''
    fuzzing_arguments: parameters needed for the fuzzing process, see argparse for details.

    sim_specific_arguments: parameters specific to the simulator used.

    fuzzing_content: a description of the search space.
        labels:
        mask:
        parameters_min_bounds:
        parameters_max_bounds:
        parameters_distributions:
        customized_constraints:
        customized_center_transforms:
        n_var:
        fixed_hyperparameters:
        search_space_info:

    run_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, ...) -> objectives, run_info: a simulation function specific to the simulator used.
        objectives:
        run_info:


    TBD: svl simulator
    TBD: flexible uniqueness filteration / bug counting, flexible search objectives


    '''

    if fuzzing_arguments.simulator == 'carla':
        from carla_specific_utils.scene_configs import customized_bounds_and_distributions
        from carla_specific_utils.setup_labels_and_bounds import generate_fuzzing_content
        from carla_specific_utils.carla_specific import run_carla_simulation, initialize_carla_specific, correct_spawn_locations_all

        customized_config = customized_bounds_and_distributions[fuzzing_arguments.scenario_type]
        fuzzing_content = generate_fuzzing_content(customized_config)
        sim_specific_arguments = initialize_carla_specific(fuzzing_arguments)
        run_simulation = run_carla_simulation

    elif fuzzing_arguments.simulator == 'svl':
        from svl_script.scene_configs import customized_bounds_and_distributions
        from svl_script.setup_labels_and_bounds import generate_fuzzing_content
        from svl_script.svl_specific import run_svl_simulation, initialize_svl_specific

        # 'apollo_6_with_signal', 'apollo_6_modular'
        fuzzing_arguments.ego_car_model = 'apollo_6_with_signal'
        fuzzing_arguments.route_type = 'BorregasAve_forward'
        fuzzing_arguments.scenario_type = 'default'
        fuzzing_arguments.ports = [8181]
        fuzzing_arguments.root_folder = 'run_results_svl'

        customized_config = customized_bounds_and_distributions[fuzzing_arguments.scenario_type]
        fuzzing_content = generate_fuzzing_content(customized_config)
        # print('fuzzing content', str(fuzzing_content))
        # print(len(fuzzing_content.__dict__))
        print(fuzzing_content.labels)
        print(len(fuzzing_content.labels))
        print(len(fuzzing_content.parameters_min_bounds))
        sim_specific_arguments = initialize_svl_specific(fuzzing_arguments)
        run_simulation = run_svl_simulation

    else:
        raise
    run_ga_general(fuzzing_arguments, sim_specific_arguments, fuzzing_content, run_simulation)

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













# def eliminate_duplicates_for_list(
#     mask, xl, xu, p, c, th, X, prev_unique_bugs, tmp_off=[]
# ):
#     new_X = []
#     similar = False
#     for x in X:
#         for x2 in prev_unique_bugs:
#             if is_similar(x, x2, mask, xl, xu, p, c, th):
#                 similar = True
#                 break
#         if not similar:
#             for x2 in tmp_off:
#                 # print(x)
#                 # print(x2)
#                 # print(mask, xl, xu, p, c, th)
#                 # print(len(x), len(x2), len(mask), len(xl), len(xu))
#                 if is_similar(x, x2, mask, xl, xu, p, c, th):
#                     similar = True
#                     break
#         if not similar:
#             new_X.append(x)
#     return new_X

# def is_similar(
#     x_1,
#     x_2,
#     mask,
#     xl,
#     xu,
#     p,
#     c,
#     th,
#     y_i=-1,
#     y_j=-1,
#     verbose=False,
#     labels=[],
# ):
#
#     if y_i == y_j:
#         eps = 1e-8
#
#         # only consider those fields that can change when considering diversity
#         variant_fields = (xu - xl) > eps
#         mask = mask[variant_fields]
#         xl = xl[variant_fields]
#         xu = xu[variant_fields]
#         x_1 = x_1[variant_fields]
#         x_2 = x_2[variant_fields]
#         variant_fields_num = np.sum(variant_fields)
#         if verbose:
#             print(
#                 variant_fields_num,
#                 "/",
#                 len(variant_fields),
#                 "fields are used for checking similarity",
#             )
#
#         int_inds = mask == "int"
#         real_inds = mask == "real"
#         # print(int_inds, real_inds)
#         int_diff_raw = np.abs(x_1[int_inds] - x_2[int_inds])
#         int_diff = np.ones(int_diff_raw.shape) * (int_diff_raw > eps)
#
#         real_diff_raw = (
#             np.abs(x_1[real_inds] - x_2[real_inds]) / (np.abs(xu - xl) + eps)[real_inds]
#         )
#         # print(int_diff_raw, real_diff_raw)
#         real_diff = np.ones(real_diff_raw.shape) * (real_diff_raw > c)
#
#         diff = np.concatenate([int_diff, real_diff])
#         # print(diff, p)
#         diff_norm = np.linalg.norm(diff, p)
#
#         th_num = np.max([np.round(th * variant_fields_num), 1])
#         equal = diff_norm < th_num
#
#         if verbose:
#             print("diff_norm, th_num", diff_norm, th_num)
#
#     else:
#         equal = False
#     return equal
#
# def is_distinct(x, X, mask, xl, xu, p, c, th, verbose=True):
#     verbose = False
#     if len(X) == 0:
#         return True
#     else:
#         mask_np = np.array(mask)
#         xl_np = np.array(xl)
#         xu_np = np.array(xu)
#         x = np.array(x)
#         X = np.stack(X)
#         for i, x_i in enumerate(X):
#             # if verbose:
#             #     print(i, '- th prev x checking similarity')
#             similar = is_similar(
#                 x,
#                 x_i,
#                 mask_np,
#                 xl_np,
#                 xu_np,
#                 p,
#                 c,
#                 th,
#                 verbose=verbose,
#             )
#             if similar:
#                 if verbose:
#                     print("similar with", i)
#                 return False
#         return True

# def get_distinct_data_points(data_points, mask, xl, xu, p, c, th, y=[]):
#
#     # ['forward', 'backward']
#     order = "forward"
#
#     mask_arr = np.array(mask)
#     xl_arr = np.array(xl)
#     xu_arr = np.array(xu)
#     # print(data_points)
#     if len(data_points) == 0:
#         return [], []
#     if len(data_points) == 1:
#         return data_points, [0]
#     else:
#         if order == "backward":
#             distinct_inds = []
#             for i in range(len(data_points) - 1):
#                 similar = False
#                 for j in range(i + 1, len(data_points)):
#                     if len(y) > 0:
#                         y_i = y[i]
#                         y_j = y[j]
#                     else:
#                         y_i = -1
#                         y_j = -1
#                     similar = is_similar(
#                         data_points[i],
#                         data_points[j],
#                         mask_arr,
#                         xl_arr,
#                         xu_arr,
#                         p,
#                         c,
#                         th,
#                         y_i=y_i,
#                         y_j=y_j,
#                     )
#                     if similar:
#                         break
#                 if not similar:
#                     distinct_inds.append(i)
#             distinct_inds.append(len(data_points) - 1)
#         elif order == "forward":
#             distinct_inds = [0]
#             for i in range(1, len(data_points)):
#                 similar = False
#                 for j in distinct_inds:
#                     if len(y) > 0:
#                         y_i = y[i]
#                         y_j = y[j]
#                     else:
#                         y_i = -1
#                         y_j = -1
#                     similar = is_similar(
#                         data_points[i],
#                         data_points[j],
#                         mask_arr,
#                         xl_arr,
#                         xu_arr,
#                         p,
#                         c,
#                         th,
#                         y_i=y_i,
#                         y_j=y_j,
#                     )
#                     if similar:
#                         # print(i, j)
#                         break
#                 if not similar:
#                     distinct_inds.append(i)
#
#     return list(np.array(data_points)[distinct_inds]), distinct_inds

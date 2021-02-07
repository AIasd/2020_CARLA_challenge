"""
threats to validity:
type of controllers
realism of carla
other type of violations
traffic sign / road shape
other search methods


# tomorrow TBD

# 1.distinct from previous unique bugs and current X
# 2.distinct from previous unique bugs
# 3.random





# rank unique 1 samples (with additional 2) and unique 2 samples (with additional 1)
# adv 1 with both previous unique bugs and current other unique 1
# adv 2 only w.r.t. previous unique bugs








# attack the one with higher ranking in the training sample confidence also dnn only be activated when the number of bugs found reach at least 30 for that type of bug?







# ** NSGA2-SM baseline debug
# ** clean up code by the end of this week
# ** check single objective (collision / out-of-road in general and consider it as bug) performance under 0.2 0.5

debug forward d
debug ga-adv multi dnn (stop threshold and run adv_nn threshold, 2 dnn VS 2 heads (collision and out-of-road separately) to attack?)
try GA+DT

writing

tsne of settled attack methods during rerun (use rerun results as baseline results)





















tsne of different types of bug
tune parameters for advnn






new def of uniqueness (entropy based? then set threshold and report numbers?)




analyze relationship between confidence and bug likelihood


try adv_nn not considering uniqueness in ga_fuzzing??? or also consider other same generation x???


try more complex adv nn?

tune performance (uniqueness params, adv eps, adv what candidates???)

tune pgd eps thresholds for town01 and town03




BADGE+high conf alternative?






try DNN for each type of bug? and maybe do ranking and adv using the one with higher confidence?

regression net baseline (consider confidence intervals)
tsne of rerun (use rerun correctness for before perturbation performance)
clean up code a bit for reproducing




fix memory error
implement BADGE
run BALD_conf and BatchBALD_conf and BADGE




# all multi adv aggressive 389 for 650
python ga_fuzzing.py -p 2015 2024 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 1 1 -1 0 0 0 -1 --n_offsprings 500 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5 --use_single_nn 0 --adv_conf_th -8

# all multi adv 394
python ga_fuzzing.py -p 2015 2024 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 1 1 -1 0 0 0 -1 --n_offsprings 500 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5 --use_single_nn 0

# all nsga2-un 393
python ga_fuzzing.py -p 2018 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 1 1 -1 0 0 0 -1 --rank_mode none --check_unique_coeff 0 0.1 0.5





# collision nsga2-un 404
python ga_fuzzing.py -p 2021 -s 8797 -d 8798 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --check_unique_coeff 0 0.1 0.5 --pgd_eps 0.0 --adv_conf_th 0.0 --attack_stop_conf 0.0

# collision nsga2-un-nn 430
python ga_fuzzing.py -p  -s 8800 -d 8801 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5 --pgd_eps 0.0 --adv_conf_th 0.0 --attack_stop_conf 0.0

# collision nsga2-un-adv-nn 450
python ga_fuzzing.py -p 2027 2030 -s 8803 -d 8804 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5

# collision regression-nn 386
python ga_fuzzing.py -p 2015 -s 8797 -d 8798 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name regression-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --check_unique_coeff 0 0.1 0.5


# collision random-un-adv-nn 318
python ga_fuzzing.py -p 2027 2030 -s 8803 -d 8804 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name random-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5

# collision random-un 335
python ga_fuzzing.py -p 2033 2036 -s 8805 -d 8806 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name random-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --check_unique_coeff 0 0.1 0.5



# collision random-un-nn 372
python ga_fuzzing.py -p 2015 2018 -s 8803 -d 8804 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name random-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5 --pgd_eps 0.0 --adv_conf_th 0.0 --attack_stop_conf 0.0






# collision nsga2-un 202
python ga_fuzzing.py -p 2021 -s 8800 -d 8801 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --check_unique_coeff 0 0.1 0.5 --pgd_eps 0.0 --adv_conf_th 0.0 --attack_stop_conf 0.0

# collision nsga2-un-nn 203
python ga_fuzzing.py -p 2027 2030 -s 8803 -d 8804 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5 --pgd_eps 0.0 --adv_conf_th 0.0 --attack_stop_conf 0.0

# collision nsga2-un-adv-nn 184
python ga_fuzzing.py -p 2034 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5

# collision nsga2-un-adv-nn aggressive 159
python ga_fuzzing.py -p 2034 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5 --adv_conf_th -8




# 1.BADGE (distance^2+critical)
2.diverse mini-batch: clustering feature vector on top uncertain+critical cases and take centroids
# 3.BatchBALD (uncertain + diversity + critical)
4.core-set on top uncertain+critical cases (or consider distance based on feature vector distance + uncertain + critical)
5.disagreement among models (average uncertainty - average individual uncertainty or maybe only the former) + average critical + feature vector distance

uncertain measurements: BALD (average entropy - mean individual entropy, where individual is applying dropout on a DNN), gradient (as in BADGE)



# collision nsga2-un 301
python ga_fuzzing.py -p 2021 2024 -s 8800 -d 8801 --n_gen 10 --pop_size 100 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 1000 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --check_unique_coeff 0 0.05 0.25 --pgd_eps 0.0 --adv_conf_th 0.0 --attack_stop_conf 0.0

# collision nsga2-un-adv-nn 271
python ga_fuzzing.py -p 2015 2018 -s 8806 -d 8807 --n_gen 10 --pop_size 100 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 1000 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.05 0.25 --adv_conf_th 0.8 --attack_stop_conf 0.8 --initial_fit_th 500






# collision nsga2-un
python ga_fuzzing.py -p 2021 2024 -s 8800 -d 8801 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --check_unique_coeff 0 0.1 0.5 --pgd_eps 0.0 --adv_conf_th 0.0 --attack_stop_conf 0.0

# collision nsga2-un-adv-nn
python ga_fuzzing.py -p 2015 2018 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5 --adv_conf_th 0.0 --attack_stop_conf 0.0 --initial_fit_th 400













# collision nsga2-un 427
python ga_fuzzing.py -p 2021 2024 -s 8800 -d 8801 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --check_unique_coeff 0 0.1 0.5

# collision nsga2-un-nn server running 401
python ga_fuzzing.py -p 2021 2024 -s 8800 -d 8801 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5 --pgd_eps 0.0 --adv_conf_th 0.0 --attack_stop_conf 0.0


# collision nsga2-un-adv-nn 339
python ga_fuzzing.py -p 2027 2030 -s 8800 -d 8801 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5





# collision nsga2-un 114
python ga_fuzzing.py -p 2015 -s 8779 -d 8780 --n_gen 14 --pop_size 50 -r 'town05_front_0' -c 'change_lane_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --check_unique_coeff 0 0.1 0.5

# collision nsga2-un-nn 104
python ga_fuzzing.py -p 2018 -s 8781 -d 8782 --n_gen 14 --pop_size 50 -r 'town05_front_0' -c 'change_lane_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5 --pgd_eps 0.0 --adv_conf_th 0.0 --attack_stop_conf 0.0

# collision nsga2-un-adv-nn 80
python ga_fuzzing.py -p 2021 -s 8785 -d 8786 --n_gen 14 --pop_size 50 -r 'town05_front_0' -c 'change_lane_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --check_unique_coeff 0 0.1 0.5











python ga_fuzzing.py -p 2015 2024 -s 8791 -d 8792 --n_gen 5 --pop_size 2 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --n_offsprings 50 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --initial_fit_th 2 --min_bug_num_to_fit_dnn 0 --uncertainty BADGE_none --model_type one_output



# 413
python ga_fuzzing.py -p 2012 2015 -s 8788 -d 8789 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BatchBALD_none

# 377 for 650
python ga_fuzzing.py -p 2012 2015 -s 8791 -d 8792 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BatchBALD_conf

# TBD
python ga_fuzzing.py -p 2018 2021 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BALD_conf

# 417 418(-157)
python ga_fuzzing.py -p 2021 -s 8797 -d 8798 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty Random_none

# 428
python ga_fuzzing.py -p 2024 -s 8800 -d 8801 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BUGCONF_none



# 397
python ga_fuzzing.py -p 2012 -s 8788 -d 8789 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_conf --model_type one_output

# 421
python ga_fuzzing.py -p 2018 -s 8802 -d 8803 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output

# 381
python ga_fuzzing.py -p 2015 -s 8800 -d 8801 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_conf --model_type one_output

# 431
python ga_fuzzing.py -p 2021 -s 8804 -d 8805 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output

# 168
python ga_fuzzing.py -p 2024 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_conf --model_type one_output



# 162(-67)
python ga_fuzzing.py -p 2024 2027 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output

# 204(-88)
python ga_fuzzing.py -p 2015 2018 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output


# TBD
python ga_fuzzing.py -p 2024 2027 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town05_front_0' -c 'change_lane_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output

# 169(-72)
python ga_fuzzing.py -p 2024 2027 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type BNN

# 193(-93)
python ga_fuzzing.py -p 2015 2018 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type BNN



#########################################################
After fixing uncertainty_conf True/False

python ga_fuzzing.py -p 2024 2027 -s 8806 -d 8807 --n_gen 7 --pop_size 2 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 10 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output --initial_fit_th 2 --min_bug_num_to_fit_dnn 0 --n_offsprings 20 --explore_iter_num 2 --exploit_iter_num 1 --high_conf_num 5 --low_conf_num 5



# 151
python ga_fuzzing.py -p 2024 2027 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output


# 225(-94)
python ga_fuzzing.py -p 2015 2018 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output


# 98
python ga_fuzzing.py -p 2015 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town05_front_0' -c 'change_lane_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output

# 403
python ga_fuzzing.py -p 2018 -s 8796 -d 8797 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output

# 360
python ga_fuzzing.py -p 2021 -s 8798 -d 8799 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output

# 365
python ga_fuzzing.py -p 2024 -s 8800 -d 8801 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_none --model_type one_output --survival_multiplier 2


############################################################

# 112
python ga_fuzzing.py -p 2015 2018 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town05_front_0' -c 'change_lane_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_conf --model_type one_output

# 361
python ga_fuzzing.py -p 2021 2024 -s 8796 -d 8797 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BADGE_conf --model_type one_output --survival_multiplier 2

# TBD
python ga_fuzzing.py -p 2021 2024 -s 8796 -d 8797 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BALD --model_type one_output --survival_multiplier 2

############################################################

# 200(-88)
python ga_fuzzing.py -p 2015 2018 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output



# 393(-154)
python ga_fuzzing.py -p 2024 2027 -s 8808 -d 8809 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output


# server running
python ga_fuzzing.py -p 2015 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output --explore_iter_num 3 --high_conf_num 50 --low_conf_num 50


# server running
python ga_fuzzing.py -p 2024 -s 8808 -d 8809 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output --explore_iter_num 3 --high_conf_num 50 --low_conf_num 50




# 189
python ga_fuzzing.py -p 2015 -s 8806 -d 8807 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output --explore_iter_num 3 --high_conf_num 25 --low_conf_num 25 --survival_multiplier 2


# 390
python ga_fuzzing.py -p 2018 -s 8808 -d 8809 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output --explore_iter_num 3 --high_conf_num 25 --low_conf_num 25 --survival_multiplier 2


# 219(-105)/221(-99)
python ga_fuzzing.py -p 2021 -s 8810 -d 8811 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output --explore_iter_num 3 --high_conf_num 200 --low_conf_num 200 --survival_multiplier 2


# 368
python ga_fuzzing.py -p 2021 2024 -s 8812 -d 8813 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output --explore_iter_num 3 --high_conf_num 200 --low_conf_num 200 --survival_multiplier 2

# 421
python ga_fuzzing.py -p 2015 2018 -s 8810 -d 8811 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --check_unique_coeff 0 0.1 0.5 --model_type one_output --explore_iter_num 3 --high_conf_num 200 --low_conf_num 200 --survival_multiplier 2


#############################################################





# 402(-161)
python ga_fuzzing.py -p 2018 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty Random_none

# 356
python ga_fuzzing.py -p 2015 2018 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BUGCONF_none


# 174(-73)
python ga_fuzzing.py -p 2021 -s 8797 -d 8798 --n_gen 14 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty Random_none

# 170
python ga_fuzzing.py -p 2021 2024 -s 8797 -d 8798 --n_gen 14 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty BUGCONF_none

# 210(-88) 224(-95)
python ga_fuzzing.py -p 2015 2018 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty Random_none

# 113
python ga_fuzzing.py -p 2027 -s 8803 -d 8804 --n_gen 14 --pop_size 50 -r 'town05_front_0' -c 'change_lane_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty Random_none

############################################################

# regression with classification label 180, reg label 183
python ga_fuzzing.py -p 2018 2021 -s 8801 -d 8802 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --ranking_model regression

# 212(-94) using 0 prob rather than 1 prob
python ga_fuzzing.py -p 2024 2027 -s 8803 -d 8804 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --ranking_model adaboost
#############################################################



# 399(-210)
python ga_fuzzing.py -p 2015 -s 8794 -d 8795 --n_gen 12 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --warm_up_path 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN'

# 419(-210)
python ga_fuzzing.py -p 2018 -s 8796 -d 8797 --n_gen 12 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --warm_up_path 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN'

# 773(-400)
python ga_fuzzing.py -p 2021 -s 8798 -d 8799 --n_gen 12 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --warm_up_path 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_01_23_16_09,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

# 841(-400)
python ga_fuzzing.py -p 2024 -s 8800 -d 8801 --n_gen 12 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --warm_up_path 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_01_23_16_09,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

# 299(-174)
python ga_fuzzing.py -p 2015 2018 -s 8794 -d 8795 --n_gen 12 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --warm_up_path 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_01_23_16_14,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

# 358(-174)
python ga_fuzzing.py -p 2021 2024 -s 8796 -d 8797 --n_gen 12 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --warm_up_path 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_01_23_16_14,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

python ga_fuzzing.py -p 2027 -s 8802 -d 8803 --n_gen 30 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 1500 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none

#############################################################







# 848(-400)
python ga_fuzzing.py -p 2015 2018 -s 8794 -d 8795 --n_gen 12 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --warm_up_path 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_01_23_16_09,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

# 423(-210)
python ga_fuzzing.py -p 2021 2024 -s 8798 -d 8799 --n_gen 12 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode alternate_nn --warm_up_path 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN'


#############################################################
# running
python ga_fuzzing.py -p 2015 -s 8794 -d 8795 --n_gen 12 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --warm_up_path 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN'

# running
python ga_fuzzing.py -p 2024 -s 8800 -d 8801 --n_gen 12 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN'

# running
python ga_fuzzing.py -p 2021 -s 8798 -d 8799 --n_gen 12 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --warm_up_path 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_01_23_16_09,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

# running
python ga_fuzzing.py -p 2027 -s 8802 -d 8803 --n_gen 12 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_01_23_16_09,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'



#############################################################

python ga_fuzzing.py -p 2021 2024 -s 8796 -d 8797 --n_gen 2 --pop_size 2 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 6 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --warm_up_path 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN' --n_offsprings 20 --high_conf_num 5 --low_conf_num 5

#
python ga_fuzzing.py -p 2015 -s 8788 -d 8789 --n_gen 12 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN' --use_alternate_nn 1

#
python ga_fuzzing.py -p 2018 -s 8790 -d 8791 --n_gen 12 --pop_size 50 -r 'town01_left_0' -c 'turn_left_town01' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_01_23_16_09,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none' --use_alternate_nn 1

#
python ga_fuzzing.py -p 2021 -s 8792 -d 8793 --n_gen 12 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --warm_up_path 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_01_23_16_14,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

#
python ga_fuzzing.py -p 2024 -s 8794 -d 8795 --n_gen 12 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --warm_up_path 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_01_23_16_14,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

#
python ga_fuzzing.py -p 2027 -s 8796 -d 8797 --n_gen 12 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_01_23_16_14,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'


# TBD
python ga_fuzzing.py -p 2027 -s 8796 -d 8797 --n_gen 12 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_01_23_16_14,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none' --use_alternate_nn 1
# TBD
python ga_fuzzing.py -p 2024 -s 8794 -d 8795 --n_gen 12 --pop_size 50 -r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --warm_up_path 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_01_23_16_14,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none' --use_alternate_nn 1



# TBD
python ga_fuzzing.py -p 2015 -s 8788 -d 8789 --n_gen 12 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --warm_up_path 'run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/2021_02_01_00_45_18,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

# TBD
python ga_fuzzing.py -p 2018 -s 8790 -d 8791 --n_gen 12 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --warm_up_path 'run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/2021_02_01_00_45_18,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

# TBD
python ga_fuzzing.py -p 2021 -s 8792 -d 8793 --n_gen 12 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 600 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path 'run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/2021_02_01_00_45_18,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

#############################################################


python ga_fuzzing.py -p 2021 2024 -s 8796 -d 8797 --n_gen 3 --pop_size 2 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 6 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --warm_up_path 'run_results/random-un/town07_front_0/go_straight_town07/lbc/2021_02_07_14_59_37,100_1_none_100_300_1.01_-4_0.9_coeff_0_0.2_0.5__one_output_use_alternate_nn_0_none' --n_offsprings 20 --high_conf_num 5 --low_conf_num 5 --use_alternate_nn 1 --diversity_mode nn_rep


# initial seeds 27
python ga_fuzzing.py -p 2015 2018 -s 8788 -d 8789 --n_gen 1 --pop_size 100 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name random-un --has_run_num 100


# running adv_nn
python ga_fuzzing.py -p 2018 -s 8782 -d 8783 --n_gen 60 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 3000 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path 'run_results/random-un/town07_front_0/go_straight_town07/lbc/2021_02_07_14_59_37,100_1_none_100_300_1.01_-4_0.9_coeff_0_0.2_0.5__one_output_use_alternate_nn_0_none'

# running nn
python ga_fuzzing.py -p 2021 -s 8784 -d 8785 --n_gen 60 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 3000 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --warm_up_path 'run_results/random-un/town07_front_0/go_straight_town07/lbc/2021_02_07_14_59_37,100_1_none_100_300_1.01_-4_0.9_coeff_0_0.2_0.5__one_output_use_alternate_nn_0_none'

# running nsga2
python ga_fuzzing.py -p 2024 -s 8786 -d 8787 --n_gen 60 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 3000 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode none --warm_up_path 'run_results/random-un/town07_front_0/go_straight_town07/lbc/2021_02_07_14_59_37,100_1_none_100_300_1.01_-4_0.9_coeff_0_0.2_0.5__one_output_use_alternate_nn_0_none'

# running alternate_div
python ga_fuzzing.py -p 2027 -s 8788 -d 8789 --n_gen 60 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 3000 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --warm_up_path 'run_results/random-un/town07_front_0/go_straight_town07/lbc/2021_02_07_14_59_37,100_1_none_100_300_1.01_-4_0.9_coeff_0_0.2_0.5__one_output_use_alternate_nn_0_none' --use_alternate_nn 1 --diversity_mode nn_rep

# running alternate_adv_nn_div
python ga_fuzzing.py -p 2015 -s 8780 -d 8781 --n_gen 60 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 3000 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path 'run_results/random-un/town07_front_0/go_straight_town07/lbc/2021_02_07_14_59_37,100_1_none_100_300_1.01_-4_0.9_coeff_0_0.2_0.5__one_output_use_alternate_nn_0_none' --use_alternate_nn 1 --diversity_mode nn_rep



#############################################################




-r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num'

-r 'town07_front_0' -c 'go_straight_town07'
-r 'town01_left_0' -c 'turn_left_town01'

-r 'town04_front_0' -c 'pedestrians_cross_street_town04'
-r 'town03_front_1' -c 'change_lane_town03_fixed_npc_num'

-r 'town05_front_0' -c 'change_lane_town05_fixed_npc_num'




python ga_fuzzing.py -p 2015 2018 -s 8794 -d 8795 --n_gen 14 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty Random_none

python ga_fuzzing.py -p 2021 2024 -s 8796 -d 8797 --n_gen 14 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode nn --check_unique_coeff 0 0.1 0.5 --uncertainty Random_none







better sampling (uniform, probability based uniform)



uniqueness (use minimum between percentage and one for the field parameter)

can also check sparsity of found bugs / unique bugs as another measure of uniqueness



compare with ConAML in terms of the number of steps to get a better valid optimum point and final performance


adv attack method:
constraints: gradient descent, solve linear equations, VAE, extra head penalty backprop?
diversity: distance from found bugs, distance from start point, avg distance from existing bugs?


1.reread model inversion
2.come up with a generative model that considers bugginess, uniqueness, and constraints



visualization




problem has following characteristics:
expensive to run simulation
no access to derivatives
many local optimum (bugs)
a mixture of discrete variables and continuous variables
configs have constraints (and potentially conditional variables)
noise (simulations are not 100% reproducible)


want to find configs that have the following three properties with few queries:
bugginess:
ga (+nn, +adv nn)
conditional normalizing flow (w/ prior + bugginess as weight of weighted MLE) + adaptive sampling (natural evolutionary search, hamiltonian monte carlo, with objective reward - prob from conditional normalizing flow to trade-off exploration + exploitation)
RL (autoregressive Gaussian to model parameters + REINFORCE to optimize policy network with objective having reward for exploitation and entropy for exploration)
bayes optimization
vae
model inversion


uniqueness:
simple filtering
prob from a modeled probability distribution
self entropy (on the generated action)
curiosity
disagreement of ensemble
information gain


constraints:
simple filtering
extra branch violation penalty





tomorrow TBD:


random un baseline


check out their carla visualization repo, consider to limit the parameters to similar to their papers 3/4 such that can be plotted on a plane for visualization and potentially demonstration of our methods


try model inversion + entropy

maybe also some more active search methods



multi-objective VS single-objective

(multi-head) regression adv attack?

uniqueness cretirion







project back into the constraints after each backprop?





The goal of their usage of DNN is different:
1. they keep an archive A of best solutions (since their goal is for best pareto front). they ranke their A and newly generated generation P to reduce the number of simulations on P that has very small chance of being better than A. Instead, since our goal is to generate as many buggy configs as possible, we rank the newly generated generation P and choose the top k ones to run simulations.

2.we additionally use adversarial attack which was rarely used in the setting of mixed categorical and continuous type of data. We project the gradient to categorical fields by taking the ind of the maximum field corresponding category. We also only normalize the continuous fields.

* 3.further, we introduce an extra ensemble DNN fitting and adversarial attack to since we observe that the fitted DNNs tend to be not very stable.





regularization for DNN?


give a nsga2 multi-objective baseline VS current single-objective to show why single-objective makes more sense here.

reproduce DNN-SM baseline
modify DNN-DT as another baseline




2.1 cross-validation ensemble? ensemble using DNNs trained with different subsets?
2.2 study ensemble adv

2.3 use similarity between validation samples and next generation samples (using some embedding distance metric?) to determine the weights of each sub-network in the ensemble?

2.5 study nsga2-un thresholds






maybe a cross-validation run to decide automatically whether to use nn and adv nn or not?


3.try to improve regression_analysis performance of DNN on town_03_front


learn a representation for configs using the objectives signal such that the representation can be further used for interpretation / adv / exploration (a new uniqueness definition is needed)?




learn embed space with supervised (maybe even temporal) signals
+
clustering analysis on embedded space
+
instance-level interpretation via running simulation on perturbed instance (leveraging existing neighbor's running results) (LIME/SHAP) to get hypothesis on most influencing feature(s)
+
causal inference via intervention on most influencing feature(s) during simulation




for other instances we can interpolate explanation of cluster center (multiplied with their distance to each class center) and finetune with small number of extra perturbations




maybe also consider eps for one-hot embed to be 1 or let eps for embed dims and non-embed dims separate (otherwise they cannot be changed), check out effect








consider about (1) sorting and then adv (2) adv and then sorting


t-sne and results change across adv iterations

new scenario that has both types of errors

analyze found bugs distribution

fix some actors are not moving on some parts of some maps

make mutation process more customizable (trade off convergence of exploration; current binary tournament only make use of top elements 2 times which leads to very slow convergence)



1.2 make adv can be tuned to make one type of bug more likely (this needs to make the classification separate classes or 2 DNNs; also may be also consider regression???)


1.5 maybe modify standardization and adv attack ? (i.e. also standardize those one-hot encoded fields but record each field's value to be used during projection?)

2.2 analyze new nsga2 correlation between objective and if bug

2.5 improve nsga2 + DNN over nsga2



2.8 try adv nn for town05_front



3.improve adv in rerun to try to make it work (early stop to avoid overfitting (increasing test loss) -> make this automatic by checking validation loss elbow point) Also early stop adv attack at some point

3.3 improve either search strategy or definition. the current way is extremelly close to sampling randomly for some scenarios due to probability theorem.

3.5 feature importance + causal analysis through intervention for important fields? LIME/SHAP for individual bug config analysis and thus define uniqueness? cluster analysis and calculate+sort std of each field of each cluster to determine uniqueness? neighbor bugness?

4.try some heuristic from MTFuzz for adv to potentially produce better performance for this particular problem?


clustering for error type analysis? (what vectors used for clustering? feature vector of some DNNs?)

maybe consider unique bugs in terms of if the actor is inside the view of the ego car?

think about better way to process the "if condition" for waypoint_follower


5 try regression

5.8 change sklearn in ga_fuzzing to pytorch and integrate adv (applying adv attack on top ranked new configs) (if it shows some early promising results).

6.try different config for town_05_front to get more out-of-road error? Or maybe improve the objective?

6.3 let is_collision to be true only when a real collision bug happens
6.5 dynamic weight adjustment (normalize each objective from previous generation objectives)


7.ensemble idea (use validation for weighted average confidence?)?

modify DNN objective for different types of bugs (i.e. different weights for different bugs assigned)?
compare with random + DNN?

maybe also tsne of learned DNN feature vector?

1.carefully design the fields chosen (perturbation, categorical)
2.analyze important features using MtFuzz hotbytes method

3.rethink about search objectives as well as optimization loss

4.redesign/select scenarios to run for potentially better property

5.consider MtFuzz method of leveraging hotbytes for generating new inputs


Methods:
0.better objective function for NN?
1.better NN for better than random performance? figure out training loss decreases but testing loss increases; NN to rank with fine-grained signal?? (right now we only use binary and have not considered different types of bugs)
2.RankNet / learn to rank to rank???
3.Adv perturbation on generated cases
4.Inversion Model


Uniqueness
- cases where enough uniqueness exist and consider eliminate cases where duplicates exist??? (rank methods)
- adv with extra projection
- inversion model with extra projection / penalty

show our method is better in previous work def, then show our method works better in generalized cases

Fix:
1.randomness / reproducibility
2.better / increasing(decreasing) objective that more correlated with the bugs
3.continuous red traffic light loss???


Fixing:
1.show finetuning with buggy data from adv will make the model harder to be found bugs than finetuning with buggy data from random


Ex:
1.show bug that happens with 1 vehicle + 1 ped while removing either one won't happen to illustrate the extra bugs found


2.multiple run for statistical significance
3.make a video for better demo



# define new bug score
# if it is a bug, y = 0 + max_i ( f_i / f_i_range over i that is not the bug's cause)
# else: y = 1 + max_i ( f_i / f_i_range over all i)
# maybe consider to improve the quality of score by redefining some such that
# nsga2 can indeed improve these scores / find more bugs over time



# DNN architecture
more examples for training? higher threshold and stop_conf?
another scenario?
also draw train examples on tsne graphs
read paper for better sampling?

"""
import sys
import os

sys.path.append("pymoo")
carla_root = "../carla_0994_no_rss"
sys.path.append(carla_root + "/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg")
sys.path.append(carla_root + "/PythonAPI/carla")
sys.path.append(carla_root + "/PythonAPI")
sys.path.append(".")
sys.path.append("leaderboard")
sys.path.append("leaderboard/team_code")
sys.path.append("scenario_runner")
sys.path.append("scenario_runner")
sys.path.append("carla_project")
sys.path.append("carla_project/src")


import pickle
import re

import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier


from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

from customized_utils import (
    encode_fields,
    decode_fields,
    remove_fields_not_changing,
    recover_fields_not_changing,
    get_labels_to_encode,
    customized_standardize,
    customized_fit,
    load_data,
    get_sorted_subfolders,
    get_picklename,
    get_distinct_data_points,
    calculate_rep_d
)


from pgd_attack import VanillaDataset


class NN_EnsembleClassifier:
    def __init__(self, num_of_nets=1):
        self.num_of_nets = num_of_nets
        self.nets = []
        for _ in range(self.num_of_nets):
            net = MLPClassifier(solver="lbfgs", activation="tanh", max_iter=10000)
            self.nets.append(net)

    def fit(self, X_train, y_train):
        for i in range(self.num_of_nets):
            net = self.nets[i]
            net.fit(X_train, y_train)

    def score(self, X_test, y_test):
        s_list = []
        for net in self.nets:
            s = net.score(X_test, y_test)
            s_list.append(s)
        s_np = np.array(s_list)
        return np.mean(s_np)

    def predict(self, X_test):
        s_list = []
        for net in self.nets:
            s = net.predict(X_test)
            s_list.append(s)
        s_np = np.array(s_list)
        # print(s_np)
        prediction = stats.mode(s_np, axis=0)[0][0]
        # print(prediction)
        return prediction

    def predict_proba(self, X_test):
        s_list = []
        for net in self.nets:
            s = net.predict_proba(X_test)
            s_list.append(s)
        s_np = np.array(s_list)
        # print(s_np)
        prediction = np.mean(s_np, axis=0)
        # print(prediction)
        return prediction


def regression_analysis(
    X, is_bug_list, objective_list, cutoff, cutoff_end, trial_num, encoded_fields
):

    # from matplotlib import pyplot as plt
    # plt.hist(objective_list[:, 1])
    # plt.show()

    # [0, 1, 2]
    # 0: 150, 3.07 VS 6.59
    # 1: 150, 9.84 VS 17.52
    # 2: 150 and 3, 0.004 VS 0.003
    ind = 2

    print(np.mean(objective_list[:, ind]), np.median(objective_list[:, ind]))

    y = objective_list[:, ind]

    X_train, X_test = X[:cutoff], X[cutoff:cutoff_end]
    y_train, y_test = y[:cutoff], y[cutoff:cutoff_end]
    standardize = StandardScaler()

    one_hot_fields_len = len(encoded_fields)

    customized_fit(X_train, standardize, one_hot_fields_len, partial=True)
    X_train = customized_standardize(
        X_train, standardize, one_hot_fields_len, partial=True
    )
    X_test = customized_standardize(
        X_test, standardize, one_hot_fields_len, partial=True
    )

    names = ["Neural Net", "Random"]

    from pgd_attack import train_regression_net
    from sklearn.metrics import mean_squared_error

    classifiers = [None, None]

    performance = {name: [] for name in names}

    for i in range(trial_num):
        print(i)
        for name, clf in zip(names, classifiers):
            if name == "Neural Net":
                clf = train_regression_net(
                    X_train, y_train, X_test, y_test, hidden_layer_size=3
                )
                y_pred = clf.predict(X_test).squeeze()
                mse = mean_squared_error(y_test, y_pred)
                # print(y_test, y_pred)
                print(f"{name}, mse:{mse:.3f}")
            elif name == "Random":
                y_pred = np.random.uniform(
                    np.min(y_train), np.max(y_train), len(y_test)
                )
                mse = mean_squared_error(y_test, y_pred)
                # print(y_test, y_pred)
                print(f"{name}, mse:{mse:.3f}")
            performance[name].append(mse)

    for name in names:
        print(name, np.mean(performance[name]), np.std(performance[name]))


def classification_analysis(
    X, is_bug_list, objective_list, cutoff, cutoff_end, trial_num, encoded_fields
):

    # from matplotlib import pyplot as plt
    # plt.hist(objective_list[:, 1])
    # plt.show()
    print(np.mean(objective_list[:, 1]), np.median(objective_list[:, 1]))

    y = is_bug_list
    print(np.sum(is_bug_list), np.sum(objective_list[:, -1] == 1))
    print(X.shape, y.shape)

    X_train, X_test = X[:cutoff], X[cutoff:cutoff_end]
    y_train, y_test = y[:cutoff], y[cutoff:cutoff_end]
    standardize = StandardScaler()

    one_hot_fields_len = len(encoded_fields)

    customized_fit(X_train, standardize, one_hot_fields_len, partial=True)
    X_train = customized_standardize(
        X_train, standardize, one_hot_fields_len, partial=True
    )
    X_test = customized_standardize(
        X_test, standardize, one_hot_fields_len, partial=True
    )

    print("y_test", y_test)

    ind_0 = y_test == 0
    ind_1 = y_test == 1

    print(
        np.sum(y_train < 0.5),
        np.sum(y_train > 0.5),
        np.sum(y_test < 0.5),
        np.sum(y_test > 0.5),
    )

    names = ["Nearest Neighbors", "Neural Net", "AdaBoost", "Random", "NN ensemble"]

    classifiers = [
        KNeighborsClassifier(5),
        MLPClassifier(
            hidden_layer_sizes=[150], solver="lbfgs", activation="tanh", max_iter=10000
        ),
        AdaBoostClassifier(),
        DummyClassifier(strategy="stratified"),
        NN_EnsembleClassifier(5),
    ]

    performance = {name: [] for name in names}
    from sklearn.metrics import roc_auc_score

    # ['sklearn', 'pytorch']
    dnn_lib = "pytorch"

    for i in range(trial_num):
        print(i)
        for name, clf in zip(names, classifiers):
            if name == "Neural Net" and dnn_lib == "pytorch":
                from pgd_attack import train_net

                clf = train_net(
                    X_train, y_train, [], [], batch_train=64, model_type="one_output", num_epochs=30
                )
                y_pred = clf.predict(X_test).squeeze()
                prob = clf.predict_proba(X_test)[:, 1]
            else:
                clf.fit(X_train, y_train)
                # score = clf.score(X_test, y_test)
                y_pred = clf.predict(X_test)
                prob = clf.predict_proba(X_test)[:, 1]

            t = y_test == 1
            p = y_pred == 1
            tp = t & p
            # print(name, dnn_lib)
            # print(t, p, tp, y_pred)
            precision = np.sum(tp) / np.sum(p)
            recall = np.sum(tp) / np.sum(t)
            f1 = 2 * precision * recall / (precision + recall)
            roc_auc = roc_auc_score(y_test, prob)

            top_50_inds = np.argsort(prob*-1)[:50]
            top_50_bugs_num = np.mean(y_test[top_50_inds]==1)*50

            print(
                f"{name}, top_50_bugs_num:{top_50_bugs_num:.3f}, roc_auc_score:{roc_auc:.3f}; f1: {f1:.3f}; precision: {precision:.3f}; recall: {recall:.3f}"
            )
            performance[name].append(top_50_bugs_num)

            from customized_utils import draw_auc_roc_for_scores

            # draw_auc_roc_for_scores(-1*prob, y_test)

    for name in names:
        print(name, np.mean(performance[name]), np.std(performance[name]))


def active_learning(
    X,
    is_bug_list,
    objective_list,
    cutoff,
    cutoff_mid,
    cutoff_end,
    trial_num,
    encoded_fields,
    retrain_num,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import transforms
    from copy import deepcopy
    from matplotlib import pyplot as plt
    from pgd_attack import train_net
    from acquisition import move_data, BALD, Random, BatchBALD, BUGCONF, BADGE

    acquisition_batch_size = 50
    lr = 3e-4
    num_train = cutoff
    num_pool = cutoff_mid - cutoff
    train_batch_size = 60
    test_batch_size = 20

    def train(model, device, train_loader, optimizer, epoch_num):
        model.train()
        target_list = []
        for epoch in range(epoch_num):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device).float(), target.to(device).float()
                optimizer.zero_grad()
                output = model(data, return_logits=True).squeeze()
                # print(output, target)
                loss = F.binary_cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )
                if epoch == 0:
                    target_list.append(target.cpu().detach().numpy())
            if epoch == 0:
                target_list = np.concatenate(target_list)
                print("train bug num", np.sum(target_list > 0))
        return np.sum(target_list > 0)

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0

        target_list = []
        output_one_hot_list = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device).float(), target.to(device).float()
                output = model(data, return_logits=True).squeeze()
                test_loss += F.binary_cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss

                output_one_hot = model.predict_proba(data)
                pred = output_one_hot.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                target_list.append(target.cpu().detach().numpy())
                output_one_hot_list.append(output_one_hot[:, 1].cpu().detach().numpy())

        test_loss /= len(test_loader.dataset)
        accuracy = float(correct) / len(test_loader.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100.0 * accuracy
            )
        )

        target_list = np.concatenate(target_list)
        output_one_hot_list = np.concatenate(output_one_hot_list)

        ranks = np.argsort(output_one_hot_list * -1)
        print("rank", np.mean(ranks[target_list == 1]))
        print("\n")
        return accuracy

    def active(model, acquirer, device, data, optimizer):
        train_data, pool_data, test_data = data
        epoch_num = 30
        test_accuracies = []
        train_bug_nums = []
        from pgd_attack import BNN

        while len(pool_data) > 0:
            model = BNN(train_data[0][0].size()[0], 1)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            print(
                f"Acquiring {acquirer.__class__.__name__} batch. Pool size: {len(pool_data)}"
            )
            # get the indices of the best batch of data
            batch_indices = acquirer.select_batch(
                model, pool_data, uncertainty_conf=True
            )
            # move that data from the pool to the training set
            move_data(batch_indices, pool_data, train_data)
            # train on it
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=train_batch_size, pin_memory=True, shuffle=True
            )
            train_bug_nums.append(
                train(model, device, train_loader, optimizer, epoch_num)
            )

            # test the accuracy
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=test_batch_size, pin_memory=True, shuffle=True
            )
            test_accuracies.append(test(model, device, test_loader))

        return test_accuracies, train_bug_nums

    y = np.array([float(obj[0] > 0.1) for obj in objective_list])

    X_train, X_extra, X_test = (
        X[:cutoff],
        X[cutoff:cutoff_mid],
        X[cutoff_mid:cutoff_end],
    )
    y_train, y_extra, y_test = (
        y[:cutoff],
        y[cutoff:cutoff_mid],
        y[cutoff_mid:cutoff_end],
    )

    standardize = StandardScaler()
    one_hot_fields_len = len(encoded_fields)
    customized_fit(X_train, standardize, one_hot_fields_len, partial=True)
    X_train = customized_standardize(
        X_train, standardize, one_hot_fields_len, partial=True
    )
    X_extra = customized_standardize(
        X_extra, standardize, one_hot_fields_len, partial=True
    )
    X_test = customized_standardize(
        X_test, standardize, one_hot_fields_len, partial=True
    )

    model = train_net(
        X_train,
        y_train,
        X_test,
        y_test,
        batch_train=64,
        hidden_size=20,
        model_type="BNN",
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load the dataset and pre-process
    dataset = VanillaDataset(
        np.concatenate([X_train, X_extra]),
        np.concatenate([y_train, y_extra]),
        to_tensor=True,
    )

    # subset_indices = np.random.choice(len(dataset), size=num_train+num_pool, replace=False)

    subset_indices = np.arange(len(dataset))
    train_indices = subset_indices[:num_train]
    pool_indices = subset_indices[-num_pool:]
    train_data = torch.utils.data.Subset(dataset, train_indices)

    test_data = VanillaDataset(X_test, y_test, to_tensor=True)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("test accuracies and train bug numbers")

    pre_acquisition_model_state = model.state_dict()

    for acquisition_strategy in [BADGE, BALD, BatchBALD, BUGCONF, Random]:
        # reset the model
        model.load_state_dict(deepcopy(pre_acquisition_model_state))
        # init the acquirer
        acquirer = acquisition_strategy(acquisition_batch_size, device)
        # and an optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # get all the data
        train_data = torch.utils.data.Subset(dataset, train_indices)
        pool_data = torch.utils.data.Subset(dataset, pool_indices)
        data = (train_data, pool_data, test_data)
        # train the model with active learning
        accuracies, train_bug_nums = active(model, acquirer, device, data, optimizer)

        ax1.plot(train_bug_nums, label=acquisition_strategy.__name__)
        ax2.plot(accuracies, label=acquisition_strategy.__name__)

    plt.legend()
    plt.show()

# def train_clf(X, objective_list, cutoff, cutoff_end, encoded_fields, clf_save_path=None):
#     y = np.array([obj[0] > 0.1 for obj in objective_list])
#
#     X_train = X[cutoff:cutoff_end]
#     y_train = y[cutoff:cutoff_end]
#
#     X_combined = np.concatenate([X_train, X_test])
#     y_combined = np.concatenate([y_train, y_test])
#
#     standardize = StandardScaler()
#     one_hot_fields_len = len(encoded_fields)
#     customized_fit(X_train, standardize, one_hot_fields_len, partial=True)
#     X_train = customized_standardize(
#         X_train, standardize, one_hot_fields_len, partial=True
#     )
#     X_test = customized_standardize(
#         X_test, standardize, one_hot_fields_len, partial=True
#     )
#
#     from pgd_attack import train_net
#     import torch
#     clf = train_net(X_train, y_train, X_test, y_test, model_type="one_output")
#     torch.save(clf.state_dict(), clf_save_path)



def compute_diversity(X, objective_list, cutoff, cutoff_end, encoded_fields,  pretrained_clf=False, clf_save_path=None):
    # measure diversity (avg min dist) of alternate_nn_div, alternate_nn, nsga2

    y = np.array([obj[0] > 0.1 for obj in objective_list])

    X_train = X[:cutoff]
    y_train = y[:cutoff]
    X_test = X[cutoff:cutoff_end]
    y_test = y[cutoff:cutoff_end]
    X_combined = np.concatenate([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])

    standardize = StandardScaler()
    one_hot_fields_len = len(encoded_fields)
    customized_fit(X_train, standardize, one_hot_fields_len, partial=True)
    X_train = customized_standardize(
        X_train, standardize, one_hot_fields_len, partial=True
    )
    X_test = customized_standardize(
        X_test, standardize, one_hot_fields_len, partial=True
    )

    from pgd_attack import train_net, SimpleNet
    import torch
    if pretrained_clf:
        input_size, hidden_size, num_classes = X_train.shape[1], 150, 1
        clf = SimpleNet(input_size, hidden_size, num_classes)
        clf.load_state_dict(torch.load(clf_save_path))
        clf.eval()
        clf.cuda()
    else:
        clf = train_net(X_train, y_train, X_test, y_test, model_type="one_output")
        torch.save(clf.state_dict(), clf_save_path)

    d_list = calculate_rep_d(clf, X_train, X_test)
    print('d_list', d_list.shape, np.sort(d_list, axis=1))
    print('d_list min', np.sort(d_list, axis=1)[:, 1])
    print('d_list avg min', np.mean(np.sort(d_list, axis=1)[:, 1]))



def draw_tsne(X, is_bug_list, objective_list, cutoff, cutoff_ends_start, cutoff_ends_end, encoded_fields, pretrained_clf=False, clf_save_path=None):

    y = np.array([obj[0] > 0.1 for obj in objective_list])

    X_train = X[:cutoff]
    y_train = y[:cutoff]
    X_test = X[cutoff:cutoff_ends_end[-1]]
    y_test = y[cutoff:cutoff_ends_end[-1]]
    X_combined = np.concatenate([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])

    standardize = StandardScaler()
    one_hot_fields_len = len(encoded_fields)
    customized_fit(X_train, standardize, one_hot_fields_len, partial=True)
    X_train = customized_standardize(
        X_train, standardize, one_hot_fields_len, partial=True
    )
    X_test = customized_standardize(
        X_test, standardize, one_hot_fields_len, partial=True
    )

    from pgd_attack import train_net, SimpleNet
    import torch
    if pretrained_clf:
        input_size, hidden_size, num_classes = X_train.shape[1], 150, 1
        clf = SimpleNet(input_size, hidden_size, num_classes)
        clf.load_state_dict(torch.load(clf_save_path))
        clf.eval()
        clf.cuda()
    else:
        clf = train_net(X_train, y_train, X_test, y_test, model_type="one_output")
        torch.save(clf.state_dict(), clf_save_path)


    y_pred = clf.predict(X_combined).squeeze()
    prob_pred = clf.predict_proba(X_combined)[:, 1]
    print('prob_pred', prob_pred)
    embed = clf.extract_embed(X_combined)

    train_inds = np.arange(0, cutoff)
    test_inds1 = np.arange(cutoff_ends_start[0], cutoff_ends_end[0])
    test_inds2 = np.arange(cutoff_ends_start[1], cutoff_ends_end[1])
    test_inds3 = np.arange(cutoff_ends_start[2], cutoff_ends_end[2])


    y_test_pred = clf.predict(X_test).squeeze()
    test_prob_pred = clf.predict_proba(X_test)[:, 1]
    test_ind_0 = y_test == 0
    test_ind_1 = y_test == 1

    pred_ind_0 = y_test_pred == 0
    pred_ind_1 = y_test_pred == 1

    print("test_ind_0", np.sum(test_ind_0))
    print("test_ind_1", np.sum(test_ind_1))
    print("pred_ind_0", np.sum(pred_ind_0))
    print("pred_ind_1", np.sum(pred_ind_1))
    print("TP", np.sum(test_ind_1 & pred_ind_1))
    print("FP", np.sum(test_ind_0 & pred_ind_1))
    print("TN", np.sum(test_ind_0 & pred_ind_0))
    print("FN", np.sum(test_ind_1 & pred_ind_0))
    from scipy.stats import rankdata

    test_prob_rank = rankdata(test_prob_pred)
    test_prob_FN = test_prob_pred[test_ind_1 & pred_ind_0]
    test_prob_rank_FN = test_prob_rank[test_ind_1 & pred_ind_0]

    print("test_prob_FN", test_prob_FN)
    print("test_prob_rank_FN", test_prob_rank_FN)

    print("test_prob[test_ind_1]", test_prob_pred[test_ind_1])
    print("test_prob_rank[test_ind_1]", test_prob_rank[test_ind_1])
    print("np.mean(test_prob_rank[test_ind_1])", np.mean(test_prob_rank[test_ind_1]))

    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt

    X_embed = TSNE(n_components=2, perplexity=30.0, n_iter=3000).fit_transform(embed)


    inds = [train_inds, test_inds1, test_inds2, test_inds3]
    colors = ['Greys','Purples', 'Blues', 'Greens']
    colors = ['grey','purple', 'blue', 'green']

    for j, (ids, cs) in enumerate(zip(inds, colors)):
        X_embed_ids, y_ids, prob_pred_ids = X_embed[ids], y[ids], prob_pred[ids]
        # print('ids', ids)
        print('y_ids.shape', y_ids.shape)
        ids_0 = y_ids == 0
        ids_1 = y_ids == 1
        X_embed_ids_0, prob_pred_ids_0 = X_embed_ids[ids_0], prob_pred_ids[ids_0]
        X_embed_ids_1, prob_pred_ids_1 = X_embed_ids[ids_1], prob_pred_ids[ids_1]
        print(j, 'prob_pred_ids_0', prob_pred_ids_0)
        print(j, 'prob_pred_ids_1', prob_pred_ids_1)
        plt.scatter(
            X_embed_ids_0[:, 0],
            X_embed_ids_0[:, 1],
            c=cs,
            label="normal",
            alpha=0.5,
            s=20,
            marker="."
        )
        plt.scatter(
            X_embed_ids_1[:, 0],
            X_embed_ids_1[:, 1],
            c=cs,
            label="bug",
            alpha=0.5,
            s=20,
            marker="^"
        )

        # plt.scatter(
        #     X_embed_ids_0[:, 0],
        #     X_embed_ids_0[:, 1],
        #     c=prob_pred_ids_0,
        #     label="normal",
        #     alpha=0.5,
        #     s=20,
        #     marker=".",
        #     cmap=cs,
        # )
        # plt.scatter(
        #     X_embed_ids_1[:, 0],
        #     X_embed_ids_1[:, 1],
        #     c=prob_pred_ids_1,
        #     label="bug",
        #     alpha=0.5,
        #     s=20,
        #     marker="^",
        #     cmap=cs,
        # )

    plt.legend(loc=2, prop={"size": 10}, framealpha=0.5)
    plt.savefig("tmp_tsne/tsne_confidence.pdf")


def encode_and_remove_x(data_list, mask, labels):
    # town_05_right
    labels_to_encode = get_labels_to_encode(labels)

    x, enc, inds_to_encode, inds_non_encode, encoded_fields = encode_fields(
        data_list, labels, labels_to_encode
    )

    one_hot_fields_len = len(encoded_fields)
    x, x_removed, kept_fields, removed_fields = remove_fields_not_changing(
        x, one_hot_fields_len
    )

    return x, encoded_fields


def analyze_objective_data(X, is_bug_list, objective_list):
    from matplotlib import pyplot as plt

    mode = "tsne_input"

    if mode == "hist":
        ind = -2
        ind2 = 1
        print(np.sum(objective_list[:, ind]))

        cond1 = (is_bug_list == 1) & (objective_list[:, ind] == 1)
        cond2 = (is_bug_list == 0) & (objective_list[:, ind] == 0)

        print(np.where(cond1 == 1))
        print(objective_list[cond1, ind2])
        print(objective_list[cond2, ind2])

        plt.hist(objective_list[cond1, ind2], label="bug", alpha=0.5, bins=50)
        plt.hist(objective_list[cond2, ind2], label="normal", alpha=0.5, bins=100)
        plt.legend()
        plt.show()
    elif mode == "tsne_input":
        from sklearn.manifold import TSNE

        X_embedded = TSNE(n_components=2, perplexity=5, n_iter=3000).fit_transform(X)
        y = np.array(is_bug_list)
        ind0 = y == 0
        ind1 = y == 1
        plt.scatter(
            X_embedded[ind0, 0], X_embedded[ind0, 1], label="normal", alpha=0.5, s=3
        )
        plt.scatter(
            X_embedded[ind1, 0], X_embedded[ind1, 1], label="bug", alpha=0.5, s=5
        )
        plt.legend()
        plt.show()


def union_analysis(parent_folder, parent_folder2):
    subfolders = get_sorted_subfolders(parent_folder)
    X, is_bug_list, objective_list, mask, labels = load_data(subfolders)

    pickle_filename = get_picklename(parent_folder)
    with open(pickle_filename, 'rb') as f_in:
        d = pickle.load(f_in)
    xl = d['xl']
    xu = d['xu']
    mask = d['mask']
    p = 0
    c = 0.1
    th = 0.5

    subfolders2 = get_sorted_subfolders(parent_folder2)
    X2, is_bug_list2, objective_list2, mask2, labels2 = load_data(subfolders2)

    X = np.array(X)[300:450]
    X2 = np.array(X2)[300:450]
    objective_list = objective_list[300:450]
    objective_list2 = objective_list2[300:450]

    X_bug = X[objective_list[:, 0]>0.1]
    X2_bug = X2[objective_list2[:, 0]>0.1]


    unique_specific_bugs, specific_distinct_inds = get_distinct_data_points(
        np.concatenate([X_bug, X2_bug]), mask, xl, xu, p, c, th
    )

    print(len(X_bug), len(X2_bug), len(unique_specific_bugs))



if __name__ == "__main__":
    # ['analysis', 'tsne', 'discrete', 'active', 'intersection']
    mode = 'compute_diversity'
    trial_num = 15
    cutoff = 700
    cutoff_end = 1300
    cutoff_ends_start = [700, 750, 800]
    cutoff_ends_end = [750, 800, 850]

    parent_folder = 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_07_00_12_47,50_12_adv_nn_600_300_1.01_-4_0.9_coeff_0_0.1_0.5__one_output_use_alternate_nn_0'

    parent_folder2 = '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN'

    warm_up_path = 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_01_23_16_14,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'

    cutoff_mid = 500
    retrain_num = 50

    pretrained_clf = True
    clf_save_path = 'tmp_tsne/tmp_model_town03'


    # warm up town07
    # d_list avg min 2.7438152 213 2.712
    # 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_06_15_58_00,50_12_alternate_nn_600_300_1.01_-4_0.75_coeff_0_0.1_0.5__one_output'
    # d_list avg min 2.7282553 209 2.677
    # 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_06_00_22_40,50_12_none_600_300_1.01_-4_0.75_coeff_0_0.1_0.5__one_output'
    # d_list avg min 2.721047 189 2.667
    # 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_06_00_22_46,50_12_alternate_nn_600_300_1.01_-4_0.75_coeff_0_0.1_0.5__one_output'
    # 244 2.468
    # 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_06_19_07_56,50_12_nn_600_300_1.01_-4_0.8_coeff_0_0.1_0.5__one_output'
    # 263 2.347
    # 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_06_19_08_05,50_12_adv_nn_600_300_1.01_-4_0.8_coeff_0_0.1_0.5__one_output'
    # 2.552 250
    # 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_07_00_12_27,50_12_adv_nn_600_300_1.01_-4_0.9_coeff_0_0.1_0.5__one_output_use_alternate_nn_1'
    #
    # warm_up_path = 'run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN'
    #
    #
    # warm up town01
    # d_list avg min 2.6244676 448
    # 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_06_15_57_43,50_12_alternate_nn_600_300_1.01_-4_0.75_coeff_0_0.1_0.5__one_output'
    # d_list avg min 2.5094328 441
    # 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_06_00_22_50,50_12_none_600_300_1.01_-4_0.75_coeff_0_0.1_0.5__one_output'
    # d_list avg min 2.4661999 371
    # 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_06_00_22_56,50_12_alternate_nn_600_300_1.01_-4_0.75_coeff_0_0.1_0.5__one_output'
    # d_list avg min 2.318 572
    # 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_06_19_08_11,50_12_nn_600_300_1.01_-4_0.8_coeff_0_0.1_0.5__one_output'
    # d_list avg min 2.304 529
    # 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_06_19_08_18,50_12_adv_nn_600_300_1.01_-4_0.8_coeff_0_0.1_0.5__one_output'
    # 2.652 463
    # 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_07_00_12_33,50_12_adv_nn_600_300_1.01_-4_0.9_coeff_0_0.1_0.5__one_output_use_alternate_nn_1'
    #
    # warm_up_path = 'run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_01_23_16_09,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'
    #
    #
    # warm up town03
    #
    # 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_07_00_12_38,50_12_none_600_300_1.01_-4_0.9_coeff_0_0.1_0.5__one_output_use_alternate_nn_0'
    #
    # 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_07_00_12_43,50_12_nn_600_300_1.01_-4_0.9_coeff_0_0.1_0.5__one_output_use_alternate_nn_0'
    #
    # 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_07_00_12_47,50_12_adv_nn_600_300_1.01_-4_0.9_coeff_0_0.1_0.5__one_output_use_alternate_nn_0'
    #
    # warm_up_path = 'run_results/nsga2-un/town03_front_1/change_lane_town03_fixed_npc_num/lbc/2021_02_01_23_16_14,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'










    # tsne town07 alternate_nn
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_04_18_18_39,50_14_alternate_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5__one_output'
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2-un/town07_front_0/go_straight_town07/lbc/2021_02_03_10_27_28,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none_BNN'

    # tsne town01 alternate_nn
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_04_18_18_52,50_14_alternate_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5__one_output'
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_01_23_16_09,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'



    # tsne town05 BADGE
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/2021_02_01_00_45_18,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2-un/town05_right_0/leading_car_braking_town05_fixed_npc_num/lbc/2021_02_02_23_43_12,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_BADGE_none_one_output'

    # tsne town01 BADGE
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_01_23_16_09,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_Random_none'
    # '/home/zhongzzy9/Documents/self-driving-car/2020_CARLA_challenge/run_results/nsga2-un/town01_left_0/turn_left_town01/lbc/2021_02_02_23_43_22,50_14_nn_700_300_1.01_-4_0.75_coeff_0.0_0.1_0.5_BADGE_none_one_output'


    # tsne town01 random



    subfolders = get_sorted_subfolders(parent_folder)
    X, is_bug_list, objective_list, mask, labels = load_data(subfolders)

    if warm_up_path:
        subfolders = get_sorted_subfolders(warm_up_path)
        X_pre, _, objective_list_pre, _, _ = load_data(subfolders)
        X = np.concatenate([X_pre, X])
        objective_list = np.concatenate([objective_list_pre, objective_list])

    X, encoded_fields = encode_and_remove_x(X, mask, labels)



    if mode == "analysis":
        analyze_objective_data(X, is_bug_list, objective_list)
    elif mode == "tsne":
        draw_tsne(X, is_bug_list, objective_list, cutoff, cutoff_ends_start, cutoff_ends_end, encoded_fields, pretrained_clf=pretrained_clf, clf_save_path=clf_save_path)
    elif mode == "discrete":
        classification_analysis(
            X,
            is_bug_list,
            objective_list,
            cutoff,
            cutoff_end,
            trial_num,
            encoded_fields,
        )
    elif mode == "active":
        active_learning(
            X,
            is_bug_list,
            objective_list,
            cutoff,
            cutoff_mid,
            cutoff_end,
            trial_num,
            encoded_fields,
            retrain_num,
        )
    elif mode == "regression":
        regression_analysis(
            X,
            is_bug_list,
            objective_list,
            cutoff,
            cutoff_end,
            trial_num,
            encoded_fields,
        )
    elif mode == "intersection":
        union_analysis(parent_folder, parent_folder2)
    elif mode == "compute_diversity":
        compute_diversity(X, objective_list, cutoff, cutoff_end, encoded_fields,  pretrained_clf=pretrained_clf, clf_save_path=clf_save_path)

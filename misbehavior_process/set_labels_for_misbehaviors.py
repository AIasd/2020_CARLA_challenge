import os
import numpy as np
import pandas as pd

infraction_types = ['collisions_layout', 'collisions_pedestrian', 'collisions_vehicle', 'wrong_lane', 'off_road', 'red_light']
infraction_ind_typ = {typ:i+1 for i, typ in enumerate(infraction_types)}

NORMAL_LABEL = "normal"

LABEL_GAP = "gap"

ANOMALY_LABEL = "anomaly"

REACTION_LABEL = "reaction"

IGNORE_END_OF_STREAM_LABEL = "ignore_end_of_stream"

HEALING_LABEL = "healing"

MISBEHAVIOR_LABEL = "misbehavior"

# scale down by multiplied with 1/10
REACTION_TIME = 10 # 50->10
ANOMALY_WINDOW_LENGTH = 10 # 30->10
NORMAL_WINDOW_LENGTH = ANOMALY_WINDOW_LENGTH # 30->10
HEALING_TIME = 0 # 60->0
MAX_CUT_END_LENGTH = REACTION_TIME + ANOMALY_WINDOW_LENGTH


NORMAL_IND = 0
REACTION_IND = -1
MISBEHAVIOR_IND = -2
IGNORE_END_OF_STREAM_IND = -3
LABEL_GAP_IND = -4


route_folder = 'collected_data_customized/customized_auto_pilot_nodebug'
route_files = os.listdir(route_folder)

behaviors = []
behaviors_names = []

for route_file in sorted(route_files):
    route_path = os.path.join(route_folder, route_file)
    if os.path.isdir(route_path):
        driving_log = os.path.join(route_path, 'driving_log.csv')
        data_df = pd.read_csv(driving_log)

        n = len(data_df['Misbehavior'])

        behaviors = []
        behaviors_names = []
        cur_start_ind = 0
        cur_end_ind = n

        misbehavior_ind = 1



        for i, m in enumerate(data_df['Misbehavior']):
            if not pd.isnull(m):
                print(m)
                for infraction_type in infraction_types:
                    if infraction_type in m:
                        cur_start_ind = min([i+1, n-1])
                        misbehavior_ind = infraction_ind_typ[infraction_type]
                        break
        # if there is no violation
        if cur_start_ind == 0:
            for j in range(cur_start_ind, cur_end_ind):
                behaviors_names.append(NORMAL_LABEL)
                behaviors.append(0)
        else:
            for j in range(cur_start_ind, cur_end_ind):
                behaviors_names.append(IGNORE_END_OF_STREAM_LABEL)
                behaviors.append(IGNORE_END_OF_STREAM_IND)
            cur_end_ind = cur_start_ind
            behaviors_names.append(MISBEHAVIOR_LABEL)
            behaviors.append(MISBEHAVIOR_IND)
            cur_end_ind -= 1

            label_frame_length_ind = [(REACTION_LABEL, REACTION_TIME, REACTION_IND), (ANOMALY_LABEL, ANOMALY_WINDOW_LENGTH, misbehavior_ind), (NORMAL_LABEL, NORMAL_WINDOW_LENGTH, NORMAL_IND)]


            for label, frame_length, ind in label_frame_length_ind:
                if label == NORMAL_LABEL:
                    cur_start_ind = cur_end_ind % frame_length
                else:
                    cur_start_ind = np.max([0, cur_end_ind - frame_length])
                for i in range(cur_start_ind, cur_end_ind):
                    behaviors_names.append(label)
                    behaviors.append(ind)
                cur_end_ind = cur_start_ind
                if cur_end_ind == 0:
                    break
            for j in range(cur_end_ind):
                behaviors_names.append(LABEL_GAP)
                behaviors.append(LABEL_GAP_IND)

        assert len(behaviors_names) == n, str(behaviors_names)+' VS '+str(n)
        data_df['behaviors_names'] = behaviors_names[::-1]
        data_df['behaviors'] = behaviors[::-1]
        print(route_file, n)
        data_df.to_csv(driving_log, index=False)

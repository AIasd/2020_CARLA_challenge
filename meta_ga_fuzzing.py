'''
sudo -E /home/zhongzzy9/anaconda3/envs/carla99/bin/python meta_ga_fuzzing.py
'''

import os
import time



num_of_epochs = 5

for i in range(num_of_epochs):
    if i > 0:
        resume_str = ' --resume-run'
    else:
        resume_str = ''
    os.system('sudo -E /home/zhongzzy9/anaconda3/envs/carla99/bin/python ga_fuzzing.py'+' --ind='+str(i)+resume_str)
    time.sleep(1)

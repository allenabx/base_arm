# @Author   : Chen Mingyang
# @Time     : 2018/11/20
# @FileName : params.py

import numpy as np


CUDA = False
'''
BATCH_SIZE = 128
EPISODE = 10000
LR = 0.0001 # learning rate
EPSILON = 0.1
EPSILON_MAX = 0.8       # greedy policy
EPSILON_INCREASE = 0.0001
GAMMA = 0.999                 # reward discount
TARGET_REPLACE_ITER = 20   # target update frequency
MEMORY_CAPACITY = 50000
MAX_STEP = 1000

N_JOINTS = 4
N_ACTIONS_Per_JOINT = 2
ACTIONS_Per_JOINT = [-1, 1]   # action list for each joint
N_ACTIONS = N_JOINTS * N_ACTIONS_Per_JOINT
N_STATES = 3 +N_JOINTS +3                # the states is position here, like (x, y, z)

PRECISION = 100             # precision for judging if done



def randAng():
    a = np.random.randint(-120, 120)
    b = np.random.randint(-95, 0)
    c = np.random.randint(-120, 120)
    d = np.random.randint(-1, 90)

    return [a, b, c, d]
'''

BATCH_SIZE = 128
EPISODE = 10000
LR = 0.0001 # learning rate

EPSILON = 0.1
EPSILON_MAX = 0.95       # greedy policy
EPSILON_INCREASE = 0.000001

# EPSILON = 1
# EPSILON_MAX = 1       # greedy policy
# EPSILON_INCREASE = None

GAMMA = 0.95              # reward discount
TARGET_REPLACE_ITER = 20   # target update frequency
MEMORY_CAPACITY = 100000
MAX_STEP = 500

N_JOINTS = 2
N_ACTIONS_Per_JOINT = 2
ACTIONS_Per_JOINT = [-1, 1]   # action list for each joint
N_ACTIONS = N_JOINTS * N_ACTIONS_Per_JOINT
N_STATES = 11               # the states is position here, like (x, y, z)

PRECISION = 100             # precision for judging if done



def randAng():
    a = np.random.randint(-120, 120)
    b = np.random.randint(-95, 0)
    c = np.random.randint(-120, 120)
    d = np.random.randint(-1, 90)

    return [a, b, c, d]
# WANT_ANGEL = [20, -30, 0, 40]
# WANT_ANGEL = randAng()
# TARGET = np.array(getPos(randAng()))
rand_goal =[[100,100],
            [300,100],
            [300,300],
            [100,300]]

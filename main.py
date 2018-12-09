from DQN import DQN
from env import *
import numpy as np
import matplotlib.pyplot as plt
import torch

dqn = DQN()

cnt,stps = 0, 0
env = ArmEnv()
plt_stp = []

for i_episode in range(EPISODE):
    curr_state = env.reset()
    ep_r, stp = 0, 0
    while True:
        if dqn.memory_counter > MEMORY_CAPACITY:
            action = dqn.choose_action(curr_state)
        else:
            action = env.sample_action()
        next_state, reward, done = env.step(action)
        env.render()
        dqn.store_transition(curr_state, action, reward, next_state)
        # print(curr_state, action, reward, next_state)
        ep_r += reward

        curr_state = next_state
        stp += 1
        stps += 1
        # if (np.linalg.norm(TARGET - curr_state.numpy()[0:3])) < distance :
        #     distance  =  np.linalg.norm(TARGET - curr_state.numpy()[0:3])


        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
        if done == True:
            cnt += 1
        if done or stp > MAX_STEP:
            plt_stp.append(stp)
            break


    if i_episode % 100 == 0:
        print('Ep: ', i_episode,
              '| Ep_r: ', round(ep_r, 2),
              '| stps: ',  stps,
              '|cnt = ',cnt)
        distance = 999999
        cnt = 0
        stps = 0
        torch.save(dqn.eval_net, 'eval_net.pkl')
        torch.save(dqn.target_net, 'target_net.pkl')




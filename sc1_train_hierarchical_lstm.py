import torch
import gym
from gym_starcraft.simple_battle_env import SimpleBattleEnv,Unit_State


from copy import deepcopy
from itertools import count

from Model_hierarchical_lstm import *
from config import *


config = DefaultConfig()
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if config.GPU >= 0 :
    torch.cuda.manual_seed(config.RANDOM_SEED)


# # for debug
#
# from hyperboard import Agent
# HBagent = Agent(username='jlb',password='123',address='127.0.0.1',port=5002)
#
# hp = deepcopy(config.todict())
# hp['mode'] = 'train_reward'
# train_r = HBagent.register(hp, 'reward',overwrite=True)

env = SimpleBattleEnv(config.ip,config.port,config.MYSELF_NUM,config.ENEMY_NUM,config.ACTION_DIM,config.DISTANCE_FACTOR,config.POSITION_RANGE,
                      config.SCREEN_BOX,config.DIE_REWARD,config.HEALTH_REWARD_WEIGHT,config.DONE_REWARD_WEIGHT,config.MY_HEALTH_WEIGHT,config.ENEMY_HEALTH_WEIGHT,
                      config.FRAME_SKIP,config.MAX_STEP,)

env.seed(config.RANDOM_SEED)


ddpg_agent = DDPG(env,config=config)

for episode in count(1):
    print('\n',episode,ddpg_agent.epsilon)
    obs = env.reset()
    state = ddpg_agent.extract_state(obs)
    cl_total,al_total,qe_total,qt_total = 0,0,0,0
    rs = []
    for step in range(config.MAX_STEP):
        action,command = ddpg_agent.select_action(state,decay_e=True)
        next_obs,reward,done,info = env.step(action)

        rs.append(np.asarray(reward))
        next_state = ddpg_agent.extract_state(next_obs)
        ddpg_agent.append_memory(state,command,action,next_state,reward,not done)

        ddpg_agent.train_unit()
        ddpg_agent.train_commander()

        if done:
            qs = []
            q = np.zeros((config.MYSELF_NUM))
            total_reward = 0
            for r in rs[::-1]:
                q = r + config.GAMMA*q
                total_reward += r.sum()/config.MYSELF_NUM
                qs.append(q)
            qs = np.asarray(qs)
            q_mean = np.mean(qs)
            ddpg_agent.train_record[episode] = total_reward
            print('memory: {}/{}'.format(ddpg_agent.commander_memory.current_index,ddpg_agent.commander_memory.max_len))
            print('q_mean: ',q_mean)
            print('train_reward',total_reward)
            # HBagent.append(train_r,episode,total_reward)

            break

        state = next_state


    if episode % config.SAVE_ITERVAL == 0:
        print('\nsave model\n')
        ddpg_agent.save(episode)







import torch
import gym
from gym_starcraft.simple_battle_env import SimpleBattleEnv,Unit_State


from copy import deepcopy
from itertools import count

from Model_hierarchical import *
from config import *

saved_folder = 'HierarchicalNet(3Dragoons_vs_1SuperUltralis.scm)(04-21_20:13)hier_4v1_10'
episode_num = 400

with open(os.path.join(saved_folder,'config.pkl'),'rb') as f:
    config = pickle.load(f)

config.MAX_STEP = 1500
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if config.GPU >= 0 :
    torch.cuda.manual_seed(config.RANDOM_SEED)


env = SimpleBattleEnv(config.ip,config.port,config.MYSELF_NUM,config.ENEMY_NUM,config.ACTION_DIM,config.DISTANCE_FACTOR,config.POSITION_RANGE,
                      config.SCREEN_BOX,config.DIE_REWARD,config.HEALTH_REWARD_WEIGHT,config.DONE_REWARD_WEIGHT,config.MY_HEALTH_WEIGHT,config.ENEMY_HEALTH_WEIGHT,
                      config.FOCUS_WEIGHT,config.FRAME_SKIP,config.MAX_STEP,)
env.step_limit = config.MAX_STEP
env.seed(config.RANDOM_SEED)


ddpg_agent = load_model(saved_folder,episode_num,env,config)

ddpg_agent.test(0,10)







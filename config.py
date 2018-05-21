

class DefaultConfig(object):
    ip = '172.18.233.20'
    port = 11111

    TEST_ITERVAL = 20  # test every TEST_ITERVAL steps
    SAVE_ITERVAL = 50  # save every SAVE_ITERVAL steps
    TEST_NUM = 1

    def __init__(self):
        self.FRAME_SKIP = 2
        self.MAX_STEP = 1500  # max_step in each episode
        self.RANDOM_SEED = 12345
        self.GPU = 0         # GPU id, -1 means use cpu

        self.K = 3           # top-K enenmy nearly myself agent
        self.MYSELF_NUM = 3  # num of myself
        self.ENEMY_NUM = 1  # num os enemy
        self.COMMAND_DIM = 10 # dim of command
        self.ACTION_DIM = 3   # dim of action
        self.MYSELF_FEATURE = 7  # dim of myself state
        self.ENEMY_FEATURE = 12  # dim of enemy state
        self.STATE_DIM = self.MYSELF_FEATURE + self.ENEMY_FEATURE*self.K

        self.MEMORY_SIZE = int(1e5)
        self.BATCH_SIZE = 128
        self.UNIT_BATCH_SIZE = 5*self.BATCH_SIZE
        self.WARMUP = 5*self.BATCH_SIZE

        # model config
        self.ACTOR_LR = 1e-5
        self.CRITIC_LR = 1e-4
        self.GAMMA = 0.99  # discount rate
        self.TAN = 0.001   # update rate between double_network
        self.EPSILON = 1   # random search
        self.EPSILON_DECAY = 6e-6  # epsilon decay rate
        self.RNN_INSIZE = 400   # input size in lstm
        self.RNN_OUTSIZE = 100  # output size in lstm
        self.HIDDEN_SIZE = 200  # hidden size in lstm

        # env_cinfig
        self.DISTANCE_FACTOR = 20   # should large than attack_range
        self.SCREEN_BOX = ((20, 20), (240, 240))  # (left_top, rigth_down)

        # reward related
        self.POSITION_RANGE = 10
        self.HEALTH_REWARD_WEIGHT = 2
        self.DONE_REWARD_WEIGHT = 5
        self.MY_HEALTH_WEIGHT = 1
        self.ENEMY_HEALTH_WEIGHT = 10
        self.DIE_REWARD = -5
        self.FOCUS_WEIGHT = 5

        self.NOTE = "hier_4v1_10"
        self.MAP = None

    def todict(self):
        return vars(self)

if __name__ == '__main__':
    c = DefaultConfig()
    print(c.todict())

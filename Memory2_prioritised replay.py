from collections import namedtuple
import random
import numpy as np
import multiprocessing
import os,sys
import pickle


Bicnet_Memory_Unit = namedtuple('Bicnet_Memory_Unit',('state','action','next_state','reward','not_die','not_done'))

CUnet_Commander_Memory_Unit = namedtuple('CUnet_Commander_Memory_Unit',('global_state','command','global_next_state','reward','not_done'))
CUnet_Unit_Memory_Unit = namedtuple('CUnet_Unit_Memory_Unit',('unit_state','command','action','unit_next_state','next_command','reward','not_done'))

class sumTree():
    '''
    https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-6-prioritized-replay/
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    sample by priority
    '''
    def __init__(self,max_len):
        self.max_len = max_len
        self.tree = np.zeros(2*max_len-1)
        self.data_index = 0

    def add(self,p):
        '''
        add item with priority p
        :param p:
        :return:
        '''
        tree_idx = self.data_index + self.max_len - 1
        self.update(tree_idx,p)

        self.data_index = (self.data_index + 1) % self.max_len

    def update(self,tree_idx,p):
        change = p-self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx-1)//2
            self.tree[tree_idx] += change

    def getleaf(self,v):
        '''
        get data index by v
        :param v:
        :return:
        '''
        parent_idx = 0
        while True:
            cl_idx = 2*parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.max_len + 1
        return data_idx

    @property
    def total_p(self):
        return self.tree[0]


class Bicnet_Memory():
    def __init__(self,max_len,batch_size,worker_num=3):
        self.max_len = max_len
        self.current_index = -1
        self.counter = 0
        self.memory = [None for _ in range(self.max_len)]
        self.priority_tree = sumTree(max_len)

        self.worker_num = worker_num
        self.batch_size = batch_size
        self.q = multiprocessing.Queue(6)

    def start_load(self):
        workers = [multiprocessing.Process(target=self.worker_loop,args=(self.q,)) for _ in range(self.worker_num)]
        for w in workers:
            w.daemon = True
            w.start()

    def worker_loop(self,q):
        while True:
            p_seg = self.priority_tree.total_p/self.batch_size
            state_batch = []
            action_batch = []
            next_state_batch = []
            reward_batch = []
            done_batch = []
            data_idxs = []
            for i in range(self.batch_size):
                l,h = p_seg*i,p_seg*(i+1)
                v = np.random.uniform(l,h)
                data_idx = self.priority_tree.getleaf(v)
                data_idxs.append(data_idx)
                mu = self.memory[data_idx]
                state_batch.append(mu[0])
                action_batch.append(mu[1])
                next_state_batch.append(mu[2])
                reward_batch.append(mu[3])
                done_batch.append(mu[4])

            # batchs = Bicnet_Memory_Unit(*zip(*batch))

            state_batch = np.asarray(state_batch,dtype=np.float32)  # batch * myself_num * enemy_num+1 * state_size
            action_batch = np.asarray(action_batch, dtype=np.float32)
            next_state_batch = np.asarray(next_state_batch, dtype=np.float32)
            reward_batch = np.asarray(reward_batch, dtype=np.float32)
            done_batch = np.asarray(done_batch, dtype=np.int32)

            q.put((data_idxs, state_batch, action_batch, next_state_batch, reward_batch, done_batch))

    def append(self, state, action, next_state, reward, done, priorty=500):
        self.current_index = (self.current_index + 1) % self.max_len
        self.memory[self.current_index] = (state, action, next_state, reward, done)
        self.priority_tree.add(priorty)
        self.counter += 1

    def update_priorty(self,index,priorty):
        tree_index = index + self.max_len - 1
        self.priority_tree.update(tree_index,priorty)

    def sample(self):
        data = self.q.get(True)
        return data

    def sample2(self):
        p_seg = self.priority_tree.total_p / self.batch_size
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        done_batch = []
        data_idxs = []
        for i in range(self.batch_size):
            l, h = p_seg * i, p_seg * (i + 1)
            v = np.random.uniform(l, h)
            data_idx = self.priority_tree.getleaf(v)
            data_idxs.append(data_idx)
            mu = self.memory[data_idx]
            state_batch.append(mu[0])
            action_batch.append(mu[1])
            next_state_batch.append(mu[2])
            reward_batch.append(mu[3])
            done_batch.append(mu[4])

        # batchs = Bicnet_Memory_Unit(*zip(*batch))

        state_batch = np.asarray(state_batch, dtype=np.float32)  # batch * myself_num * enemy_num+1 * state_size
        action_batch = np.asarray(action_batch, dtype=np.float32)
        next_state_batch = np.asarray(next_state_batch, dtype=np.float32)
        reward_batch = np.asarray(reward_batch, dtype=np.float32)
        done_batch = np.asarray(done_batch, dtype=np.int32)

        return data_idxs, state_batch, action_batch, next_state_batch, reward_batch, done_batch

    def save(self, path):
        memory_data = {'memory':self.memory,'priorty_tree':self.priority_tree,'max_len':self.max_len,
                       'current_index':self.current_index,'counter':self.counter,'worker_num':self.worker_num
                       ,'batch_size':self.batch_size}
        with open(os.path.join(path,'memory.pkl'),'wb') as f:
            pickle.dump(memory_data,f)

    @staticmethod
    def memory_load(path):
        with open(os.path.join(path,'memory.pkl'),'rb') as f:
            memory_data = pickle.load(f)
        memory = Bicnet_Memory(memory_data['max_len'],memory_data['batch_size'],memory_data['worker_num'])
        memory.counter = memory_data['counter']
        memory.current_index = memory_data['current_index']
        memory.priority_tree = memory_data['priorty_tree']
        memory.memory = memory_data['memory']
        return memory


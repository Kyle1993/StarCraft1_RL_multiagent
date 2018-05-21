from collections import namedtuple
import random
import numpy as np
import os
import pickle


# Commander_Memory_cell = namedtuple('Bicnet_Memory_Unit',('state','action','next_state','reward','not_die','not_done'))
# Unit_Memory_cell = namedtuple('Bicnet_Memory_Unit',('state','action','next_state','reward','not_die','not_done'))

class Commander_Memory():
    def __init__(self,max_len,batch_size):
        self.max_len = max_len
        self.current_index = -1
        self.counter = 0
        self.memory = [None for _ in range(self.max_len)]
        self.batch_size = batch_size

    def append(self, state, command, next_state, reward, done):
        self.current_index = (self.current_index + 1) % self.max_len
        self.memory[self.current_index] = (state, command, next_state, reward, done)
        self.counter += 1

    def sample(self):

        batch = random.sample(self.memory[:min(self.counter, self.max_len)], self.batch_size)
        batch = list(zip(*batch))

        state_batch = np.asarray(batch[0], dtype=np.float32)  # batch * myself_num * enemy_num+1 * state_size
        command_batch = np.asarray(batch[1], dtype=np.float32)
        next_state_batch = np.asarray(batch[2], dtype=np.float32)
        reward_batch = np.asarray(batch[3], dtype=np.float32)
        done_batch = np.asarray(batch[4], dtype=np.int32)

        return state_batch, command_batch, next_state_batch, reward_batch, done_batch

    def save(self, path):
        memory_data = {'memory':self.memory,'max_len':self.max_len,
                       'current_index':self.current_index,'counter':self.counter,
                       'batch_size':self.batch_size}
        with open(os.path.join(path,'memory.pkl'),'wb') as f:
            pickle.dump(memory_data,f)

    @staticmethod
    def memory_load(path):
        with open(os.path.join(path,'memory.pkl'),'rb') as f:
            memory_data = pickle.load(f)
        memory = Commander_Memory(memory_data['max_len'],memory_data['batch_size'])
        memory.counter = memory_data['counter']
        memory.current_index = memory_data['current_index']
        memory.memory = memory_data['memory']
        return memory

class Unit_Memory():
    def __init__(self,max_len,batch_size):
        self.max_len = max_len
        self.current_index = -1
        self.counter = 0
        self.memory = [None for _ in range(self.max_len)]
        self.batch_size = batch_size

    def append(self, state, command, action, next_state, reward, done):
        self.current_index = (self.current_index + 1) % self.max_len
        self.memory[self.current_index] = (state, command, action, next_state, reward, done)
        self.counter += 1

    def sample(self):

        batch = random.sample(self.memory[:min(self.counter, self.max_len)], self.batch_size)
        batch = list(zip(*batch))

        state_batch = np.asarray(batch[0], dtype=np.float32)
        command_batch = np.asarray(batch[1], dtype=np.float32)
        action_batch = np.asarray(batch[2],dtype=np.float32)
        next_state_batch = np.asarray(batch[3], dtype=np.float32)
        reward_batch = np.asarray(batch[4], dtype=np.float32)
        done_batch = np.asarray(batch[5], dtype=np.int32)

        return state_batch, command_batch, action_batch, next_state_batch, reward_batch, done_batch

    def save(self, path):
        memory_data = {'memory':self.memory,'max_len':self.max_len,
                       'current_index':self.current_index,'counter':self.counter,
                       'batch_size':self.batch_size}
        with open(os.path.join(path,'memory.pkl'),'wb') as f:
            pickle.dump(memory_data,f)

    @staticmethod
    def memory_load(path):
        with open(os.path.join(path,'memory.pkl'),'rb') as f:
            memory_data = pickle.load(f)
        memory = Commander_Memory(memory_data['max_len'],memory_data['batch_size'])
        memory.counter = memory_data['counter']
        memory.current_index = memory_data['current_index']
        memory.memory = memory_data['memory']
        return memory

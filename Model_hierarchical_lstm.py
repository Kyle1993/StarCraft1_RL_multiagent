import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
import copy
import os, sys
import time
import math
import pickle
import numpy as np
import operator
import torchcraft.Constants as tcc

from random_process import OrnsteinUhlenbeckProcess
from Memory_hierarchical import *


def load_model(path, episode, env, config):
    agent = DDPG(env, config)
    agent.load(path, episode)
    return agent


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def copy_parameter(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(sp.data)


def update_parameter(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


class Commander_Actor(nn.Module):
    def __init__(self, state_size, command_size, rnn_insize=400, rnn_outsize=100):
        super(Commander_Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, rnn_insize)
        self.fc2 = nn.Linear(2 * rnn_outsize, command_size)
        self.birnn = nn.LSTM(input_size=rnn_insize, hidden_size=rnn_outsize, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.init_weights()

    def forward(self, s, h0, c0):
        x = F.relu(self.fc1(s))
        out, _ = self.birnn(x, (h0, c0))
        out = F.tanh(self.fc2(out))
        return out

    def init_weights(self, ):
        torch.nn.init.xavier_uniform(self.fc1.weight.data)
        self.fc2.weight.data.uniform_(3e-3, 3e-3)
        # for weight_list in self.birnn.all_weights:
        #     for weight in weight_list[:2]:
        #         torch.nn.init.xavier_uniform(weight)


class Commander_Critic(nn.Module):
    def __init__(self, state_size, command_size, batch_size, rnn_insize=200, rnn_outsize=50):
        super(Commander_Critic, self).__init__()

        self.h0 = Variable(torch.zeros(2, batch_size, rnn_outsize),requires_grad=False)  # (num_layer*num_driection)*bs*rnn_ooutsize
        self.c0 = Variable(torch.zeros(2, batch_size, rnn_outsize),requires_grad=False)  # (num_layer*num_driection)*bs*rnn_ooutsize

        self.fc1 = nn.Linear(state_size + command_size, rnn_insize)
        self.fc2 = nn.Linear(2 * rnn_outsize, 1)
        self.birnn = nn.LSTM(input_size=rnn_insize, hidden_size=rnn_outsize, num_layers=1, batch_first=True,
                            bidirectional=True)

        self.init_weights()

    def forward(self, s, c):

        out = torch.cat((s, c), 2)
        out = F.relu(self.fc1(out))

        out, hn = self.birnn(out, (self.h0, self.c0))
        out = self.fc2(out).squeeze(2)
        return out

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.fc1.weight.data)
        self.fc2.weight.data.uniform_(3e-3, 3e-3)
        # for weight_list in self.birnn.all_weights:
        #     for weight in weight_list[:2]:
        #         torch.nn.init.xavier_uniform(weight)


class Unit_Actor(nn.Module):
    def __init__(self, state_size, command_size, action_size, hidden_size=200):
        super(Unit_Actor, self).__init__()

        self.fc1 = nn.Linear(state_size + command_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.init_weights()

    def forward(self, s, c):
        out = torch.cat((s, c), 1)
        out = F.relu(self.fc1(out))
        out = F.tanh(self.fc2(out))
        return out

    def init_weights(self, ):
        torch.nn.init.xavier_uniform(self.fc1.weight.data)
        self.fc2.weight.data.uniform_(3e-3, 3e-3)


class Unit_Critic(nn.Module):
    def __init__(self, state_size, command_size, action_size, hidden_size=200):
        super(Unit_Critic, self).__init__()

        self.fc1 = nn.Linear(state_size + command_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.init_weights()

    def forward(self, s, c, a):
        out = torch.cat((s, c, a), 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.fc1.weight.data)
        self.fc2.weight.data.uniform_(3e-3, 3e-3)


class DDPG(object):
    def __init__(self, env, config):
        self.name = 'HierarchicalNet'
        self.save_folder = None
        self.test_record = {}
        self.train_record = {}

        self.config = config
        self.env = env
        self.epsilon = config.EPSILON

        self.commander_memory = Commander_Memory(config.MEMORY_SIZE, config.BATCH_SIZE)
        self.unit_memory = Unit_Memory(2 * config.MEMORY_SIZE, config.UNIT_BATCH_SIZE)

        self.commander_actor = Commander_Actor(config.STATE_DIM, config.COMMAND_DIM, config.RNN_INSIZE)
        self.commander_actor_target = Commander_Actor(config.STATE_DIM, config.COMMAND_DIM, config.RNN_INSIZE)
        self.commander_critic = Commander_Critic(config.STATE_DIM, config.COMMAND_DIM, config.BATCH_SIZE,
                                                 config.RNN_INSIZE)
        self.commander_critic_target = Commander_Critic(config.STATE_DIM, config.COMMAND_DIM, config.BATCH_SIZE,
                                                        config.RNN_INSIZE)

        self.unit_actor = Unit_Actor(config.STATE_DIM, config.COMMAND_DIM, config.ACTION_DIM)
        self.unit_actor_target = Unit_Actor(config.STATE_DIM, config.COMMAND_DIM, config.ACTION_DIM)
        self.unit_critic = Unit_Critic(config.STATE_DIM, config.COMMAND_DIM, config.ACTION_DIM, config.HIDDEN_SIZE)
        self.unit_critic_target = Unit_Critic(config.STATE_DIM, config.COMMAND_DIM, config.ACTION_DIM,
                                              config.HIDDEN_SIZE)

        self.commander_actor_h0 = Variable(torch.zeros(2, 1, config.RNN_OUTSIZE), requires_grad=False)
        self.commander_actor_c0 = Variable(torch.zeros(2, 1, config.RNN_OUTSIZE), requires_grad=False)

        if config.GPU >= 0:
            self.commander_actor.cuda(device=config.GPU)
            self.commander_actor_target.cuda(device=config.GPU)
            self.commander_critic.cuda(device=config.GPU)
            self.commander_critic_target.cuda(device=config.GPU)
            self.unit_actor.cuda(device=config.GPU)
            self.unit_actor_target.cuda(device=config.GPU)
            self.unit_critic.cuda(device=config.GPU)
            self.unit_critic_target.cuda(device=config.GPU)
            self.commander_critic.h0 = self.commander_critic.h0.cuda(device=config.GPU)
            self.commander_critic.c0 = self.commander_critic.c0.cuda(device=config.GPU)
            self.commander_critic_target.h0 = self.commander_critic_target.h0.cuda(device=config.GPU)
            self.commander_critic_target.c0 = self.commander_critic_target.c0.cuda(device=config.GPU)
            self.commander_actor_h0 = self.commander_actor_h0.cuda(device=config.GPU)
            self.commander_actor_c0 = self.commander_actor_c0.cuda(device=config.GPU)


        copy_parameter(self.commander_actor, self.commander_actor_target)
        copy_parameter(self.commander_critic, self.commander_critic_target)
        copy_parameter(self.unit_actor, self.unit_actor_target)
        copy_parameter(self.unit_critic, self.unit_critic_target)

        self.commander_actor_optimizer = optim.Adam(self.commander_actor.parameters(), lr=config.ACTOR_LR)
        self.unit_actor_optimizer = optim.Adam(self.unit_actor.parameters(), lr=config.ACTOR_LR)
        self.commander_critic_optimizer = optim.Adam(self.commander_critic.parameters(), lr=config.CRITIC_LR)
        self.unit_critic_optimizer = optim.Adam(self.unit_critic.parameters(), lr=config.CRITIC_LR)

        self.criterion = nn.MSELoss()
        self.action_noise = OrnsteinUhlenbeckProcess(size=(config.MYSELF_NUM, config.ACTION_DIM), theta=10, mu=0.,
                                                     sigma=2)
        self.command_noise = OrnsteinUhlenbeckProcess(size=(1, config.MYSELF_NUM, config.COMMAND_DIM), theta=10, mu=0.,
                                                      sigma=2)

        # self.action_noise = OrnsteinUhlenbeckProcess(size=(config.MYSELF_NUM, config.ACTION_DIM), theta=30, mu=0., sigma=3)
        # self.command_noise = OrnsteinUhlenbeckProcess(size=(1,config.MYSELF_NUM, config.COMMAND_DIM), theta=30, mu=0., sigma=3)


        # normalize
        state_normalization_myelf = [1, 100, 100, 1, 100, 100, 1]
        state_normalization_enemy = [1, 100, 100, 100, 100, 10, 100, 100, 1, 1, 1, 10]
        self.state_normalization = state_normalization_myelf
        for i in range(config.K):
            self.state_normalization += state_normalization_enemy
        self.state_normalization = np.asarray(self.state_normalization, dtype=np.float32)

    def append_memory(self, states, commands, actions, next_states, rewards, dones):
        self.commander_memory.append(states, commands, next_states, rewards, [dones] * self.config.MYSELF_NUM)
        for i in range(self.config.MYSELF_NUM):
            self.unit_memory.append(states[i], commands[i], actions[i], next_states[i], rewards[i], dones)

    def get_command(self, states):
        group_states = states.view(int(self.config.UNIT_BATCH_SIZE / 8), 8, self.config.STATE_DIM)
        group_states.volatile = True
        command = self.commander_actor_target(group_states,self.commander_actor_h0.repeat(1, int(self.config.UNIT_BATCH_SIZE / 8),1),self.commander_actor_c0.repeat(1, int(self.config.UNIT_BATCH_SIZE / 8),1)).contiguous()
        command = command.view(self.config.UNIT_BATCH_SIZE, self.config.COMMAND_DIM)
        command.volatile = False
        command.requires_grad = False
        return command

    def train_commander(self):
        # if not warm up
        if self.commander_memory.counter < self.config.WARMUP:
            return 0, 0, 0, 0

        state_batch, command_batch, next_state_batch, reward_batch, done_batch = self.commander_memory.sample()

        sb = Variable(torch.from_numpy(state_batch)).float()
        cb = Variable(torch.from_numpy(command_batch)).float()
        nsb = Variable(torch.from_numpy(next_state_batch)).float()
        rb = Variable(torch.from_numpy(reward_batch)).float()
        db = Variable(torch.from_numpy(done_batch)).float()

        if self.config.GPU >= 0:
            sb = sb.cuda(device=self.config.GPU)
            cb = cb.cuda(device=self.config.GPU)
            nsb = nsb.cuda(device=self.config.GPU)
            rb = rb.cuda(device=self.config.GPU)
            db = db.cuda(device=self.config.GPU)

        # update critic
        self.commander_critic_optimizer.zero_grad()
        ncb = self.commander_actor_target(nsb, self.commander_actor_h0.repeat(1, self.config.BATCH_SIZE, 1),self.commander_actor_c0.repeat(1, self.config.BATCH_SIZE, 1))
        nqb = self.commander_critic_target(nsb, ncb)
        q_target = (rb + self.config.GAMMA * db * nqb)
        q_eval = self.commander_critic(sb, cb)
        q_target = q_target.detach()
        value_loss = F.mse_loss(q_eval, q_target)
        value_loss.backward()
        self.commander_critic_optimizer.step()

        # update actor
        self.commander_actor_optimizer.zero_grad()
        acb = self.commander_actor(sb, self.commander_actor_h0.repeat(1, self.config.BATCH_SIZE, 1),self.commander_actor_c0.repeat(1, self.config.BATCH_SIZE, 1))
        q = self.commander_critic(sb, acb)
        policy_loss = -torch.mean(q)
        policy_loss.backward()
        self.commander_actor_optimizer.step()

        # update parameter between two network
        update_parameter(self.commander_critic_target, self.commander_critic, self.config.TAN)
        update_parameter(self.commander_actor_target, self.commander_actor, self.config.TAN)

        # # update priorty
        # tderror = torch.mean(torch.abs(q_target-q_eval),1).cpu().data.numpy()
        # for i,id in enumerate(idxs):
        #     self.memory.update_priorty(id,float(tderror[i]))

        return value_loss.data[0], policy_loss.data[0], torch.mean(q_eval).data[0], torch.mean(q_target).data[0]

    def train_unit(self):
        # if not warm up
        if self.unit_memory.counter < 10 * self.config.WARMUP:
            return 0, 0, 0, 0

        state_batch, command_batch, action_batch, next_state_batch, reward_batch, done_batch = self.unit_memory.sample()

        sb = Variable(torch.from_numpy(state_batch)).float()
        cb = Variable(torch.from_numpy(command_batch)).float()
        ab = Variable(torch.from_numpy(action_batch)).float()
        nsb = Variable(torch.from_numpy(next_state_batch)).float()
        rb = Variable(torch.from_numpy(reward_batch)).float()
        db = Variable(torch.from_numpy(done_batch)).float()

        rb = rb.view(self.config.UNIT_BATCH_SIZE, 1)
        db = db.view(self.config.UNIT_BATCH_SIZE, 1)

        if self.config.GPU >= 0:
            sb = sb.cuda(device=self.config.GPU)
            cb = cb.cuda(device=self.config.GPU)
            ab = ab.cuda(device=self.config.GPU)
            nsb = nsb.cuda(device=self.config.GPU)
            rb = rb.cuda(device=self.config.GPU)
            db = db.cuda(device=self.config.GPU)

        ncb = self.get_command(nsb)

        # update critic
        self.unit_critic_optimizer.zero_grad()
        nab = self.unit_actor_target(nsb, ncb)
        nqb = self.unit_critic_target(nsb, ncb, nab)
        q_target = (rb + self.config.GAMMA * db * nqb)
        q_eval = self.unit_critic(sb, cb, ab)
        q_target = q_target.detach()
        value_loss = F.mse_loss(q_eval, q_target)
        value_loss.backward()
        self.unit_critic_optimizer.step()

        # update actor
        self.unit_actor_optimizer.zero_grad()
        aab = self.unit_actor(sb, cb)
        q = self.unit_critic(sb, cb, aab)
        policy_loss = -torch.mean(q)
        policy_loss.backward()
        self.unit_actor_optimizer.step()

        # update parameter between two network
        update_parameter(self.unit_critic_target, self.unit_critic, self.config.TAN)
        update_parameter(self.unit_actor_target, self.unit_actor, self.config.TAN)

        # # update priorty
        # tderror = torch.mean(torch.abs(q_target-q_eval),1).cpu().data.numpy()
        # for i,id in enumerate(idxs):
        #     self.memory.update_priorty(id,float(tderror[i]))

        return value_loss.data[0], policy_loss.data[0], torch.mean(q_eval).data[0], torch.mean(q_target).data[0]

    def select_action(self, s, is_train=True, decay_e=True, ignor_warmup=False):
        '''

        :param is_train: if true, action += noise
        :param decay_e:  if true, decay epsilon every step
        :param ignor_warmup: if true, select op will not wait for warmup
        :return: action
        '''
        state = Variable(torch.from_numpy(s), volatile=True).unsqueeze(0)
        if self.config.GPU >= 0:
            state = state.cuda(device=self.config.GPU)

        self.commander_actor.eval()
        self.unit_actor.eval()
        if self.commander_memory.counter < self.config.WARMUP and is_train and not ignor_warmup:
            command = Variable(
                torch.from_numpy(np.random.uniform(-1, 1, (1, self.config.MYSELF_NUM, self.config.COMMAND_DIM))),
                volatile=True).float()
            if self.config.GPU >= 0:
                command = command.cuda(device=self.config.GPU)
        else:
            command = self.commander_actor(state, self.commander_actor_h0,self.commander_actor_c0)
            command_noise = Variable(torch.from_numpy(self.command_noise.sample())).float()
            if self.config.GPU >= 0:
                command_noise = command_noise.cuda(device=self.config.GPU)
            command += is_train * max(self.epsilon, 0.2) * command_noise
            command = command.clamp(-1, 1)

        actions = []
        for i in range(self.config.MYSELF_NUM):
            c = command[:, i]
            s = state[:, i]
            a = self.unit_actor(s, c)
            actions.append(a)
        actions = torch.cat(actions, 0).cpu().data.numpy()
        action_noise = self.action_noise.sample()
        actions += is_train * max(self.epsilon, 0.2) * action_noise
        actions = np.clip(actions, -1., 1.)

        if decay_e:
            if self.commander_memory.counter > self.config.WARMUP:
                self.epsilon -= self.config.EPSILON_DECAY
        self.commander_actor.train()
        self.unit_actor.train()

        return actions, command.cpu().data.numpy()[0]

    def extract_state(self, obs):
        '''
        extract state info from obs

        :param obs:
        :return: state
        '''
        # enemy basic :[die, health, shield, x, y, delta_health, attackCD, targetX, targetY]
        # enemy add   :[attack this myself_agent,if closest,dx,dy,distance]
        enemy_basic_len = 8
        enemy_add_len = 4
        assert self.config.ENEMY_FEATURE == enemy_basic_len + enemy_add_len

        myself_units_state = []
        units_enemy = {}
        myself_units_underattack = []

        for index, unit in enumerate(obs['enemy']):
            state = [unit.die, unit.health, unit.shield, unit.x, unit.y, unit.attackCD, unit.targetX, unit.targetY,
                     unit.targetUnitId]
            units_enemy[unit.id] = state
            myself_units_underattack.append(unit.targetUnitId)
        myself_units_underattack = set(myself_units_underattack)
        for index, unit in enumerate(obs['myself']):
            state = [unit.die, unit.health, unit.shield, unit.health > (0.2 * unit.max_health), unit.x, unit.y,
                     int(unit.id in myself_units_underattack)]

            # enemy_state
            nearly_enemy_ids, num = self.nearyl_topK(unit, obs['enemy'], k=self.config.K)
            if num == 0:
                enemy_state = [0] * self.config.K * (enemy_basic_len + enemy_add_len)
            else:
                enemy_state = []
                for idx, enemy_id in enumerate(nearly_enemy_ids):
                    # if there no enough alive enemy, chose the farthest one
                    if enemy_id == -1:
                        enemy_id = nearly_enemy_ids[num - 1]
                    es = units_enemy[enemy_id]
                    if unit.die:
                        enemy_state_add = [0, 255 / self.config.DISTANCE_FACTOR, 255 / self.config.DISTANCE_FACTOR,
                                           255 * 1.4]
                    else:
                        under_attack = int(es[-1] == unit.id)
                        dx, dy = es[3] - unit.x, es[4] - unit.y
                        distance = math.sqrt(dx ** 2 + dy ** 2)
                        enemy_state_add = [under_attack, dx / self.config.DISTANCE_FACTOR,
                                           dy / self.config.DISTANCE_FACTOR, distance]
                    enemy_state.extend(es[:-1] + enemy_state_add)
            # cat to state
            state.extend(enemy_state)
            myself_units_state.append(state)
        myself_units_state = np.asarray(myself_units_state, dtype=np.float32)
        myself_units_state = myself_units_state / self.state_normalization

        assert myself_units_state.shape == (self.config.MYSELF_NUM, self.config.STATE_DIM)

        return myself_units_state

    def nearyl_topK(self, unit, enemy_units, k=3):
        x, y = unit.x, unit.y
        nearly_enemys = {}
        for enemy in enemy_units:
            if not enemy.die:
                uid = enemy.id
                d = math.sqrt((enemy.x - x) ** 2 + (enemy.y - y) ** 2)
                if len(nearly_enemys) < k:
                    nearly_enemys[uid] = d
                else:
                    max_id, max_d = max(nearly_enemys.items(), key=operator.itemgetter(1))
                    if d < max_d:
                        del nearly_enemys[max_id]
                        nearly_enemys[uid] = d
        sorted_nearly_enemys = sorted(nearly_enemys.items(), key=lambda x: x[1])
        nearly_enemys_id = [id for id, d in sorted_nearly_enemys]
        num = len(nearly_enemys)
        return nearly_enemys_id + [-1] * (k - num), num

    def test(self, eposide, test_num, ):
        '''
        test model, without noise

        :return: mean test eposide_total_reward
        '''
        total_reward = 0
        win = 0
        for i in range(test_num):
            obs = self.env.reset()
            state = self.extract_state(obs)
            for test_step in range(self.config.MAX_STEP):
                action, command = self.select_action(state, is_train=False, decay_e=False)
                next_obs, reward, done, info = self.env.step(action)
                next_state = self.extract_state(next_obs)
                time.sleep(0.02)
                if done:
                    if self.env.win:
                        win += 1
                    break
                total_reward += sum(reward)
                state = next_state

        win_rate = win / test_num
        total_reward = total_reward / (self.config.MYSELF_NUM * test_num)

        self.test_record.append((eposide, total_reward))

        return total_reward, win, win_rate

    def save(self, episode):
        '''
        save model, hyperparameters, test record

        :param episode:
        :return:
        '''
        timestr = time.strftime('(%m-%d_%H:%M)', time.localtime(time.time()))
        mapname = '({})'.format(self.env.getMapName())
        if self.save_folder is None:
            self.save_folder = self.name + mapname + timestr + self.config.NOTE
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
            self.config.MAP = self.env.getMapName()
            with open(os.path.join(self.save_folder, 'config'), 'w') as f:
                f.write('config:\n\n')
                for k, v in sorted(self.config.todict().items()):
                    f.write('{:<16} : {}\n'.format(k, v))

            with open(os.path.join(self.save_folder, 'config.pkl'), 'wb') as f:
                pickle.dump(self.config, f)

        torch.save(self.commander_actor.cpu(), os.path.join(self.save_folder, 'commander_actor_%d.mod' % episode))
        torch.save(self.commander_critic.cpu(), os.path.join(self.save_folder, 'commander_critic_%d.mod' % episode))
        torch.save(self.unit_actor.cpu(), os.path.join(self.save_folder, 'unit_actor_%d.mod' % episode))
        torch.save(self.unit_critic.cpu(), os.path.join(self.save_folder, 'unit_critic_%d.mod' % episode))

        with open(os.path.join(self.save_folder, 'test_record.pkl'), 'wb') as f:
            pickle.dump(self.test_record, f)
        with open(os.path.join(self.save_folder, 'train_record.pkl'), 'wb') as f:
            pickle.dump(self.train_record, f)
        self.commander_memory.save(self.save_folder)
        self.unit_memory.save(self.save_folder)

        if self.config.GPU >= 0:
            self.commander_actor.cuda(device=self.config.GPU)
            self.commander_critic.cuda(device=self.config.GPU)
            self.unit_actor.cuda(device=self.config.GPU)
            self.unit_critic.cuda(device=self.config.GPU)

    def load(self, saved_folder, episode):
        # load network
        self.commander_actor = torch.load(os.path.join(saved_folder, 'commander_actor_{}.mod'.format(episode)))
        self.commander_actor_target = torch.load(os.path.join(saved_folder, 'commander_actor_{}.mod'.format(episode)))
        self.commander_critic = torch.load(os.path.join(saved_folder, 'commander_critic_{}.mod'.format(episode)))
        self.commander_critic_target = torch.load(os.path.join(saved_folder, 'commander_critic_{}.mod'.format(episode)))
        self.unit_actor = torch.load(os.path.join(saved_folder, 'unit_actor_{}.mod'.format(episode)))
        self.unit_actor_target = torch.load(os.path.join(saved_folder, 'unit_actor_{}.mod'.format(episode)))
        self.unit_critic = torch.load(os.path.join(saved_folder, 'unit_critic_{}.mod'.format(episode)))
        self.unit_critic_target = torch.load(os.path.join(saved_folder, 'unit_critic_{}.mod'.format(episode)))

        if self.config.GPU >= 0:
            self.commander_actor.cuda(device=self.config.GPU)
            self.commander_actor_target.cuda(device=self.config.GPU)
            self.commander_critic.cuda(device=self.config.GPU)
            self.commander_critic_target.cuda(device=self.config.GPU)
            self.unit_actor.cuda(device=self.config.GPU)
            self.unit_actor_target.cuda(device=self.config.GPU)
            self.unit_critic.cuda(device=self.config.GPU)
            self.unit_critic_target.cuda(device=self.config.GPU)

        self.commander_critic_optimizer = optim.Adam(self.commander_critic.parameters(), lr=self.config.CRITIC_LR)
        self.commander_actor_optimizer = optim.Adam(self.commander_actor.parameters(), lr=self.config.ACTOR_LR)
        self.unit_critic_optimizer = optim.Adam(self.unit_critic.parameters(), lr=self.config.CRITIC_LR)
        self.unit_actor_optimizer = optim.Adam(self.unit_actor.parameters(), lr=self.config.ACTOR_LR)

        # load memory
        self.commander_memory = Commander_Memory.memory_load(saved_folder)
        self.unit_memory = Unit_Memory.memory_load(saved_folder)

        # # load record
        # with open(os.path.join(self.save_folder, 'test_record.pkl'), 'rb') as f:
        #     self.test_record = pickle.load(f)
        # with open(os.path.join(self.save_folder, 'train_record.pkl'), 'rb') as f:
        #     self.train_record = pickle.load(f)
        # for r in self.test_record:
        #     if r[0]>episode:
        #         self.test_record.remove(r)
        # for r in self.train_record:
        #     if r[0]>episode:
        #         self.train_record.remove(r)

    def print_action(self, action):
        env_cmd = self.env._make_commands(action)
        ecs = []
        for ec in env_cmd:
            if len(ec) > 2:
                if ec[2] == tcc.unitcommandtypes.Attack_Unit:
                    c = [ec[1], 'A', ec[3]]
                else:
                    c = [ec[1], 'M', ec[4], ec[5]]
                print(c)
                ecs.append(c)
        return ecs






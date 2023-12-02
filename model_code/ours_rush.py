import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import sys
from collections import namedtuple
from itertools import count
from env_rush_hour import Env
import config

import csv
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
env.seed(config.RANDOM_SEED)
torch.manual_seed(0)

feature_dim = env.feature_dim # 7 features
f_tran = env.f_node
f_task = env.f_task
f_actor = env.f_node * 2 + env.f_task #concatenate
num_action = env.node_num #server or local
n_of_node = env.node_num #server_node + WDs 


d_model = 7
TRAN_NUM_HEAD = 1
TRAN_NUM_LAYER = 2

Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'a_log_prob', 'next_state']
)
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(f_actor, 64)
        self.fc2 = nn.Linear(64, 16)
        self.action_head = nn.Linear(16, 2)

    def forward(self, x):
        assert x.shape[1] == f_actor,f"the shape of the actor input is {x.shape}"
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.action_head(x)

        action_prob = F.softmax(x, dim=1,dtype=torch.double) 

        return action_prob


class Transformer_Model(nn.Module):
    def __init__(self, **args):
        super().__init__()
        __enc_layer = nn.TransformerEncoderLayer(d_model, nhead=TRAN_NUM_HEAD, batch_first=True)
        __enc_layer.pos_enc = None
        self.transformer_enc = nn.TransformerEncoder(__enc_layer, num_layers=TRAN_NUM_LAYER)
        
        self.input_batch_norm = nn.BatchNorm1d(feature_dim)
        self.input_hidden  = nn.Linear(feature_dim, d_model)

        self.hidden1 = nn.Linear(d_model*n_of_node, d_model*n_of_node)
        self.hidden2 = nn.Linear(d_model*n_of_node, d_model*n_of_node)
        self.output = nn.Linear(d_model*n_of_node, f_tran)

        self.fc1 = nn.Linear(f_task, 64)
        self.fc2 = nn.Linear(64, f_task)

    def forward(self, node_f, task_f, b_s):


        assert node_f.shape[1] == 7,f"the shape of the input is{node_f.shape}" # test feature_dim if it's 7
        norm_x = self.input_batch_norm(node_f) 
        norm_x = norm_x.reshape(b_s, -1, feature_dim)  # 为了匹配Linear和Transformer的输入维度
        norm_x = nn.functional.leaky_relu(self.input_hidden(norm_x))

        #transformer
        x_encode = norm_x.reshape(b_s, -1, feature_dim)
        assert node_f.shape[1] == d_model,f"the transformer input node shape is {node_f.shape}"
        enc_hiddens  = self.transformer_enc(x_encode)
        enc_hiddens = enc_hiddens.reshape(b_s, -1)

        x_h = F.dropout(F.leaky_relu(self.hidden1(enc_hiddens)), p=0.1)
        x_h = F.dropout(F.leaky_relu(self.hidden2(x_h)), p=0.1)
        x_h = nn.functional.sigmoid(self.output(enc_hiddens))
        node_f = node_f.reshape(b_s, -1)

        task_f = F.leaky_relu(self.fc1(task_f))
        task_f = F.leaky_relu(self.fc2(task_f)) 
        task_f = task_f.reshape(b_s,-1)

        con_state = torch.cat((node_f, x_h, task_f), dim=1)
        assert con_state.shape[1] == f_actor,f"the output of transformer shape is {con_state.shape}"
        return con_state
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(f_actor, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 16)
        self.state_value = nn.Linear(16, 1)
        # self.apply(init_weights)

    def forward(self, x):
        x = F.dropout(F.leaky_relu(self.fc1(x)))
        x = F.dropout(F.leaky_relu(self.fc2(x)))
        x = F.leaky_relu(self.fc3(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 1000
    batch_size = 300
    
    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().to(device)
        self.embed_net = Transformer_Model().to(device)
        self.critic_net = Critic().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-4)

    def select_action(self, task, node_f, task_f): 
        gen_action_b_s = 1
        node_f = torch.tensor(node_f, dtype=torch.float).unsqueeze(0).to(device)
        task_f = torch.tensor(task_f, dtype=torch.float).to(device)

        state= self.embed_net(node_f, task_f, gen_action_b_s) #过transformer,此时b_s为1，输出为1*84
        with torch.no_grad():
            # action_prob = torch.mul(self.actor_net(state), action_mask)
            action_prob = self.actor_net(state)

        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample().item()

        return 0 if action == 0 else task.n_id, 0

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        #state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        #state_encoder = state.reshape(1000,7,6)
        state = [t.state for t in self.buffer] 
        state = torch.stack(state)

        assert state.shape == torch.Size([self.buffer_capacity, 1, env.f_node+env.f_task]), f"the shape is{state.shape}"
        node_f = state[:, :, :env.f_node]  # shape is (1000, 1, 42)
        task_f = state[:, :, env.f_node:]   # shape is (1000, 1, 2)
        assert node_f.shape == torch.Size([self.buffer_capacity, 1, env.f_node]), f"the shape is{state.shape}"
        node_f = node_f.view(self.buffer_capacity,feature_dim,env.node_num).squeeze(0)
        assert node_f.shape == torch.Size([self.buffer_capacity, feature_dim, env.node_num]), f"the shape is{state.shape}"
       
        state = self.embed_net(node_f,task_f,self.buffer_capacity)
 
        #action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.buffer]

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + 0.98 * R
            Gt.insert(0, R)

        Gt = torch.tensor(Gt, dtype=torch.float).to(device)

        actor_losses = []
        value_losses = []

        _temp_action = torch.tensor([0 if t.action == 0 else 1 for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)

        old_action_log_prob = torch.log(self.actor_net(state).gather(1, _temp_action)).detach()
        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                Gt_index = Gt[index].view(-1, 1)

                V = self.critic_net(state[index]) # critic network

                delta = Gt_index - V # δt = rt + γV (st+1) − V (st)
                advantage = delta.detach()
                action_prob = torch.log(self.actor_net(state[index]).gather(1, _temp_action[index]))
                
                ratio = torch.exp(action_prob-old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage 
                action_loss = -torch.min(surr1, surr2).mean() # eq(18)
                action_loss_lr = self.actor_optimizer.param_groups[0]['lr'] * action_loss
                # print(f"action_loss_lr is {action_loss_lr}")
                self.actor_optimizer.zero_grad()

                action_loss.backward(retain_graph=True) # actor 参数更新
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, V)
                value_loss_lr = self.critic_net_optimizer.param_groups[0]['lr'] * value_loss
                # print(f"value_loss_lr is {value_loss_lr}")
               
                self.critic_net_optimizer.zero_grad()
                value_loss.backward(retain_graph=True) # critic network 参数更新
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()


                self.training_step += 1
                actor_losses.append(action_loss.item())
                value_losses.append(value_loss.item())
        
        del self.buffer[:]
        return np.mean(actor_losses), np.mean(value_losses)


def store_data_baseline(t_list,group_size):
    l_result = []
    # 定义每个组的大小
    # 循环遍历列表
    for i in range(0, len(t_list), group_size):
        group = t_list[i:i + group_size]  # 获取当前组
        group_avg = sum(group)/group_size  # 计算当前组的总和
        l_result.append(group_avg)  # 将总和添加到结果列表中

    return l_result

def main():
    agent = PPO()
    return_reward_list = []
    save_per_time_list = []

    #save loss
    value_loss = []
    policy_loss = []

    #save data
    save_energy_ep = []
    save_time_ep = []

    for i_epoch in tqdm(range(1100)):
        # ppo
        ep_reward = []
        ep_energy = []
        ep_time = []
        
        ep_action_prob = 0

        # Reset environment 
        env.reset(i_epoch) 
        cnt = 0
        for t in count():
            # done, upgrade, idx = env.env_up()
            env.env_up()
            # task不为空且 第一个task开始执行
            while env.task and env.task[0].start_time == env.time:
                    cnt += 1
                    curr_task = env.task.pop(0)
                    # ----------ppo--------------
                    # get current state
                    node_f,task_f = env.get_obs(curr_task)

                    # Select action at according to πθ (at | st )
                    action, action_prob = agent.select_action(curr_task,node_f,task_f)
                    node_f = torch.tensor(node_f, dtype=torch.float).to(device)
                    node_f = node_f.reshape(1,f_tran)
                    task_f = torch.tensor(task_f, dtype=torch.float).to(device)
                    task_f = task_f.reshape(1,f_task)
                    state = torch.cat((node_f,task_f), dim=1)
                    assert state.shape[1] == f_tran+f_task,f"the state shape is{state.shape}"
                        
                    # 1.Execute action at and obtain the reward rt
                    # 2.Get the next state st+1 
                    node_f,task_f,reward,energy,time= env.step(curr_task, action)
                    node_f = torch.tensor(node_f, dtype=torch.float).to(device)
                    node_f = node_f.reshape(1,f_tran)
                    task_f = torch.tensor(task_f, dtype=torch.float).to(device)
                    task_f = task_f.reshape(1,f_task)
                    next_state = torch.cat((node_f,task_f), dim=1)
                    assert next_state.shape[1] == f_tran+f_task,f"the next state shape is{next_state.shape}"

                    # PPO save data
                    ep_reward.append(reward)
                    ep_action_prob += action_prob
                    ep_energy.append(energy)
                    ep_time.append(time)
                    # ep_utility.append(0.5*energy+0.5*time)
                    # Store transition (st , at , rt , st+1 ) in D
                    trans = Transition(state, action, reward, action_prob, next_state)

                    # for training step ....
                    if agent.store_transition(trans):
                        ac_loss, v_loss = agent.update()
                        value_loss.append(v_loss)
                        policy_loss.append(ac_loss)
                        # print('ac_loss', ac_loss)
                        # print('v_loss', v_loss)
                    # ----------ppo-------------- 


            if not env.task: #queue里的task耗尽，计算所有数值
                #save data
                return_reward_list.append(np.mean(ep_reward))
                save_energy_ep.append(np.mean(ep_energy))
                save_time_ep.append(np.mean(ep_time))

                save_per_time_list.append(sum(ep_time))
    
                # the output of the current episode
                #print('Episode: {}, reward: {}, total_time: {}, total_energy: {}'.format(i_epoch, round(np.mean(ep_reward), 3), total_times_ep, total_energys_ep))
                print("final min energy cost: ",np.mean(save_energy_ep[-100:]))
                print("final min time cost: ",np.mean(save_time_ep[-100:]))
                print("final reward: ",np.mean(return_reward_list[-100:]))
                break


    return return_reward_list, save_per_time_list,env.file_name
            
if __name__ == '__main__':
    per_slot_time = []
    _,per_slot_time,file_name = main()
    csv_filename = 'ours_per_time'+file_name+'.csv'
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for value in tqdm(per_slot_time):
            csv_writer.writerow(value)

 




 
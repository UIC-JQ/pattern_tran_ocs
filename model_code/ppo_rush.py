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

f_tran = env.f_node
f_task = env.f_task
f_actor = env.f_node + env.f_task #concatenate
num_state = f_actor
num_action = env.node_num #server or local

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
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 16)
        # self.action_head = nn.Linear(16, num_action)
        # self.attention_weights = nn.Linear(num_state, num_state)
        # self.attention_context = nn.Linear(num_state, num_state)
        self.action_head = nn.Linear(16, 2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.action_head(x)
        action_prob = F.softmax(x, dim=1,dtype=torch.double) 
       
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
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
    batch_size = 32
    
    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().to(device)
        self.critic_net = Critic().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-4)

    def select_action(self, task, node_f, task_f):
        node_f = torch.tensor(node_f, dtype=torch.float).reshape(1, -1).to(device)
        task_f = torch.tensor(task_f, dtype=torch.float).view(1, -1).to(device)
        state= torch.cat((node_f, task_f),dim=1) 
        assert state.shape[1] == f_actor,f"the actor input shape is {state.shape}"
        with torch.no_grad():
            action_prob = self.actor_net(state)
  
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample().item()
        return 0 if action == 0 else task.n_id, action_prob
        # return action, 0

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        state = [t.state for t in self.buffer]

        state = torch.stack(state).squeeze(1).to(device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
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
     
        old_action_log_prob = torch.log(self.actor_net(state).squeeze(1).gather(1, _temp_action)).detach()
   
        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                Gt_index = Gt[index].view(-1, 1)

                V = self.critic_net(state[index]) # critic network

                delta = Gt_index - V # δt = rt + γV (st+1) − V (st)
                advantage = delta.detach()

                action_prob = torch.log((self.actor_net(state[index]).squeeze(1)).gather(1, _temp_action[index]))
                
                ratio = torch.exp(action_prob-old_action_log_prob[index])

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage 
                action_loss = -torch.min(surr1, surr2).mean() # eq(18)
                self.actor_optimizer.zero_grad()

                action_loss.backward() # actor 参数更新
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward() # critic network 参数更新
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
    return_list = []

    #save loss
    value_loss = []
    policy_loss = []

    #选择最小的打出来
    save_energy_ep = []
    save_time_ep = []

    for i_epoch in tqdm(range(800)):
        #ppo
        ep_reward = []
        ep_energy = []
        ep_time = []


        ep_action_prob = 0
        env.reset(i_epoch) # reset the env

        cnt = 0
        for t in count():
            # done, upgrade, idx = env.env_up()
            env.env_up()
            # task不为空且 第一个task开始执行

            while env.task and env.task[0].start_time == env.time:
            #循环内说明正在执行某一个时段内同时产生的任务，cnt从进入到出循环差值小于13
            #while env.task:
                    cnt += 1
                    curr_task = env.task.pop(0)
                    # ----------ppo--------------
                    # get current state
                    node_f,task_f  = env.get_obs(curr_task)
                    # Select action at according to πθ (at | st )
                    # action: node index -> TODO: {0, 1}
                    # action_prob : 0
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
                return_list.append(np.mean(ep_reward))
                # total_times.append(env.total_time/cnt) 
                # total_energys.append(env.total_energy/cnt)
               
                #ppo
                total_times_ep = []
                total_energys_ep = []
                total_energys_ep = store_data_baseline(ep_energy,cnt) #the enegry cost of one ep
                total_times_ep = store_data_baseline(ep_time,cnt) #the time cost of one ep

                #抽取最近的50个数的均值作为输出 -->可以改为不用均值直接最小
                save_energy_ep.append(np.mean(total_energys_ep))
                save_time_ep.append(np.mean(total_times_ep))
                print('Episode: {}, reward: {}, total_time: {}'.format(i_epoch, round(np.mean(ep_reward), 3), np.mean(total_times_ep[-50:])))
                print("final result energy cost: ",min(save_energy_ep[-50:]))
                print("final result time cost: ",min(save_time_ep[-50:]))
                break
    #return return_list, value_loss, policy_loss 
    return return_list,
            
if __name__ == '__main__':

    _ = main()
 
    


 
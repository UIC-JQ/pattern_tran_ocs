import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from collections import namedtuple
from itertools import count
from env_rush_baseline import Env
import config

from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
env.seed(config.RANDOM_SEED)
torch.manual_seed(0)

feature_dim = env.feature_dim # 7个feature
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


# TODO: 改输入，输出的维度
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(f_actor, 64)
        self.fc2 = nn.Linear(64, 16)
        self.action_head = nn.Linear(16, 2)

    def forward(self, x):
        assert x.shape[1] == 86,f"the shape of the actor input is {x.shape}"
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

        node_f = torch.tensor(node_f, dtype=torch.float).unsqueeze(0).to(device)
        task_f = torch.tensor(task_f, dtype=torch.float).to(device)

        assert node_f.shape[1] == 7 # test feature_dim if it's 7
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

        assert task_f.shape == torch.Size([2]),f"the transformer input of task shape is {task_f.shape}"
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
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        #state_encoder = state.reshape(1000,7,6)
        assert state.shape == torch.Size([1000, 7, 6]), f"the shape is{state.shape}"
        state = self.embed_net(state,1000)
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

    return_list = []
    #保存优化的值
    total_times = []
    total_energys = []
    #保存loss
    value_loss = []
    policy_loss = []



    for i_epoch in tqdm(range(150)):

        #all local
        lcl_reward = []
        lcl_energy = []
        lcl_time = []
        #all egde
        edg_reward = []
        edg_energy = []
        edg_time = []
        #all random
        ran_reward = []
        ran_energy = []
        ran_time = []

        
        ep_action_prob = 0
        env.reset(i_epoch) #重置环境 节点全部重新更新
        cnt = 0
        for t in count():
            # done, upgrade, idx = env.env_up()
            env.env_up()
            # task不为空且 第一个task开始执行

            while env.task and env.task[0].start_time == env.time:
                    # print(len(env.task))

                    cnt += 1
                    curr_task = env.task.pop(0)

                    # all edge 
                    _,_, reward_e,energy_e,time_e = env.step(curr_task, 0)
                    edg_reward.append(reward_e)
                    edg_energy.append(energy_e)
                    edg_time.append(time_e)



            if not env.task: #queue里的task耗尽，计算所有数值

                #all edge
                total_times_e = []
                total_energys_e = []
                total_energys_e = store_data_baseline(edg_energy,cnt)
                total_times_e = store_data_baseline(edg_time,cnt)


                print('Episode: {}, reward: {}, total_time: {}'.format(i_epoch, round(np.mean(), 3), np.mean(total_times_ep[-50:])))
                # print('all local: total_energy: {}, total_time: {}'.format(np.mean(total_energys_l),np.mean(total_times_l)))
                # print('all edge: total_energy: {},total_time: {}'.format(np.mean(total_energys_e),np.mean(total_times_e)))
                # print('random: total_energy: {},total_time: {}'.format(np.mean(total_energys_r), np.mean(total_times_r)))

                break
    return return_list, value_loss, policy_loss  
            
if __name__ == '__main__':
    # main()
    #reward = main()
    _,V_Loss,P_Loss = main()

    # 指定要保存的CSV文件名
    # csv_filename1 = 'v_loss_data.csv'
    # csv_filename2 = 'ac_loss_data.csv'
    # # 将数组逐行写入CSV文件
    # with open(csv_filename, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["iterations","reward"])
    #     for i, value in tqdm(enumerate(reward, start=1)):
    #         csv_writer.writerow([i, value])

    #print(len(V_Loss))
    # with open(csv_filename1, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["iterations","value_loss"])
    #     for i, value in tqdm(enumerate(V_Loss, start=1)):
    #         csv_writer.writerow([i, value])
    

    # with open(csv_filename2, mode='w', newline='') as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #     csv_writer.writerow(["iterations","ac_loss"])
    #     for i, value in tqdm(enumerate(P_Loss, start=1)):
    #         csv_writer.writerow([i, value])
    


 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple
from itertools import count
from env_for_dqn import Env
import config

from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
env.seed(config.RANDOM_SEED)
torch.manual_seed(0)

feature_dim = env.feature_dim # 7个feature
f_tran = env.f_node
f_task = env.f_task
f_net = env.f_node + env.f_task #concatenate
num_action = env.node_num #server or local
n_of_node = env.node_num #server_node + WDs 


d_model = 7
TRAN_NUM_HEAD = 1
TRAN_NUM_LAYER = 2

Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'next_state']
)
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(f_net, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        result =  self.fc2(x).squeeze(0)
        return result
    
class DQN:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 10000
    size_min = 500
    batch_size = 300
    
    def __init__(self):
        self.gamma = 0.98 # 折扣因子
        self.epsilon = 0.01 # epsilon-greedy
        self.target_update = 10 # target network update frequency
        self.q_net = Qnet() # Q-net
        self.target_q_net = Qnet() # target Q-net
        self.q_net_optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
      
        self.buffer = []
        self.counter = 0
        self.update_count = 0 # the counter for update the target Q-net
        self.training_step = 0
        
    def select_action(self, task, state): 
        if np.random.random() < self.epsilon: # epsilon-greedy strategy
            action = np.random.randint(0,1) # random choose action 0 or 1
        else:
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float).to(device)
                action = self.q_net(state).argmax().item()
        return 0 if action == 0 else task.n_id
        

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter > self.size_min == 0
    
    def update(self):

        state = torch.tensor([t.state for t in self.buffer],dtype=torch.float).to(device)
        # state = torch.stack(state)
        
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
        action_zero = torch.ones_like(action)
        action_zero[action == 0] = 0
        reward = torch.tensor([t.reward for t in self.buffer]).view(-1, 1).to(device)
        next_state = torch.tensor([t.next_state for t in self.buffer],dtype=torch.float).to(device)
        dqn_losses = []

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                q_values = self.q_net(state[index]).gather(1, action_zero[index])  # Q value

                # The maximum Q-value of the next state
                max_next_q_values = self.target_q_net(next_state[index]).max(1)[0].view(-1, 1)
   
                # temporal difference error target
                # q_targets = reward[index] + self.gamma * max_next_q_values * (1 - dones)  

                q_targets = reward[index] + self.gamma * max_next_q_values # Not setting the final state
                
                # loss function
                dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
                # reset the gradients to zero
                self.q_net_optimizer.zero_grad()
                dqn_loss.backward()  
                self.training_step += 1
                dqn_losses.append(dqn_loss.item())

        if self.update_count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.update_count += 1


        print("loss is, ",np.mean(dqn_losses))
        return np.mean(dqn_losses)

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
    agent = DQN()

    return_list = []
    #保存优化的值
    total_times = []
    total_energys = []
    #保存loss
    save_loss = []


    #选择最小的打出来
    save_energy_ep = []
    save_time_ep = []

    for i_epoch in tqdm(range(150)):
        #ppo
        ep_reward = []
        ep_energy = []
        ep_time = []
        
        ep_action_prob = 0
        # env.reset(i_epoch) #重置环境 节点全部重新更新
        env.reset() #重置环境 节点全部重新更新
        cnt = 0
        for t in count():
            # done, upgrade, idx = env.env_up()
            env.env_up()
            # task不为空且 第一个task开始执行

            while env.task and env.task[0].start_time == env.time:
            #循环内说明正在执行某一个时段内同时产生的任务，cnt从进入到出循环差值小于13
                    cnt += 1
                    curr_task = env.task.pop(0)
                    # ----------ppo--------------
                    # get current state
                    state = env.get_obs(curr_task)

                    # Select action at according to πθ (at | st )
                    action = agent.select_action(curr_task,state)

                    # 1.Execute action at and obtain the reward rt
                    # 2.Get the next state st+1 
                    next_state,reward,energy,time= env.step(curr_task, action)

                    # PPO save data
                    ep_reward.append(reward)
                    ep_energy.append(energy)
                    ep_time.append(time)

                    # Store transition (st , at , rt , st+1 ) in D
                    trans = Transition(state, action, reward, next_state)

                    # for training step ....
                    if agent.store_transition(trans):
                        dqn_loss = agent.update()
                        save_loss.append(dqn_loss)
                        
                    # ----------ppo-------------- 


            if not env.task: #queue里的task耗尽，计算所有数值
                return_list.append(np.mean(ep_reward))
                
                # total_times.append(env.total_time/cnt)
                # total_energys.append(env.total_energy/cnt)
                # total_energys.append(env.total_energy/cnt)

                #ppo
                total_times_ep = []
                total_energys_ep = []
                total_energys_ep = store_data_baseline(ep_energy,cnt)
                total_times_ep = store_data_baseline(ep_time,cnt)

                #抽取最近的50个数的均值作为输出 -->可以改为不用均值直接最小
                save_energy_ep.append(np.mean(total_energys_ep))
                save_time_ep.append(np.mean(total_times_ep))
                print('Episode: {}, reward: {}, total_time: {}'.format(i_epoch, round(np.mean(ep_reward), 3), np.mean(total_times_ep[-50:])))
                print("final result energy cost: ",min(save_energy_ep[-50:]))
                print("final result time cost: ",min(save_time_ep[-50:]))
                break
    return return_list  
            
if __name__ == '__main__':

    _ = main()

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
    


 
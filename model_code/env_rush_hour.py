import config
import numpy as np
import random
from queue import PriorityQueue
from lfu import LFU
from math import log
import ast
import csv


class Node:
    def __init__(self,disk, cpu_freq, bandwidth, p_tran, p_comp,x, y):
        # self.cpu = cpu
        # self.mem = mem
        self.disk = disk
        # self.init_cpu = self.cpu
        # self.init_mem = self.mem

        self.cpu_freq = cpu_freq
        self.bandwidth = bandwidth

        self.p_tran = p_tran
        self.p_comp = p_comp

        
        self.task_queue = PriorityQueue()
        self.lfu_cache_image = {}

        self.download_finish_time = 0 
        # image states  0: Not Locally Stored  1:Downloading  2:Locally available
        self.image_list = [0] * config.IMAGE_NUM
        self.image_download_time = [0] * config.IMAGE_NUM

        self.x = x
        self.y = y

class Image:
    def __init__(self, image_size):
        self.image_size = image_size

class Task:
    def __init__(self,cpu_freq, start_time, image_id, ddl, n_id, x, y,task_size):
        # self.cpu = cpu
        # self.mem = mem
        self.cpu_freq = cpu_freq
        self.start_time = start_time
        self.image_id = image_id
        self.ddl = ddl
        self.n_id = n_id #the task is produced by which local node
        self.x = x
        self.y = y
        self.task_size = task_size


class Env:
    def __init__(self):
        self.__init_env()
        # Save the rush hour task dataset
        self.rush_hour_task_number = []
        self.hour_reader()
    
    def seed(self, seed):
        self.seed = seed

    def __init_env(self):
        self.server_num = config.SERVER_NUM
        self.local_num = config.WD_NUM
        self.node_num = config.EDGE_NODE_NUM
        self.image_num = config.IMAGE_NUM

        self.feature_dim = 7
        self.f_node = self.node_num * 7
        self.f_task = 3
        self.node = []
        self.image = []
        self.task = []

        self.time = -1
        self.reward = 0
        self.per_task_energy = 0
        self.per_task_time = 0
        self.node_bandwidth = random.uniform(config.node_band_min, config.node_band_max)

        self.total_time = 0
        self.download_time = 0
        self.trans_time = 0
        self.comp_time = 0
        self.total_energy = 0

        self.task_finish_list = []
        self.file_name = 'mix_rush_hour_0.25pec'


    def hour_reader(self):
        csv_filename =  self.file_name+'.csv'
        with open(csv_filename, mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                self.rush_hour_task_number.append(row)

    def reset(self,save_data_row):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.__init_env()

        # 1. create server nodes
        for _ in range(self.server_num):
            cpu_frq = random.randint(config.node_disk_min, config.node_disk_max) 
            self.node.append(Node(
                random.randint(config.node_disk_min, config.node_disk_max), 
                #random.randint(config.node_cpu_freq_min, config.node_cpu_freq_max),
                cpu_frq,
                random.uniform(config.node_band_min, config.node_band_max),
                cpu_frq*2,
                cpu_frq*10,

                random.random()*config.max_x,
                random.random()*config.max_y))
        for _ in range(self.local_num):
            cpu_frq_l = random.randint(config.l_cpu_freq_min, config.l_cpu_freq_max)
            self.node.append(Node(
                # random.randint(config.l_cpu_min, config.l_cpu_max), 
                # random.randint(config.l_mem_min, config.l_mem_max), 
                random.randint(config.l_disk_min, config.l_disk_max), 
                cpu_frq_l,
                random.uniform(config.l_band_min, config.l_band_max),
                cpu_frq*1,
                cpu_frq*5,
                random.random()*config.max_x,
                random.random()*config.max_y))

        # 2. create image
        for _ in range(self.image_num):
            self.image.append(Image(
                random.uniform(config.image_size_min, config.image_size_max)))
        
        
        # 3. create task
        # For each episode, read a line consisting of 200 slots
        slots_str = self.rush_hour_task_number[save_data_row]
        slots = list(map(float, slots_str))
        slots = list(map(int, slots))
        for i in range(200): 
            for _ in range(slots[i]):
                # create an image id
                randn = int(np.random.normal(self.image_num // 2, 8, 1)) 
                image_id = randn if randn >= 0 and randn <= self.image_num - 1 else random.randint(0,self.image_num - 1)
                cpu_freq = random.random()*config.task_cpu_freq_max
                ddl = i + cpu_freq / config.node_cpu_freq_max + random.uniform(0,0.2) #任务的期待ddl
                self.task.append(Task(
                    cpu_freq,
                    start_time = i, image_id = image_id, ddl = ddl,
                    n_id = random.randint(1,config.WD_NUM),
                    x = random.random()*config.max_x,
                    y = random.random()*config.max_y,
                    task_size=random.random()*config.task_size_max/1000))

        # 初始化cache
        self.lfu_cache_image = {i:LFU() for i in range(self.node_num)}

        # 4. node上初始的image
        for i in range(self.node_num):
            for _ in range(10):
                id = random.randint(0, self.image_num-1)
                if self.node[i].disk - self.image[id].image_size < 0:
                    break
                self.node[i].image_list[id] = 2
                self.node[i].disk -= self.image[id].image_size
      
        # 5. node上初始的task
        for i in range(self.node_num):
            node_x = self.node[i].x
            node_y = self.node[i].y
            for _ in range(random.randint(0,6)):
                indices = [i for i, x in enumerate(self.node[i].image_list) if x == 2]
                random_index = random.choice(indices)
                cpu_freq = random.random()*config.task_cpu_freq_max
                # TODO
                ddl = cpu_freq / config.node_cpu_freq_max + random.uniform(0,0.2)
                
                task = Task(
                    # random.random()*config.task_cpu_max,
                    # random.random()*config.task_mem_max,
                    cpu_freq,
                    start_time=0, image_id = random_index, ddl = ddl,
                    n_id = random.randint(1,config.WD_NUM),
                    x = node_x, y = node_y,
                    task_size=random.random()*config.task_size_max/1000)
                self.lfu_cache_image[i].put(random_index)
                self._add_task(task, i) #传到node i上

        # 6. node位置信息
        # xy坐标，不同的task有不同的disk，传输到不同node的时间不同，但时间不会太长

    def _add_task(self,task,idx):
        # Nodes miss task execution images and need to be downloaded.
        if self.node[idx].image_list[task.image_id] == 0:

            while self.node[idx].disk - self.image[task.image_id].image_size < 0:
                id = -1
                for i in self.lfu_cache_image[idx].get_all():
                    if self.node[idx].image_list[i] == 2:
                        id = i
                        break

                self.lfu_cache_image[idx].remove(id)
                self.node[idx].image_list[id] = 0
                self.node[idx].disk += self.image[id].image_size

            # 下载image
            self.node[idx].image_list[task.image_id] = 1
            self.node[idx].disk -= self.image[task.image_id].image_size
            self.lfu_cache_image[idx].put(task.image_id)
            
            # 下载时间
            # TODO
            download_time = self.image[task.image_id].image_size / (2*self.node[idx].bandwidth)  # 下载需要的时间： 时间段
            self.node[idx].image_download_time[task.image_id] = max(self.node[idx].download_finish_time, self.time) + download_time # 下载完成的时间：时间点

            # node上的image排队下载
            self.node[idx].download_finish_time = self.node[idx].image_download_time[task.image_id]

            download_finish_time = self.node[idx].image_download_time[task.image_id]

        # 如果正在下载这个image，获得剩余下载时间
        elif self.node[idx].image_list[task.image_id] == 1:
            download_finish_time = self.node[idx].image_download_time[task.image_id]
        # node上有这个image
        else:
            download_finish_time = max(0,self.time)
       
        comp_time = task.cpu_freq / self.node[idx].cpu_freq
        comp_energy = comp_time*self.node[idx].p_comp

        if idx == 0:
            #trans_time = task.task_size/self.node_bandwidth if up else task.task_size/self.uplink_trans_rate(task,self.node[idx])
            trans_time = task.task_size/self.uplink_trans_rate(task,self.node[idx])
            
        else:
            trans_time = 0
        download_time = download_finish_time - task.start_time

        tran_energy = trans_time*self.node[task.n_id].p_tran + download_time*self.node[idx].p_tran

        task_finish_time = download_finish_time + comp_time + trans_time

        self.node[idx].task_queue.put((task_finish_time,random.random(), task))#这里才放task, random是什么?

        total_energy = tran_energy+comp_energy
   
        #-----data saving-----
        w_t = 0.5
        self.reward = w_t*(task.ddl - task_finish_time) - (1-w_t)*total_energy
        one_task_time = task_finish_time - task.start_time
        self.per_task_energy = total_energy
        self.per_task_time = one_task_time
        
        #所有的累加会因为多次调用类多加上别的数据变得不正确
        # self.total_time += one_task_time
        # #print(self.total_time,"time")
        # #self.download_time += download_finish_time - task.start_time

        # self.total_energy += total_energy
        # #print(self.total_energy,"22222")

        # self.download_time +=  download_time
        # self.trans_time += trans_time
        # self.comp_time += comp_time
        
        #print(self.total_energy,"energy")
        task.x = self.node[idx].x
        task.y = self.node[idx].y


    # 每个时间t更新环境
    def env_up(self): 
        self.time += 1 #每次更新环境时间往下走1
        for idx, n in enumerate(self.node):
            # 判断每个node上的任务是否执行完成
            # task执行的时间不是每个t时间 -1，而是到了某个时间，task直接执行完成
            # 这样task在另外一个node新开的时候，可以记录已经执行了多少
            while n.task_queue.empty() == False:
                curr_task = n.task_queue.get() #所有task有一个总queue, 每个node中有一个queue
                if self.time >= curr_task[0]:# 0为finish time
                    # n.cpu += curr_task[2].cpu #已结束的任务将资源返回
                    # n.mem += curr_task[2].mem
                    self.task_finish_list.append(str(self.time-curr_task[2].start_time)) #当前任务结束的时间
                else:
                    n.task_queue.put(curr_task)
                    break
            
            # 判断每个node上的image是否下载完成
            for i in range(len(n.image_download_time)):
                if n.image_list[i] == 1 and self.time >= n.image_download_time[i]:
                    n.image_list[i] = 2

        return 
    def cal_dist(self, task, node):
        x = np.array([task.x, task.y])
        y = np.array([node.x, node.y])
        return np.sqrt(sum(np.power((x - y), 2)))

    def uplink_trans_rate(self, task, node):
        trans_power = 23
        noise_power = -174
        dist = self.cal_dist(task, node) / 1000 + 1e-5 # 统一单位
        # channel_gain = 127 + 30 * log(dist, 10)
        channel_gain = dist ** (-2) * 10 ** (-2)
               # channel_gain = 5 + 50 * log(dist, 2)
        gamma = (trans_power * channel_gain) / noise_power ** 2
        eta = log(1 + gamma, 2)
        return node.bandwidth * eta
    
    def get_obs(self, task):
        # create frature
        node_disk_list = np.array([n.disk / 10 for n in self.node])
        cpu_freq_list = np.array([n.cpu_freq / 3 for n in self.node])
        #t_power_list = [n.p_tran / 2 for n in self.node]
        t_power_list = np.array([n.p_tran / 10 for n in self.node])
        c_power_list = np.array([n.p_comp / 10 for n in self.node])
        bandwidth_list = np.array([n.bandwidth for n in self.node])#此处只是把数值尽量统一并没有真正使用计算

        # image state
        node_image_list = np.array([n.image_list[task.image_id] for n in self.node]) #这个任务需要的image是否在这些节点上
        download_time_list = []
        for n in self.node:
            if n.image_list[task.image_id] == 2:# locally available 
                download_time_list.append(0)
            elif n.image_list[task.image_id] == 1:# downloading
                download_time_list.append(n.image_download_time[task.image_id] - self.time)
            else:# Requires download
                download_time_list.append(max(self.time, n.download_finish_time) + self.image[task.image_id].image_size / n.bandwidth - self.time)

        #obs["image"] = np.hstack((node_image_list, download_time_list, self.image[task.image_id].image_size))
        node_f = np.hstack((node_disk_list,cpu_freq_list,t_power_list,c_power_list,bandwidth_list,node_image_list, download_time_list))
        node_f = node_f.reshape(self.node_num,self.feature_dim)
        node_f = np.transpose(node_f)
        assert node_f.shape == (self.feature_dim,self.node_num) 

        # task state
        # cpu_fre represents the required CPU cycles for the task
        task_f = [self.image[task.image_id].image_size, task.cpu_freq / 3, task.image_id]

        return node_f,task_f


            
    def step(self, task, action):

        self._add_task(task, action) #神经网络算出任务分配节点，更新环境

        # terminated = self._get_done()


        node_f,task_f = self.get_obs(task)
        return node_f,task_f, self.reward,self.per_task_energy,self.per_task_time

if __name__=='__main__':

    env = Env()
    env.seed(config.RANDOM_SEED)
    env.reset(5)
    cpu_freq = random.random()*config.task_cpu_freq_max

    task = Task(
                    # random.random()*config.task_cpu_max,
                    # random.random()*config.task_mem_max,
                    
                    cpu_freq,
                    start_time=0, image_id = 2, ddl = 2,
                    n_id = random.randint(1,config.WD_NUM),
                    x = 1, y = 1,
                    task_size=random.random()*config.task_size_max/1000)

    env.step(task,1)
    env.get_obs(task)

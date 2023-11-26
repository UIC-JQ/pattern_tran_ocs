import random
#单位为Mb
# 0. experiments
RANDOM_SEED = 300


WD_NUM = 5
SERVER_NUM = 1
EDGE_NODE_NUM = SERVER_NUM + WD_NUM

# 1. node
# node_cpu_max = 120
# node_cpu_min = 80 

# node_mem_max = 130
# node_mem_min = 70

node_disk_max = 125
node_disk_min = 75

node_cpu_freq_max = 35 #GHz
node_cpu_freq_min = 15

node_band_max = 2 #200Mb
node_band_min = 1 #100Mb

# node_p_tran= 2 #传输功率 直接用 cpu fre*2
# node_p_comp = 10 #计算功率 直接用 cpu fre*10，这样算力越高越耗能


# helen local node: weak
# l_cpu_max = 60
# l_cpu_min = 40 

# l_mem_max = 60
# l_mem_min = 20

l_disk_max = 200
l_disk_min = 100

l_cpu_freq_max = 10
l_cpu_freq_min = 5

l_band_max = 1
l_band_min = 1


# 2. image
IMAGE_NUM = 50

image_size_min = 3
image_size_max = 15

# 3. task
TASK_NUM = 100000

task_cpu_freq_max = 100 * 5 #指task需要的总cycles cpu_freq_needs
task_size_max = 0.5 * 20

# 4. pos
max_x = 100
max_y = 100

#task distribution




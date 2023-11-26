import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import numpy as np

def poltreward():
    # 读取 CSV 文件，假设文件名为 'reward_data.csv'，包含 'iteration' 和 'reward' 两列
    data = pd.read_csv('reward_data.csv')
    data_subset = data.head(200)
    # 获取 'iteration' 和 'reward' 列的数据
    iterations = data_subset['iterations']
    rewards = data_subset['reward']

    # 绘制奖励图
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rewards, marker='o', markersize=0, linestyle='-', color='b', label='Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    #plt.title('Reward Over Iterations')
    #plt.legend()


    # 将 y 轴刻度标签改为科学计数法
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # 控制指数范围
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.grid()
    plt.show()

def plotenergy():

    # 定义数据
    sensor_nodes = [5, 10, 15, 20, 25]
    all_local = [20380.41447, 20406.80463, 22161.23813, 19971.77193, 20073.21203]
    all_edge = [12333.53337, 12847.07058, 9774.038325, 10824.3885, 11510.22504]
    random = [16384.05186, 16523.52276, 16034.00036, 15246.52794, 15708.59236]
    sncs = [11073.21771, 11018.76365, 8947.491538, 9657.431713, 10279.16748]

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(10, 6))
    # 创建柱状图
    #plt.figure(figsize=(10, 6))  # 设置图的大小

   # 使用自定义颜色和样式
    bar_width = 0.15
    bar_positions = np.arange(len(sensor_nodes))

    # 自定义颜色，可以使用HTML颜色代码或命名颜色
    colors = ['#80AFBF', '#608595',  '#E2C3C9', '#C07A92']

    # 设置 x 轴位置
    x = range(len(sensor_nodes))


    # 绘制柱状图，并指定颜色和标签
    ax.bar(bar_positions, all_local, width=bar_width, color=colors[0], label='All-Local')
    ax.bar(bar_positions + bar_width, all_edge, width=bar_width, color=colors[1], label='All-MC')
    ax.bar(bar_positions + 2 * bar_width, random, width=bar_width, color=colors[2], label='random')
    ax.bar(bar_positions + 3 * bar_width, sncs, width=bar_width, color=colors[3], label='MCCS(ours)')

    # 设置 x 轴刻度标签
    ax.set_xticks([i + 1.5 * bar_width for i in x], sensor_nodes)

    # 添加标题和标签
    ax.set_xlabel('Number of WDs')
    ax.set_ylabel('Energy Cost (J)')


    # 添加图例
    ax.legend()

    # 设置 y 轴为科学计数法格式
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


    # 显示横向的网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

   
    # 显示图形
    plt.show()

def plottime():
    # 提供的数据
    sensor_nodes = [5, 10, 15, 20, 25]
    all_local = [97.66216095, 97.9540775, 95.45187236, 92.83859318, 94.69594101]
    all_edge = [53.49334091, 56.14165604, 38.67274564, 45.66144418, 49.28863077]
    random = [76.10617966, 76.74545769, 67.61903475, 68.09327974, 71.68460408]
    sncs = [43.68911455, 39.61965078, 28.90534107, 32.34677349, 35.04725853]

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(10, 6))
    # 创建柱状图
    #plt.figure(figsize=(10, 6))  # 设置图的大小

   # 使用自定义颜色和样式
    bar_width = 0.15
    bar_positions = np.arange(len(sensor_nodes))

    # 自定义颜色，可以使用HTML颜色代码或命名颜色
    colors = ['#80AFBF', '#608595',  '#E2C3C9', '#C07A92']

    # 设置 x 轴位置
    x = range(len(sensor_nodes))


    # 绘制柱状图，并指定颜色和标签
    ax.bar(bar_positions, all_local, width=bar_width, color=colors[0], label='All-Local')
    ax.bar(bar_positions + bar_width, all_edge, width=bar_width, color=colors[1], label='All-MC')
    ax.bar(bar_positions + 2 * bar_width, random, width=bar_width, color=colors[2], label='random')
    ax.bar(bar_positions + 3 * bar_width, sncs, width=bar_width, color=colors[3], label='MCCS(ours)')

    # 设置 x 轴刻度标签
    ax.set_xticks([i + 1.5 * bar_width for i in x], sensor_nodes)

    # 添加标题和标签
    ax.set_xlabel('Number of WDs')
    ax.set_ylabel('Time Cost (S)')


    # 添加图例
    ax.legend()

    # # 设置 y 轴为科学计数法格式
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


    # 显示横向的网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

   
    # 显示图形
    plt.show()

def pltweight_time():
    sensor_nodes = [5, 10, 15, 20, 25]
    all_local = [97.66216095, 97.9540775, 95.45187236, 92.83859318, 94.69594101]
    all_edge = [53.49334091, 56.14165604, 38.67274564, 45.66144418, 49.28863077]
    random = [76.10617966, 76.74545769, 67.61903475, 68.09327974, 71.68460408]
    mccs_weight_0 = [41.72608458, 38.43100017, 30.36997521, 33.52560261, 34.45801291]
    mccs_weight_0_5 = [43.68911455, 39.61965078, 28.90534107, 32.34677349, 35.04725853]
    mccs_weight_1 = [41.58871216, 36.7048606, 28.31333507, 30.38698821, 31.30672603]

     # 创建一个图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用自定义颜色和样式
    bar_width = 0.15
    bar_positions = np.arange(len(sensor_nodes))

    # 自定义颜色，可以使用HTML颜色代码或命名颜色
    colors = ['#80AFBF', '#608595',  '#E2C3C9', '#C07A92']

    # 设置 x 轴位置
    x = range(len(sensor_nodes))


    # 绘制柱状图，并指定颜色和标签
    #ax.bar(bar_positions, all_local, width=bar_width, color=colors[0], label='All-Local')
    ax.bar(bar_positions, all_edge, width=bar_width, color=colors[0], label='All-MC')
    #ax.bar(bar_positions + 2 * bar_width, random, width=bar_width, color=colors[2], label='random')
   

    ax.bar(bar_positions + 1* bar_width, mccs_weight_0, width=bar_width, color=colors[3], label='MCCS(w_t = 0)')
    ax.bar(bar_positions + 2*bar_width, mccs_weight_0_5, width=bar_width, color=colors[1], label='MCCS(w_t = 0.5)')
    ax.bar(bar_positions + 3 * bar_width, mccs_weight_1, width=bar_width, color=colors[2], label='MSSC(w_t = 1)')
    # 设置 x 轴刻度标签
    ax.set_xticks([i + 1.5 * bar_width for i in x], sensor_nodes)

    # 添加标题和标签
    ax.set_xlabel('Number of  WDs')
    ax.set_ylabel('Time Cost (s)')


    # 添加图例
    ax.legend()

    # 设置 y 轴为科学计数法格式
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


    # 显示横向的网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

   
    # 显示图形
    plt.show()


def poltvloss():
    # 读取 CSV 文件
    data = pd.read_csv('v_loss_data.csv')
    data_subset = data.head(200)
    # 获取 'iteration' 和 'reward' 列的数据
    iterations = data_subset['iterations']
    rewards = data_subset['value_loss']

    # 绘制奖励图
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rewards, marker='o', markersize=0, linestyle='-', color='b', label='Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Value Function Loss')


    # 将 y 轴刻度标签改为科学计数法
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # 控制指数范围
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.grid()
    plt.show()

def pltacloss():
    # 读取 CSV 文件
    data = pd.read_csv('ac_loss_data.csv')
    data_subset = data.head(200)
    # 获取 'iteration' 和 'reward' 列的数据
    iterations = data_subset['iterations']
    rewards = data_subset['ac_loss']

    # 绘制奖励图
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rewards, marker='o', markersize=0, linestyle='-', color='b', label='Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Policy Network Loss')


    # 将 y 轴刻度标签改为科学计数法
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # 控制指数范围
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.grid()
    plt.show()


def pltweight_energy():
    sensor_nodes = [5, 10, 15, 20, 25]
    all_local = [20380.41447, 20406.80463, 22161.23813, 19971.77193, 20073.21203]
    all_edge = [12333.53337, 12847.07058, 9774.038325, 10824.3885, 11510.22504]
    random = [16384.05186, 16523.52276, 16034.00036, 15246.52794, 15708.59236]
    sncs_weight_0_5 = [11073.21771, 11018.76365, 8947.491538, 9657.431713, 10279.16748]
    sncs_weight_1 = [10954.1125, 10890.95809, 8918.852242, 9716.726483, 10120.63925]
    sncs_weight_0 = [11188.3741, 11906.77928, 9303.892931, 10183.01218, 10467.04425]

     # 创建一个图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用自定义颜色和样式
    bar_width = 0.15
    bar_positions = np.arange(len(sensor_nodes))

    # 自定义颜色，可以使用HTML颜色代码或命名颜色
    colors = ['#80AFBF', '#608595',  '#E2C3C9', '#C07A92']

    # 设置 x 轴位置
    x = range(len(sensor_nodes))


    # 绘制柱状图，并指定颜色和标签
    #ax.bar(bar_positions, all_local, width=bar_width, color=colors[0], label='All-Local')
    ax.bar(bar_positions, all_edge, width=bar_width, color=colors[0], label='All-MC')
    #ax.bar(bar_positions + 2 * bar_width, random, width=bar_width, color=colors[2], label='random')
   

    ax.bar(bar_positions + 1* bar_width, sncs_weight_0, width=bar_width, color=colors[3], label='MCCS(w_t = 1)')
    ax.bar(bar_positions + 2*bar_width, sncs_weight_0_5, width=bar_width, color=colors[1], label='MCCS(w_t = 0.5)')
    ax.bar(bar_positions + 3 * bar_width, sncs_weight_1, width=bar_width, color=colors[2], label='MSSC(w_t = 0)')
    # 设置 x 轴刻度标签
    ax.set_xticks([i + 1.5 * bar_width for i in x], sensor_nodes)

    # 添加标题和标签
    ax.set_xlabel('Number of  WDs')
    ax.set_ylabel('Energy Cost (J)')


    # 添加图例
    ax.legend()

    # 设置 y 轴为科学计数法格式
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


    # 显示横向的网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

   
    # 显示图形
    plt.show()
# plotenergy()
# plottime()
# pltweight_energy()
# pltweight_time()
# poltvloss()
# pltacloss()

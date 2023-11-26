try list
- head number 
    改为1 比较有用
- 改learnning rate 
    self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-8)
    self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-14)
    乘积为e-2数量级
    没用 58.393

- 不要batch norm #结果没变好 52.629 
    没什么用
- encoder提取后 和feature拼在一起
    41.54450456258267好，很有用 已比纯ppo强
- 去掉position encoding  做消融实验
    稍微有用
- 增加训练轮数
    6000轮加去位置编码 5600
    40.74007621749795 又好了一点但不多也许是数据设置的问题
-增加task feature embedding更好了 到31
-改正了原来错误的实现，原本的 all loacl和 random实现错误

# find bugs
- baseline实现错误 all local 不应该都是 1 相当于都传到节点1 了
- 关于action_prob的作用
class Segment():
    def __init__(self, low, slope, base):
        self.low = low
        self.slope = slope
        self.base = base
    
    def process(self, x, y):
        xp = x - self.low
        y[xp >= 0] = (self.slope * xp).to(y.dtype)[xp >= 0] + self.base
        return y

class DiTi_cont():
    def __init__(self, n_timesteps, K, stages, k_per_stage):
        self.K = K
        assert k_per_stage
        k_per_stage = k_per_stage.split(",")
        self.k_per_stage = [int(k) for k in k_per_stage]
        assert stages
        stages = stages.split(",")
        self.stages = [int(k) for k in stages]
        n_stages = len(self.stages)
        self.stages = [0] + self.stages
        self.segments = []
        acc = 0
        for i in range(n_stages):
            self.segments.append(Segment(
                self.stages[i], float(self.k_per_stage[i]) / (self.stages[i+1]-self.stages[i]), acc
            ))
            acc += self.k_per_stage[i]

    def to_indices(self, t):
        ind = torch.zeros_like(t)
        for segment in self.segments:
            ind = segment.process(t, ind)
        return ind.to(torch.long).clamp(0, self.K - 1)
    
    def get_position(self, k):
        return 1000 + (k * 8)

import torch
import numpy as np
import matplotlib.pyplot as plt

diti = DiTi_cont(1000, 512, '200,400,600,800,1000', '192,184,72,48,16')

cnt = torch.zeros(512)
c = 0
while True:
    t = torch.rand(10)
    k_batch = diti.to_indices(t * 1000.0)
    # print(k_batch.min())
    # print(k_batch.max())
    for i in range(k_batch.shape[0]):
        cnt[:(k_batch[i]+1)] += 1
    
    c += 1
    # if c == 1000 or c == 10000 or c == 50000 or c == 100000 or c == 250000 or c == 500000 or c == 1000000 or c == 10000000:
    if c == 50000:

        print(cnt)
        weight = [10*c/cnt[i] for i in range(512)]
        # print(weight)
        # print([round(weight[i].item(),1) for i in range(512)])
        print([weight[i].item() for i in range(512)])
        # import pdb; pdb.set_trace()
        x = np.arange(512)  # 如果你希望从1开始，可以使用 np.arange(1, 513)

        # 绘制柱状图
        plt.bar(x, weight, width=0.8, align='center', alpha=0.7, color='blue')
        plt.xticks(x) 
        # 添加标题和标签
        plt.title('Bar Chart of Tensor Values')
        plt.xlabel('Position')
        plt.ylabel('weight')

        # 显示图表
        print(c)
        plt.savefig("weight.png")
        break
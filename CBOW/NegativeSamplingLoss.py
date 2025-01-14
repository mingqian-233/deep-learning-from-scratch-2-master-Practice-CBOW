import numpy as np


class embedding:  # embed层其实就是取出其中一层（几层）的权重矩阵
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.index = None

    def forward(self, index):
        (W,) = self.params  # 注意这里有解包操作，要把Params拆封为一行行的
        self.index = index
        return W[index]

    def backward(self, dout):
        


class UnigramSampler:
    def __init__():
        pass


class NegativeSamplingLoss:
    # 在初始化的时候，传入参数权重W，语料库corpus，以及负采样的次数sample_size
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

import numpy as np


class Embedding:  # embed层其实就是取出权重矩阵中的其中一个词（几个词）
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.index = None

    def forward(self, index):
        (W,) = self.params
        # 注意这里有解包操作，如果不解包的话，W就是一个列表，而不是一个numpy矩阵。并且是浅拷贝。
        self.index = index
        return W[index]  # index是列表，抽出这个列表里面的数字对应的行

    def backward(self, dout):
        dW = self.grads  # 这个地方直接引用（浅拷贝）了self.grads
        dW[...] = 0
        np.add.at(dW, self.index, dout)
        return None


class EmbeddingDot:
    # 在上面抽取的基础上，加上了和h的点积
    def __init__(self, W):
        self.embed = Embedding(W)
        # 要先把权重矩阵转化为Embedding对象
        se = self.embed
        self.params = se.params
        self.grads = se.grads
        self.cache = None
        # 用于存入正向传播的中间结果，以便反向传播时使用

    def forward(self, h, index):
        se = self.embed
        targetW = se.forward(index)
        out = np.sum(targetW * h, axis=1)
        # 哈夫曼乘，结果是一个矩阵（因为用了mini_batch），然后再对这个矩阵的每一行求和，得到一个98行的向量，长度和mini_batch的大小一样。这个向量就是这一层的输出
        self.cache = (h, targetW)
        # 常识：圆括号括起来是元组，方括号括起来是列表
        return out

    def backward(self, dout):
        h, targetW = self.cache
        # dout是长为mini_batch的行向量，转置dout为列向量
        dout = dout.reshape(dout.shape[0], 1)


class UnigramSampler:
    def __init__():
        pass


class NegativeSamplingLoss:
    # 在初始化的时候，传入参数权重W，语料库corpus，以及负采样的次数sample_size
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

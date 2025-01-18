import cupy as np
import cupyx as cpx


class Embedding:  # embed层其实就是取出权重矩阵中的其中一个词（几个词）
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.index = None

    def forward(self, index):
        (W,) = self.params  # 浅拷贝
        # 注意这里有解包操作，如果不解包的话，W就是一个列表，而不是一个numpy矩阵。
        self.index = index
        return W[index]  # index是列表，抽出这个列表里面的数字对应的行

    def backward(self, dout):
        (dW,) = self.grads  # 这个地方直接引用（浅拷贝）了self.grads
        dW[...] = 0
        cpx.scatter_add(dW, self.index, dout)
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
        targetW = se.forward(index)  # 抽出词向量
        out = np.sum(targetW * h, axis=1)
        # 哈夫曼乘，结果是一个矩阵（因为用了mini_batch），然后再对这个矩阵的每一行求和，得到长度为batch_size的向量（一维矩阵）。这个向量就是这一层的输出
        # a[0][1][2][3]... axis=x表示把对应的第x维压缩掉

        self.cache = (h, targetW)
        # 常识：圆括号括起来是元组，方括号括起来是列表

        return out

    def backward(self, dout):
        h, targetW = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        # 变成一个第一维为batch_size，第二维为1的矩阵，才能和h相乘
        dtargetW = dout * h
        # 这里广播了，dout是有batch_size行1列的矩阵，h是一个有batch_sizes行的矩阵，广播后dout的每一行的那个数字都和h的每一个元素相乘
        self.embed.backward(dtargetW)  # 这个是Dot层的反向传播

        dh = dout * targetW
        return dh


class UnigramSampler:
    def __init__():
        pass


class NegativeSamplingLoss:
    # 在初始化的时候，传入参数权重W，语料库corpus，以及负采样的次数sample_size
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

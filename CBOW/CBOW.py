"""
网络结构：输入层->隐藏层->输出层
输入层：用Embedding层提取出词向量，相加求平均值
隐藏层&输出层：用Negative正负采样，计算损失
"""

import sys

sys.path.append("..")
import cupy as np
from NegativeSamplingLoss import Embedding
from NegativeSamplingLoss import NegativeSamplingLoss as NSL
import copy



class CBOW:
    
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        
        W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype("f")
        W_out = 0.01 * np.random.randn(vocab_size, hidden_size).astype("f")

        self.in_num = 2*window_size
        
        self.in_layers = []
        for i in range(self.in_num):
            one_layer = Embedding(W_in)
            self.in_layers.append(one_layer)
        self.ns_loss = NSL(W_out, corpus, power=0.75, sample_size=5)
        
        layers = self.in_layers + [self.ns_loss]
        self.params=[param for layer in layers for param in layer.params]
        self.grads=[grad for layer in layers for grad in layer.grads]
        
        self.word_vecs=W_in
        
        return None

    
    def forward(self,contexts,target):
        h=0
        i=0
        # print(len(self.in_layers))
        for layer in self.in_layers:
            h+=layer.forward(contexts[:,i])
            i+=1
        h*=1.0/self.in_num
        
        loss=self.ns_loss.forward(h,target)
        return loss
    
    def backward(self,dout=1):
        dout=self.ns_loss.backward(dout)
        dout*=1.0/self.in_num
        
        for layer in self.in_layers:
            layer.backward(dout)
            
        return None
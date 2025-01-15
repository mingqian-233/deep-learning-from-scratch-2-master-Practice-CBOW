# 深度学习进阶-自然语言处理练习（部分）

## 描述
个人在阅读书籍的时候的自行学习。common和dataset来源于原书 [deep-learning-from-scratch-2](https://github.com/oreilly-japan/deep-learning-from-scratch-2/tree/master)，其余代码大多数都为本人学习书本内容后自行编写。

因为是完全看着书写的，所以写法肯定和书上别无二致。


---

## 对部分原代码的修改
由于高版本的cupy已经不支持cupy.scatter_add（该功能自 CuPy v4 起已被弃用），需要从cupyx中调用，因此:


1. 在np.py（原书用于控制是否开启GPU加速）中添加：
```python
    import cupyx as cpx
```

2. 修改common\layers.py下的Embedding层实现中的np.scatter_add(dW, self.idx, dout)为cpx.scatter_add(dW, self.idx, dout)
import tensorflow as tf 
import numpy as np 

class SKRM:
    #初始化
    def __init__(self):
        self.kernel = tf.load_op_library('./count_skrm.so')
        self.tensorShape = tf.zeros([8],tf.int64)
        self.count = tf.zeros([8],tf.int64)
        self.store = []

    #取得紀錄
    def GetCount(self):
        return self.count 
    
    #清除count紀錄
    def Reset(self):
        self.count = tf.zeros([8],tf.int64)

    #將目前的結果儲存起來
    def Store(self):
        self.store.append(self.count)
        self.Reset()
    
    #根據兩個tensor來計算
    def Count(self, tensorA, tensorB):
        self.count = self.count + self.kernel.count_skrm(tensorA, tensorB, self.tensorShape)
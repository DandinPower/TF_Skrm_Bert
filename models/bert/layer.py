import tensorflow as tf
from tensorflow.python.framework import ops
from .skrm import SKRM
import numpy as np

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim,skrms):
        super().__init__()
        self.skrms = skrms
        self.w = self.add_variable(name='w',
            shape=[input_dim, output_dim], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
            shape=[output_dim], initializer=tf.zeros_initializer())

    def call(self, inputs):
        matmul = tf.matmul(inputs, self.w)
        bias = matmul + self.b
        #self.skrms.Count(matmul, bias)
        #self.skrms.Count(tf.convert_to_tensor(inputs, dtype=tf.float32), tf.convert_to_tensor(matmul, dtype=tf.float32))
        return bias

class AddNorm(tf.keras.Model):
    def __init__(self, dropout,skrms):
        super(AddNorm, self).__init__()
        self.skrms = skrms
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(axis = 2)

    def call(self, inputs):
        (X,Y) = inputs
        add = self.dropout(Y) + X 
        output = self.ln(add)
        self.skrms.Count(Y, add)
        self.skrms.Count(add, output)
        return output

class PositionWiseFFN(tf.keras.Model):
    def __init__(self, config, parameters,index,skrms):
        super(PositionWiseFFN, self).__init__()
        self.config = config 
        self.parameters = parameters 
        self.skrms = skrms
        self.index = index 
        self.dense1 = LinearLayer(config.ffnNumInput, config.ffnNumHiddens,skrms)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = LinearLayer(config.ffnNumHiddens, config.ffnNumInput,skrms)

    def call(self, X):
        output1 = self.dense1(X)
        output2 = self.relu(output1)
        output3 = self.dense2(output2)
        self.skrms.Count(X, output1)
        self.skrms.Count(output1, output2)
        self.skrms.Count(output2, output3)
        return output3

    def LoadParameters(self):
        self.dense2.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense1.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense2.bias"]])
        self.dense1.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense2.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense1.bias"]])
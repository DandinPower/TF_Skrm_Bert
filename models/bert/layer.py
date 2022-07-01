import tensorflow as tf
from tensorflow.python.framework import ops
from .skrm import SKRM

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
        self.skrms.Count(inputs, matmul)
        bias = matmul + self.b
        self.skrms.Count(matmul, bias)
        return bias

class AddNorm(tf.keras.Model):
    def __init__(self, dropout,skrms):
        super(AddNorm, self).__init__()
        self.skrms = skrms
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(axis = 2)

    def call(self, inputs):
        (X,Y) = inputs
        return self.ln(self.dropout(Y) + X)

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
        return self.dense2(self.relu(self.dense1(X)))

    def LoadParameters(self):
        self.dense2.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense1.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense2.bias"]])
        self.dense1.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense2.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense1.bias"]])
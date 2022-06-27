from .encoder import BERTEncoder
from .block import EncoderBlock
from .layer import LinearLayer
import tensorflow as tf
import time 
from ..train.timer import GetTimeByDict
from tensorflow.python.framework import ops

@ops.RegisterGradient("CountSkrm")
def _count_skrm_grad(op, grad):
  return [grad] 

class BERTModel(tf.keras.Model):
    def __init__(self, config, parameters,skrms):
        super(BERTModel, self).__init__()
        self.skrms = skrms
        self.parameters = parameters
        self.encoder = BERTEncoder(config,parameters,skrms)
        self.block1 = EncoderBlock(config,parameters,0,skrms,True)
        self.block2 = EncoderBlock(config,parameters,1,skrms,True)
        self.hidden = tf.keras.Sequential()
        tempLinearLayer = LinearLayer(config.numHiddens, config.numHiddens,skrms)
        tempLinearLayer.set_weights([parameters["hidden.0.weight"],parameters["hidden.0.bias"]])
        self.hidden.add(tempLinearLayer)
        self.hidden.add(tf.keras.layers.Activation('tanh'))

    def call(self, inputs):
        (tokens, segments, valid_lens) = inputs
        embeddingX = self.encoder((tokens,segments))
        X1 = self.block1((embeddingX, valid_lens))
        X2 = self.block2((X1, valid_lens))
        X3 = self.hidden(X2[:, 0, :])
        return X3

    def LoadParameters(self):
        self.encoder.LoadParameters()
        self.block1.LoadParameters()
        self.block2.LoadParameters()

class BERTClassifier(tf.keras.Model):
    def __init__(self, config, parameters, skrms):
        super(BERTClassifier, self).__init__()
        self.config = config 
        self.skrms = skrms
        self.parameters = parameters
        self.bert = BERTModel(config, self.parameters,skrms)
        self.classifier = LinearLayer(config.numHiddens, 2,skrms)

    def call(self, tokens):
        tempSegments = tokens * 0
        tempValid = self.GetValidLen(tokens)
        inputs = (tokens,tempSegments,tempValid)
        output1 = self.bert(inputs)
        output2 = self.classifier(output1)
        result = tf.nn.softmax(output2)
        self.skrms.Count(output2,result)
        return result

    def GetValidLen(self,inputs):
        tokens = inputs
        padding = tf.constant(1)
        temp = tf.math.equal(tokens, padding)
        paddingNums = tf.math.count_nonzero(temp,axis=1,dtype=tf.dtypes.float32)
        paddingNums = self.config.maxLen - paddingNums
        return paddingNums

    def LoadParameters(self):
        self.bert.LoadParameters()


    


    
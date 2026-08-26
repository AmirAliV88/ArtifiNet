import os, re, safeserialize, numpy as np
from collections import Counter, defaultdict
from . import *



######################################################################################################
#* Neurons ( @----<< )


#:: A Micro-Node of Neural Networks
class Neuron:
   __slots__ = ("W", "Bias")
   def __init__(_, W, Bias=.1): _.W, _.Bias = W, Bias
   __call__ = GET = lambda _,X: X*_.W+_.Bias

#:: A Micro-Node of RNN
class RNeuron:
   __slots__ = ("Wx", "Wh", "Bias")
   def __init__(_, Wx, Wh, Bias=.1): _Wx, _.Wh, _.Bias = Wx, Wh, Bias
   __call__ = GET = lambda _,X, hp: (X*_.Wx)+(hp*_.Wh)+_.Bias

INSTALLED_NEURONS = [ Neuron, RNeuron ]


######################################################################################################
#* Layers
#:: A part of Neural Networks


class Layer:
   def __init__(_, Args:dict, Index=None, Act:str="Relu", Dropout=0):
      _.W, _.Bias, _.LastCall = Array(Args["W"]), Array(Args["Bias"]), None
      _.Index, _.Inputs, _.Dropout, _.__Act_str__, _.Act = Index, _.W.shape[1], Dropout, Act, eval(f"_.{Act}")
      _.TrainModeOff()

   Neurons = lambda _: [ Neuron(w,b) for w,b in zip(_.W.T, _.Bias.T) ]
   ReLU = Relu = lambda _,Z: np.maximum(0, Z)
   Sigmoid = lambda _,Z: 1/(1+np.exp(-Z))
   Linear = lambda _,Z: Z #: pass X
   Tanh = lambda _,Z: np.tanh(Z)
   def Softmax(_,Z): 
      e = np.exp(Z-np.max(Z, axis=-1, keepdims=True))
      return e / np.sum(e, axis=-1, keepdims=True)

   def _a(_):
      Z, A = _.LastCall[1], _.LastCall[2]
      match _.__Act_str__:
         case "Relu": return (Z>0).astype(float)
         case "ReLU": return (A>0).astype(float)
         case "Linear": return np.ones_like(A)
         case "Softmax": return np.ones_like(A) #<< Softmax derivative is Jacobian, but we handle it separately 
         case "Sigmoid": return A*(1-A)
         case "Tanh": return 1-A**2
      raise ValueError(f'Activation {_.Act.lower()} not supported')

   def GET(_, X, Return_Thinks=False):
      Z = Array(X) @_.W + _.Bias; A =_.Act(Z)
      Mask = A*npra.binomial(1, 1-_.Dropout, Z.shape)/(1-_.Dropout) if _.Train and _.Dropout else A
      _.LastCall = (X,Z,A,Mask); return _.LastCall if Return_Thinks else Mask
   __call__ = GET

   def backward(_, dA, Lr):
      W_, dZ, m = _.W.copy(), dA*_._a(), _.LastCall[0].shape[0]
      _.W -= Lr* _.LastCall[0].T @ dZ /m; _.Bias -= Lr*np.sum(dZ, axis=0, keepdims=True) /m #$ Bias -= Lr*dB
      return dZ @ W_.T

   def TrainModeOff(_): _.Train = False
   def TrainModeOn(_): _.Train = True


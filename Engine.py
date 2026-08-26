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


######################################################################################################


#? Neural Networks
#@ Artifical Neural Network | ANN
class ANN:
   def __init__(_, NetName=None, Layers:list=[], ImportData=None): _.NetName = NetName; _.Layers = Layers or _.__Load__(ImportData)

   def __Load__(_, ImportData=None):
      if ImportData: _.__setstate__(ImportData)
      else: 
         with open(f"./db/{_.NetName}.Ai", 'rb') as File: _.__setstate__(safeserialize.load(File))
      return _.Layers

   def SaveNetwork(_, AsFile=None): #$ --> File
      Domain = { 
         I:{ "Activation":L.__Act_str__, "Dropout":L.Dropout, "Args":{ "W":L.W, "Bias":L.Bias } } for I,L in enumerate(_.Layers)
      }
      Domain = {"__class__":"ANN", "NetName":_.NetName, ">>":Domain} 
      with open(AsFile or f"./db/{_.NetName}.Ai", "wb") as File: safeserialize.dump(Domain, File)
   Save = SaveModel = SaveNetwork

   def AddLayer(_, Weights, Bias, Activation="Relu", Dropout=0):
      newLayer = Layer( Index=len(_.Layers)+1, Args={"W":Weights, "Bias":Bias}, Act=Activation, Dropout=Dropout )
      _ += newLayer

   __iadd__ = lambda _,newLayer: (_, _.Layers.append(newLayer)) [0]
   __getitem__ = lambda _,i: _.Layers[i]
   __len__ = lambda _: len(_.Layers)

   def Loss(_, Yp, Y, Type="MSE"):
      Y = Y[:, None] if Y.ndim == 1 else Y
      match Type:
         case "MSE":  Loss, dA = Mean((Y-Yp)**2), 2*(Yp-Y)/len(Y)
         case "MAE":  Loss, dA = Mean(np.abs(Y-Yp)), np.sign(Yp-Y)/len(Y)
         case "RMSE": Loss, dA = np.sqrt(Mean((Y-Yp)**2)), (Yp-Y)/(len(Y)*np.sqrt(Loss))
      return Loss, dA

   def Accuracy(_, Yp, Y, Type="R2"):
      match Type:
         case "R2": return 1-( np.sum((Y-Yp)**2) / (np.sum((Y-Mean(Y))**2)+1e-8) )
         case "Binary":
            Yc = np.argmax(Y, axis=1) if len(Y.shape)>1 else Y.astype(int)
            return Mean( (Yp>.5).astype(int).flatten() == Yc.flatten())
         case "MultiClass": return Mean(np.argmax(Yp, axis=1) == np.argmax(Y, axis=1))

   def Activity(_, X): #<< Auto Return Thinks
      S, A, Z = [], [], [], #<< S: Inputs(Signals)
      for L in _.Layers:
         s,z,_,X = L(X, Return_Thinks=True); S.append(s); A.append(X); Z.append(z)
      return S,A,Z
   Thinking = Activity

   def Training(_, XT, YT):
      #<< Training_Options:  Lrate*, Epochs*:int, BatchSize=1, LossMethod="MSE", AccuracyMethod="R2", AutoSave=True, EarlyStop=False, shuffle=False
      if not hasattr(_, "Training_Options"): raise AttributeError("Training allowed just with Net.Training_Options")
      
      O, XT, YT, Best = _.Training_Options, Array(XT), Array(YT), float('inf')
      for L in _: L.TrainModeOn()

      for e in range(O["Epochs"]+1):
         tLoss, tAcc, Bs = [], [], O.get("BatchSize", 1)

         if O.get("shuffle") or O.get("Shuffle"):
            Sh = npra.permutation(len(XT)); XT, YT = XT[Sh], YT[Sh]

         for k in range(0, len(XT), Bs):
            X, Y = XT[k:k+Bs], YT[k:k+Bs]; A = _.Thinking(X)[1]
            Loss, dA = _.Loss(A[-1], Y, O.get("LossMethod", "MSE")); tLoss.append(Loss)
            for L in reversed(_): dA = L.backward(dA, O["Lrate"])
            tAcc.append(_.Accuracy(A, Y, O.get("AccuracyMethod"))) #<< Accuracy

         os.system("cls" if os.name=="nt" else "clear")
         print(f"Epoch: {e:4d} \t|\t Loss: {Mean(Array(tLoss)):.6f} \t|\t Accuracy: {Mean(Array(tAcc)):.2%} ")

         if O.get("EarlyStop"):
            if Loss<BestLoss: BestLoss = Loss
            else:
               print(" EARLY STOPPING !!! "); break

      for L in _: L.TrainModeOff()
      if O.get("AutoSave", True): _.SaveModel()
      return _

   def GET(_,X):
      for l in _.Layers: X = l.GET(X)
      return X
   __call__ = GET



import os, re, numpy as np
from collections import defaultdict
from . import *



######################################################################################################
#* Neurons ( @----<< )


#:: A Micro-Node of Neural Networks
class Neuron:
   __slots__ = ("W", "Bias")
   def __init__(_, W, Bias=.1): _.W, _.Bias = W, Bias
   __call__ = GET = lambda _,X: X*_.W+_.Bias

#:: A Micro-Node of ENN|AENN
class ENeuron:
   __slots__ = ("W", "e", "Bias")
   def __init__(_, W, e, Bias=.1): _.W, _.e, _.Bias = W, e, Bias
   __call__ = GET = lambda _,X,E: np.sum(X*_.W)+np.sum(E*_.e)+_.Bias

#:: A Micro-Node of RNN
class RNeuron:
   __slots__ = ("Wx", "Wh", "Bias")
   def __init__(_, Wx, Wh, Bias=.1): _Wx, _.Wh, _.Bias = Wx, Wh, Bias
   __call__ = GET = lambda _,X, hp: (X*_.Wx)+(hp*_.Wh)+_.Bias

INSTALLED_NEURONS = [ Neuron, ENeuron, RNeuron ]


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
      _.W -= Lr* _.LastCall[0].T @ dZ /m; _.Bias -= Lr*np.sum(dZ, axis=0) /m #$ Bias -= Lr*dB
      return dZ @ W_.T

   def TrainModeOff(_): _.Train = False
   def TrainModeOn(_): _.Train = True


class ELayer(Layer):
   def __init__(_, Args:dict, Index=None, Act:str="Relu", Dropout=0):
      _.W, _.We, _.Bias, _.LastCall = Array(Args["W"]), Array(Args["We"]), Array(Args["Bias"]), None
      _.Index, _.Inputs, _.Dropout, _.__Act_str__, _.Act = Index, _.W.shape[1], Dropout, Act, eval(f"_.{Act}")
      _.TrainModeOff()

   Neurons = lambda _: [ ENeuron(w,e,b) for w,e,b in zip(_.W.T, _.We.T, _.Bias.T) ]

   def GET(_, X, E, Return_Thinks=False):
      Z = Array(X) @_.W + np.sum(E*_.We, axis=1) +_.Bias; A = _.Act(Z)
      Mask = A*npra.binomial(1, 1-_.Dropout, Z.shape)/(1-_.Dropout) if _.Train and _.Dropout else A
      _.LastCall = (E, X, Z, A, Mask); return _.LastCall if Return_Thinks else Mask
   __call__ = GET

   def backward(_, dA, Lr):
      W_, dZ, m = _.W.copy(), dA*_._a(), _.LastCall[1].shape[0]
      _.W -= Lr* _.LastCall[1].T @ dZ /m; _.Bias -= Lr*np.sum(dZ, axis=0) /m #$ Bias -= Lr*dB
      return dZ @ W_.T

   def Ebackward(_, dA, Lr):
      E = _.LastCall[0]; _.We -= Lr * E.T @ dA*_._a() / E.shape[0]  #$ dWe = E * dZ   |   We = Lr * sum(dWe) /m
      return dZ @ _.W.T


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


#? Neural Network
#@ ENN | AENN | Artificial emotional neural networks | شبکه های عصبی احساسی مصنوعی
class ENN(ANN):

   SterioEnv = lambda _: np.zeros(_[0].We.shape[1])

   def SaveNetwork(_, AsFile=None): #$ --> File
      Domain = { 
         I:{ "Activation":L.__Act_str__, "Dropout":L.Dropout, "Args":{ "W":L.W, "We":L.We, "Bias":L.Bias } } for I,L in enumerate(_.Layers)
      }
      Domain = {"__class__":"ANN", "NetName":_.NetName, ">>":Domain} 
      with open(AsFile or f"./db/{_.NetName}.Ai", "wb") as File: safeserialize.dump(Domain, File)
   Save = SaveModel = SaveNetwork

   def AddLayer(_, Weights, We, Bias, Activation="Relu", Dropout=0):
      newLayer = Layer( Index=len(_.Layers)+1, Args={"W":Weights, "We":We, "Bias":Bias}, Act=Activation, Dropout=Dropout )
      _ += newLayer

   def Activity(_, X, E=None): #<< Auto Return Thinks
      E = E if type(E)==np.ndarray else _.SterioEnv(); S, A, Z = [], [], [], #<< S: Inputs (Signals)
      for L in _: 
         _, s, z, __, X = L(X, E, Return_Thinks=True); S.append(s); A.append(X); Z.append(z)
      return S,A,Z
   Thinking = Activity

   def ETraining(_, XT, YT, E):
      #<< Training_Options:  Lrate*, Epochs*:int, BatchSize=1, LossMethod="MSE", AccuracyMethod="R2", AutoSave=True, EarlyStop=False
      if not hasattr(_, "Training_Options"): raise AttributeError("Training allowed just with Net.Training_Options")

      O, XT, YT, Best, Bs = _.Training_Options, Array(XT), Array(YT), float('inf'), _.Training_Options.get("BatchSize", 1)
      for L in _: L.TrainModeOn()

      for epo in range(O["Epochs"]+1):
         tLoss, tAcc = [], []

         for k in range(0, len(XT), Bs):
            X, Y = XT[k:k+Bs], YT[k:k+Bs]; A = _.Thinking(X, E)[1]
            Loss, dA = _.Loss(A[-1], Y, O.get("LossMethod", "MSE")); tLoss.append(Loss)
            for L in reversed(_): dA = L.backward(dA, O["Lrate"])
            tAcc.append(_.Accuracy(A, Y, O.get("AccuracyMethod"))) #<< Accuracy

         os.system("cls" if os.name=="nt" else "clear")
         print(f"Epoch: {epo:4d} \t|\t Loss: {Mean(Array(tLoss)):.6f} \t|\t Accuracy: {Mean(Array(tAcc)):.2%} ")

         if O.get("EarlyStop"):
            if Loss<Best: Best = Loss
            else:
               print(" EARLY STOPPING !!! "); break         

      for L in _: L.TrainModeOff()
      if O.get("AutoSave", True): _.SaveModel()
      return _

   def GET(_, X, E):
      for l in _: X = l.GET(X, E)
      return X
   __call__ = GET


#? Algoritm
#@ KNN Algoritm
class KNN:
   def __init__(_, X, Y, W="Uniform", Meteric="Euclidean"): _.X, _.Y, _.W, _.meteric = Array(X), Array(Y), W, Meteric #: meterics: Manhattan or Euclidean

   def SpaceMeter(_, X):
      match _.meteric:
        case "Euclidean": return np.sqrt(np.sum((X-_.X)**2, axis=1)) 
        case "Manhattan": return np.sum(np.abs(X-_.X), axis=1) 
   
   def __NK__(_, X, k=3): #? کلید همسایه
      O = _.SpaceMeter(X).argsort()[:k] #? متراژ
      Xp, Yp = Array(_.X)[O], Array(_.Y)[O]
      return (O, Xp, Yp)

   def _Predict_1(_, X, k=3):
      D = _.SpaceMeter(X); O = np.argsort(D)[:k]
      if _.W == "Uniform": #? رای اکثریت یا همون مد گیری
         a,b = np.unique(_.Y[O], return_counts=True); return a[np.argmax(b)]
      else:  #? Distance
         S = {}
         for l, w in zip(_.Y[O], 1/(D+1e-8)): S[l] = S.get(l, 0)+w
         return max(S, key=S.get)
   __call__ = predict = _Predict_1

   def predict_proba(_, X, k=3):
      X, probas = Array(X), []
      if X.ndim == 1: X = X.reshape(1, -1)
      for x in X:
         O, D = _.__NK__(x, k); L, uniq = _.Y[O], np.unique(_.Y)      
         if _.W == "Uniform": proba = {l: Counter(L).get(l, 0)/k for l in uniq}
         else:
            S = { l:0 for l in uniq }
            for l, w in zip(L, 1/(D+1e-8)): S[l] += w
            proba = { l: S[l]/sum(S.values()) for l in uniq }
         probas.append([proba[l] for l in uniq])
      return Array(probas)
   P_class = predict_proba

   Score = lambda _,X,Y: Mean( Array([ _(x) for x in X ]) == Y )
   __repr__ = lambda _: f"<KNN Metric='{_.meteric}'>"


class Tokenizer:
   def __init__(_): _.Vocab, _.spT, _.PadI, _.UNKI, _.SI, _.EI, _.Minfreq = [], ['<PAD>', '<UNK>', '<START>', '<END>'], 0, 1, 2, 3, 1
   __len__ = nVoc = lambda _: len(_.Vocab)
   __repr__ = lambda _: f"<T:{_.Voc}>"
   #TODO :  Save and Load

   def fit(_, Documents, MaxVocabSize=10000, Minfreq=1):
      _.Minfreq, WordC = Minfreq, defaultdict(int)
      for doc in Documents:
         for W in _._T_(doc): WordC[W] += 1
      _.Vocab = _.spT
      for W, fr in sorted(WordC.items(), key=lambda x:x[1], reverse=True):
         if fr>=Minfreq and len(_)<MaxVocabSize and W not in _.Vocab: _.Vocab.append(W)
      _.W2I, _.I2W = { W:I for I,W in enumerate(_.Vocab) }, { I:W for I,W in enumerate(_.Vocab) }
      return _

   @staticmethod
   def _T_(T):
      T = T.lower()
      T = re.sub(r'[.,!?;:"()\[\]{}]', ' ', T.lower()) #<< جایگزینی علائم نگارشی با فاصله
      T = re.sub(r"'", " ' ", T) #<< مدیریت آپاستروف (مثل don't -> don ' t)
      T = re.sub(r'\s+', ' ', re.sub(r"-", " - ", T)).strip()
      return Tokenizer.Normalize(T).split()

   @staticmethod
   def Normalize(T:str)->str:
      return T.strip() .replace("ي", "ی") .replace("ى", "ی") .replace("ك", "ک") .replace("ۀ", "ه") .replace("ـ", "") .replace("\u200c", "")

   def Correcter(_, W):
      if not W or not W.strip(): return W
      Orgi, Tar = W, _.Normalize(W)
      for vw in _.Vocab:
         if _.Normalize(vw) == Tar: return vw
      Bw, Bdist = Orgi, float("inf")

      for vw in _.Vocab:
         if abs(len(Tar)-len(C))>3: continue
         prev = list(range(len(C)+1))
         for i, ca in enumerate(Tar, start=1):
            cur = [i]
            for j, cb in enumerate(_.Normalize(vw), start=1):
               cost = 0 if ca==cb else 1
               cur.append(min(
                  cur[j-1]+1,     # حذف
                  prev[j]+1,      # اضافه
                  prev[j-1]+cost  # جایگزینی
               ))
            prev = cur
         dist = prev[-1]
         if dist<Bdist: Bdist, Bw = dist, vw

      Maxdist = 1 if len(Tar)<=4 else 2 if len(Tar)<=8 else 3
      return Bw if Bdist <= Maxdist else Orgi

   def encode(_, T, Maxn=10, Padd="R", Turncate="R"): #? R: Right  | L: Left
      if Maxn<3: raise ValueError("Maxn must be at least 3 (for START and END Tokens)")
      I = [ _.W2I.get(T, _.UNKI) for T in _._T_(T) ]
      if len(I)>(Maxn-2): I = I[:(Maxn-2)] if Turncate == "R" else I[-(Maxn-2):]
      R = [_.SI]+I+[_.EI]
      if Maxn-len(R)>0: R = R+[_.PadI]*(Maxn-len(R)) if Padd == "R" else [_.PadI]*(Maxn-len(R))+R
      return R

   def decode(_, I, IgnoreSpecial=True, asList=False):
      W = []
      for i in I:
         if IgnoreSpecial and i in [_.PadI, _.UNKI, _.SI, _.EI]: continue
         W.append(_.I2W.get(i, "<UNK>"))
      return W if asList else ' '.join(W)




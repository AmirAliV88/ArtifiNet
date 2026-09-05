import os, re, safeserialize, numpy as np
from collections import defaultdict, Counter
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

   Neurons = staticmethod( lambda _: [ Neuron(w,b) for w,b in zip(_.W.T, _.Bias.T) ] )
   ReLU = Relu = staticmethod( lambda _,Z: np.maximum(0, Z) )
   Sigmoid = staticmethod( lambda _,Z: 1/(1+np.exp(-Z)) )
   Linear = staticmethod( lambda _,Z: Z ) #: pass X
   Tanh = staticmethod( lambda _,Z: np.tanh(Z) )
   @staticmethod
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


#? Algoritm
#@ Test Tokenizer and Test Normalizer
class Tokenizer:
   def __init__(_): _.Vocab, _.spT, _.PadI, _.UNKI, _.SI, _.EI, _.Minfreq = [], ['<PAD>', '<UNK>', '<START>', '<END>'], 0, 1, 2, 3, 1
   __getstate__ = lambda _: { "Vocab":_.Vocab, "Minfreq":_.Minfreq }   #$ --> File
   __setstate__ = lambda _,D: (_, _.__dict__.update(D)) [0]            #$ <-- File
   __len__ = nVoc = lambda _: len(_.Vocab)
   __repr__ = lambda _: f"<T:{_.Voc}>"

   def Save(_, AsFile=None):
      with open(AsFile or f"./db/{float(nprArr(1))}.Tokenizer", "wb") as File: 
         safeserialize.dump({ "Vocab":_.Vocab, "Minfreq":_.Minfreq }, File)
      return _
   
   def __Load__(_, FromFile=None, ImportData=None):
      if not bool(FromFile)^bool(ImportData): raise ImportError("!: FromFile or ImportData ?")
      if FromFile:
         with open(FromFile, "rb") as File: _.__dict__.update(safeserialize.load(File))
      elif ImportData: _.__dict__.update(ImportData)
      return _

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


#? Algoritm
#@ Word Embeddings | Word to Vec | Word2Vec
class Embeddings: 
   def __init__(_, dim=20, padd=2, negative_samples=5, Lr=.01, epochs=100):
      _.dim, _.padd, _.negs, _.Lr, _.epochs = dim, padd, negative_samples, Lr, epochs
      _.Domain, _.S, _.Voc, _.W2I, _.I2W, _.V, = [], [], {}, {}, {}, 0
      _.W_center, _.W_ct, _.Embedds = None, None, None
      _.Pairs, _.Tokenizer = [], Tokenizer()
    
   Tokenize = lambda _,T: _.Tokenizer._T_(T)
    
   def _BuildVocab(_):
      Wordc = Counter([w for s in _.S for w in s])
      _.W2I = {w: i for i, (w, _) in enumerate(Wordc.items())}
      _.I2W = {i: w for w, i in _.W2I.items()}
      _.Voc, _.V = Wordc, len(Wordc)
    
   def _BuildTrainingPairs(_):
      _.Pairs = []
      for s in _.S:
         for i, c in enumerate(s):
            cti = list(range(max(0, i-_.padd), min(len(s), i+_.padd+1) ))
            cti.remove(i)
            for j in cti: _.Pairs.append((_.W2I[c], _.W2I[s[j]]))
      _.Pairs = Array(_.Pairs, dtype=np.intp)
    
   def fit(_, Domain):
      _.Tokenizer.fit(Domain)
      _.Domain, _.S = Domain, [ _.Tokenize(S) for S in Domain ]
      _._BuildVocab(); _._BuildTrainingPairs(); _._init_Weights(); _.Train(); _.BuildEmbedds(); _.Tokenizer.fit(Domain)
      return _
    
   def _init_Weights(_): _.W_center, _.W_ct = npra.uniform(-0.5, 0.5, (_.V, _.dim)), npra.uniform(-0.5, 0.5, (_.V, _.dim))
    
   _Sigmoid = staticmethod( lambda x: 1/(1+np.exp(-np.clip(x, -50, 50))) )

   def _negsUp(_, ci, cti): #! این متد خیلی مهم نی
      if not (0<=ci<_.V): raise IndexError("ci out of range")
      if not (0<=cti<_.V): raise IndexError("cti out of range")
      # if _.negs>_.V-1: raise ValueError("negs must be <= V -1")
      ni = npra.choice(_.V-1, _.negs, replace=False); ni[ni>=cti] += 1
      vc, vw, vn = _.W_center[ci], _.W_ct[cti], _.W_ct[ni].copy()
      ps, pn = _._Sigmoid(np.dot(vc, vw)), _._Sigmoid(vn @ vc); Gc = (ps-1)*vw+pn@vn
      _.W_center[ci] -= _.Lr*Gc; _.W_ct[cti] -= _.Lr*(ps-1)*vc; _.W_ct[ni] -= _.Lr*pn[:,None]*vc

   def Train(_, EchoEpo=True):
      if _.V <= 1: return
      for epo in range(_.epochs):
         if EchoEpo: print(f"Epoch: {epo}") 
         npra.shuffle(_.Pairs)
         for ci, cti in _.Pairs:
            ni = npra.choice(_.V-1, _.negs, replace=False); ni[ni>=cti] += 1
            vc, vw, vn = _.W_center[ci], _.W_ct[cti], _.W_ct[ni].copy()
            ps, pn = _._Sigmoid(np.dot(vc, vw)), _._Sigmoid(vn @ vc); Gc = (ps-1)*vw+pn@vn
            _.W_center[ci] -= _.Lr*Gc; _.W_ct[cti] -= _.Lr*(ps-1)*vc; _.W_ct[ni] -= _.Lr*pn[:,None]*vc
    
   def BuildEmbedds(_): _.Embedds = (_.W_center+_.W_ct)/2
    
   def solve(_, eq, topn=1):
      eq = eq.strip()
      if not eq.split(): return []
      TarVec, excW, op = np.zeros(_.dim), [], "+"
      for t in eq.split():
         if t in ['+', '-']: op = t
         else:
            if t not in _.W2I: return []
            excW.append(t)
            TarVec += -1*_.Embedds[_.W2I[t]] if op=="-" else _.Embedds[_.W2I[t]]
      fi = _._Find(TarVec, ex=excW, topn=topn)
      return fi[0] if topn==1 and fi else fi
   
   def FMS(_, W, topn=5):
      if W not in _.W2I: return []  
      V, sim = _.Embedds[_.W2I[W]], []
      for ow, i in _.W2I.items():
         if ow == W: continue
         cos_sim = np.dot(V, _.Embedds[i]) / ( np.linalg.norm(V) * np.linalg.norm(_.Embedds[i]) + 1e-8 )
         sim.append((ow, cos_sim))
      sim.sort(key=lambda x: x[1], reverse=True)
      return sim[:topn]
   find_most_similar = FMS

   def _Find(_, TarVec, ex=None, topn=5, metric='cosine'):
      ex, S = (ex or []), []  
      for W, i in _.W2I.items():
         if W in ex: continue
         Vw = _.Embedds[i]
         if metric == 'cosine': s = np.dot(TarVec, Vw) / ( np.linalg.norm(TarVec) * np.linalg.norm(Vw) + 1e-8 )
         elif metric == 'euclidean': s = -np.linalg.norm(TarVec - Vw)
         elif metric == 'dot': s = np.dot(TarVec, Vw)
         else: s = np.dot(TarVec, Vw) / ( np.linalg.norm(TarVec) * np.linalg.norm(Vw) + 1e-8 )   
         S.append((W, s))
      S.sort(key=lambda x: x[1], reverse=True)
      return S[:topn]
   
   GetVec = lambda _,W: _.Embedds[_.W2I[W]] if W in _.W2I else None

   def Save(_, AsFile):
      _.Tokenizer = { "Vocab":_.Tokenizer.Vocab, "Minfreq":_.Tokenizer.Minfreq }
      with open(AsFile, 'wb') as f: safeserialize.dump(_.__dict__, f)
      return _

   def __Load__(_, FilePath):
      with open(FilePath, 'rb') as f: data = _.__dict__.update(safeserialize.load(f))
      _.Tokenizer = Tokenizer().__Load__(ImportData=_.Tokenizer)
      return _

   def _FitWithPreTrained_(_, Dom, PreTrainedPath=None, SavePath=None):
      if PreTrainedPath and _.__Load__(PreTrainedPath):        
         _.Domain, _.S = Dom, [ _.Tokenize(s) for s in Domain]
         _._BuildVocab()
         for W, i in _.W2I.items():
            if i >= len(_.W_center):
               nc, nct = npra.uniform(-.5, .5, (1, _.dim)), npra.uniform(-.5, .5, (1, _.dim))
               _.W_center, _.W_ct = np.vstack([_.W_center, nc]), np.vstack([self.W_ct, nct])
      _._BuildTrainingPairs()
      if not PreTrainedPath: _._init_Weights()
      _.Train(); _.Build() 
      if SavePath: _.Save(SavePath)
      return _
   _FPT_ = _FitWithPreTrained_




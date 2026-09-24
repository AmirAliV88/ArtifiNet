from . import *
from .Engine import *
import numpy as np



INSTALLED_OPTIMIZERS = [ BaseOptimizer, SGD, SGDM, SGDMomentum, Adagrad, RMSProp, Adadelta, Adam, NAdam, AdamW, RAdam ]

INSTALLED_ORGANS = INSTALLED_NEURONS + + INSTALLED_OPTIMIZERS + [ 
   Layer, ELayer, ANN, ENN, KNN, KMeans, SVM, KNN, Tokenizer, Embeddings, LeafNode, DecisionTree, RandomForest 
]

ORGANS_GEN_INDEXS = {
   "Neuron":(0, 1)
   "ENeuron":(2, 4)
   "RNeuron":(5, 7)
   "Layer":(8, 11)
   "ELayer":(12, 15)
   "SGD":(16, 17)
   "SGDM":(18, 21)
   "Adagrad":(22, 23)
   "RMSProp":(24, 26)
   "Adadelta":(27, 29)
   "Adam":(30, 33)
   "NAdam":(34, 35)
   "AdamW":(36, 40)
   "RAdam":(41, 44)
   "ANN":(45, 48)
   "ENN":(49, 52)
   "K-Menas":(53, 56)
}

ORGANS_REQUIREMENTS = {

   #$* Neuron
   0:lambda x:nprArr(x),
   1:lambda x:nprArr(x),

   #$* ENeuron
   2:lambda x:nprArr(x),
   3:lambda x:nprArr(x),
   4:lambda x:nprArr(x),

   #$* RNeuron
   5:lambda x:nprArr(x),
   6:lambda x:nprArr(x),
   7:lambda x:nprArr(x),

   #$* Layer
   8: lambda w,n,b: {"W":nprArr((w,n))/20, "Bias":nprArr((1,b))/50},
   #<< + 9 10 11

   #$* ELayer
   12: lambda w,n,e,b: {"W":nprArr((w,n))/20, "We":nprArr((e,w))/40, "Bias":nprArr((1,b))/50},
   #<< + 13 14 15

   #$* SGD
   16:lambda w,n:nprArr((w,n)),
   #<< + 17

   #$* SGDM
   18:lambda w,n:nprArr((w,n)),
   #<< + 19 20 21

   #$* AdaGrad
   22:lambda w,n:nprArr((w,n)),
   #<< + 23

   #$* RMSProp
   24:lambda w,n:nprArr((w,n)),
   #<< + 25 26

   #$* Adadelta
   27:lambda w,n:nprArr((w,n)),
   #<< + 28 29

   #$* Adam
   30:lambda w,n:nprArr((w,n)),
   #<< + 31 32 33

   #$* NAdam
   34:lambda w,n:nprArr((w,n)),
   #<< + 35

   #$* AdamW
   36:lambda w,n:nprArr((w,n)),
   #<< + 37 38 39 40

   #$* RAdam
   41:lambda w,n:nprArr((w,n)),
   #<< + 42 43 44

   #$* ANN
   45:lambda n:str(n),
   46:lambda Net,Act,out:[ 
      Layer(Args={"W":nprArr((Net[i-1], n))/20, "Bias":nprArr((1,n))/50} Act=Act, Dropout=out) for i,n in enumerate(Net[1:]) 
   ]
   #<< + 47 48

   #$* ENN
   49:lambda n:str(n),
   50:lambda Net,e,Act,out:[ 
      Layer(Args={"W":nprArr((Net[i-1], n))/20, "We":nprArr((e,n))/40, "Bias":nprArr((1,n))/50} Act=Act, Dropout=out) 
      for i,n in enumerate(Net[1:]) 
   ]
   #<< + 51 52

   #$* K-Menas
   #<< + 53 54 55
   56:lambda x:str(x)

}



class WeightInitializer:
   XAVIER_ACTIVATIONS, HE_ACTIVATIONS = [ "Sigmoid", "Tanh", "Softmax", "Linear", "None" ], [ "Relu" ]
   VALID_METHODS, VALID_DISTRIBUTIONS = [ "Auto", "Xavier", "Glorot", "He", "Kaiming" ], [ "Normal", "Uniform" ]
   def __init__(_, Activation="None", method="Auto", distribution="Normal", Seed=None ):
      _.Act, _.Method, _.dist = _._norm_a(Activation), _._norm_m(method), _._norm_d(distribution)
      _.nprrng = npra.default_rng(Seed)
      if _.Method == "Auto": _.Method = _._select_method()

   @staticmethod
   def _norm_a(Act):
      if not isinstance(Act, str): raise TypeError("Type of Act != string")
      return Act.strip().replace("-", "_").replace(" ", "_")

   @staticmethod
   def _norm_m(method):
      if not isinstance(method, str): raise TypeError("Type of method != string")
      method = method.strip()
      if method not in WeightInitializer.VALID_METHODS: raise ValueError(f"{method} is not in \n {sorted(WeightInitializer.VALID_METHODS)}")
      return "Xavier" if method=="Glorot" else "He" if method=="Kaiming" else method

   @staticmethod
   def _norm_d(dist):
      if not isinstance(dist, str): raise TypeError("Type of distribution != string")
      dist = dist.strip()
      if dist not in WeightInitializer.VALID_DISTRIBUTIONS: raise ValueError(f"{dist} is not in \n {sorted(WeightInitializer.VALID_DISTRIBUTIONS)}")
      return dist

   def _select_method(_):
      # return "He" if _.Act in _.HE_ACTIVATIONS else "Xavier" if _.Act in _.XAVIER_ACTIVATIONS else "Xavier"
      return "He" if _.Act in _.HE_ACTIVATIONS else "Xavier" #<< in default -> Xavier

   @staticmethod
   def calcf(shape):

      if not isinstance(shape, (tuple, list)): raise TypeError("Type of shape must be tuple|list")
      if len(shape) == 0: raise ValueError("len(shape)==0 | shape is empty -_-")
      shape = tuple(shape)
      for dim in tuple(shape):
         if not isinstance(dim, (int, np.integer)): raise TypeError("shape.data != Numberic")
         if dim<=0: raise ValueError("How can I Weighting 0 data -_-")

      if len(shape)==1: fi, fout = shape[0], shape[0]
      elif len(shape) == 2: fi, fout = shape
      else:
         rfs = int(np.prod(shape[:-2])) #<< receptive_field_size
         fi, fout = rfs*shape[-2], rfs*shape[-1]

      return fi, fout

   def __Xavier__(_, shape):
      fin, fout = _.calcf(shape); deno = fin+fout #<< denominator
      if _.dist == "Normal": return _.nprrng.normal( loc=0, scale=np.sqrt(2/deno), size=shape )
      else:
         limit = np.sqrt(6/deno); return _.nprrng.uniform( low=-limit, high=limit, size=shape )

   def __He__(_, shape):
      fin, __ = _.calcf(shape)
      g = np.sqrt(2)
      if _.dist == "Normal": return _.nprrng.normal( loc=0.0, scale=g/np.sqrt(fin), size=shape )
      else:
         limit = np.sqrt(3)*g/np.sqrt(fin)
         return _.nprrng.uniform( low=-limit, high=limit, size=shape )

   def __call__(_, shape, dtype=np.float64):
      _.calcf(shape)
      if _.Method == "Xavier": return _.__Xavier__(shape).astype(dtype)
      elif _.Method == "He": return _.__He__(shape).astype(dtype)
      else: raise RuntimeError(f"Error: Method = {_.Method} ?")

   __repr__ = lambda _: f"<WeightInitializer:[ Act={_.Act}  Method={_.Method} ]>"


#* Cell Core | Genome | Generator
class Core:

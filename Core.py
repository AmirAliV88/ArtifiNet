from . import *
from .Engine import *
import numpy as np



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


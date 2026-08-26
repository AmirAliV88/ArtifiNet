import numpy as np
from . import *



######################################################################################################
#* Neurons ( @----<< )


#:: A Micro-Node of Neural Networks
class Neuron:
   __slots__ = ("W", "Bias")
   def __init__(_, W, Bias=.1): _.W, _.Bias = W, Bias
   __call__ = GET = lambda _,X: X*_.W+_.Bias

INSTALLED_NEURONS = [ Neuron ]


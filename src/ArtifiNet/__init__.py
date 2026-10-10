import numpy as np

Array = lambda A,dtype=np.float32: np.array(A, dtype=dtype)
Rint = lambda x,dec=4: np.round(x, decimals=dec) if type(x)==np.ndarray else float(np.round(x, decimals=dec))
Mean = lambda x,dec=4: Rint(np.mean(x), dec=dec)
npra, nprArr = np.random, np.random.random

from .Engine import *
from .Core import *

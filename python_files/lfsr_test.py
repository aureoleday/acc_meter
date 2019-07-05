#!/usr/bin/python3

import numpy as py
from pylfsr import LFSR

fpoly = [4,2,1]
state = [0,0,0,1]
L = LFSR(fpoly, state, 'true')
L.info()
L.runKCycle(32)
L.info()

e#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:37:16 2020

@author: aureoleday
"""
# import pandas as pd
# import socket
# import threading
# import time
# import struct
# import queue
# import serial
# import ctypes


import numpy as np
# from scipy import signal
# from ringbuf import RingBuffer
import matplotlib.pyplot as plt 
# from matplotlib import animation
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# df = pd.read_table("~/share/dat.txt")
a1 = np.loadtxt('/home/aureoleday/share/dat.txt',int)
# arr = np.array(df).T[0]

plt.plot(a1)
# arr = np.loadtxt("~/share/dat.txt")
# arr

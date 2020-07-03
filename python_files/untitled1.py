#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:29:32 2020

@author: aureoleday
"""
import numpy as np

def calc_n(n):
    a = 0
    for i in list(range(n))[1:]:
        a += 1/(2*i - 1)**2
#        print(a)
    return a

print(icalc_n(100000))


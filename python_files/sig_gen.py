#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:16:43 2020

@author: aureoleday
"""
import os
duration = 5000  # second
freq = 470  # Hz
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
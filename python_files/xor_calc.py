# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:06:03 2019

@author: Administrator
"""



def calc_xor(din):
    temp = 0
    for item in din:
        temp = temp^item
    return temp

# a = [0x1bdf9bdf,0x00020002,0x00010001,0x00000000]
# a = [0x1b,0xdf,0x10,0x01,0x01]
a = [0x10,0x01,0x01]

print("%x" % calc_xor(a))

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:45:01 2019

@author: Administrator
"""

def temp_2p(x,ta1,ts1,ta2,ts2):
    k = (ta2-ta1)/(ts2-ts1)
    b = (ta1*ts2-ta2*ts1)/(ts2-ts1)
    res = k*x + b
    return k,b,res


def temp_cal(temp,k,b):
    return (1+k)*temp + b

print(temp_2p(27.24,0,0.09,40,40.04))

#print(temp_cal(40,-0.00175,0.09))


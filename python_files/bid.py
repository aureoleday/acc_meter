# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:01:17 2019

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt 

my_bid       = 215
my_bid_cnt   = 3
op_bid_start = 180
op_bid_end   = 220
op_bid_cnt   = 1
bid_step     = 0.1

my_offer = 215


def calc_base(mb,mbc,opbs,opbe,opbc,bid_step):
    
    x = np.arange(opbs,opbe,bid_step)
    y = (x*opbc +mb*mbc)/(mbc+opbc)
    y = y*0.6 + 88
    
    return x,y

def calc_score(my_offer,base):
    temp = my_offer - base
    for i in range(len(temp)):
        if temp[i]<0:
            temp[i] *= -0.8
    return temp

#def get_base(bids):
#    return np.dot(bid.T[0],bid.T[1])/np.sum(bid.T[1])

x,y = calc_base(my_bid,my_bid_cnt,op_bid_start,op_bid_end,op_bid_cnt,bid_step)
z = calc_score(my_offer,y)

fig = plt.figure()
ax = plt.subplot(211)
ax.plot(x,z,label='diff')
ax.legend()
ax.set_title("d")
bx = plt.subplot(212)
bx.plot(x,y,label='base')
bx.legend()



#print(calc_base(my_bid,my_bid_cnt,op_bid_start,op_bid_end,op_bid_cnt,bid_step))


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def calc_base(mb,mbc,opbs,opbe,opbc,bid_step,alpha):
    
    x = np.arange(opbs,opbe,bid_step)
    y = (x*opbc +mb*mbc)/(mbc+opbc)
    y = (y*0.6 + 88)*(1-alpha/100)
    
    return x,y

def calc_score(my_offer,base):
    temp = (my_offer - base)*100/base
    for i in range(len(temp)):
        if temp[i]<0:
            temp[i] *= -0.8
    ret = (100-temp)*0.6
    return ret

my_bid       = 216
my_bid_cnt   = 3
op_bid_start = 180.0
op_bid_end   = 220.0
op_bid_cnt   = 2
bid_step     = 0.1
alpha        = 2

my_offer = 215
op_offer = 180

x,y = calc_base(my_bid,my_bid_cnt,op_bid_start,op_bid_end,op_bid_cnt,bid_step,alpha)
z = calc_score(my_offer,y)
z0 = calc_score(x,y)

#plt
fig = plt.figure()
ax = fig.add_subplot(311)
bx = fig.add_subplot(312)


ax.set_xlabel('op avg bid')
ax.set_ylabel('score')
lineax0, = ax.plot(x,z,label='my_score')
lineax1, = ax.plot(x,z0,label='op_score')
ax.grid()
ax.legend()
bx.set_title("my:%0.2f@%d, op_cnt:%0.2f@%d" % (my_bid,my_bid_cnt,op_offer,op_bid_cnt))
bx.set_xlabel('op avg bid')
bx.set_ylabel('base')
linebx0, = bx.plot(x,y,label='call',color='red')
bx.grid()

target_bid = Slider(plt.axes([0.15, 0.01, 0.7, 0.02]), 'my_offer', valmin=op_bid_start, valmax=op_bid_end, valinit=216.67)

av_bid = Slider(plt.axes([0.15, 0.035, 0.7, 0.02]), 'my_avg_bid', valmin=op_bid_start, valmax=op_bid_end, valinit=214.0)

salpha = Slider(plt.axes([0.15, 0.06, 0.7, 0.02]), 'my_avg_bid', valmin=0.8, valmax=2, valinit= 0.8)

def update(event):
    fig.canvas.draw_idle()
    x,y = calc_base(av_bid.val,my_bid_cnt,op_bid_start,op_bid_end,op_bid_cnt,bid_step,salpha.val)
    z = calc_score(target_bid.val,y)
    z0 = calc_score(x,y)
    bx.set_title("my:%0.2f@%d, op_cnt:%d" % (av_bid.val,my_bid_cnt,op_bid_cnt))
    lineax0.set_ydata(z)
    lineax1.set_ydata(z0)
    
    ax.set_ylim(np.min([z,z0]),np.max([z,z0]))    
    
    linebx0.set_ydata(y)
    bx.set_ylim(np.min(y),np.max(y))

    print(np.argmax(z)*bid_step+x[0])

target_bid.on_changed(update)
av_bid.on_changed(update)
salpha.on_changed(update)
plt.show()

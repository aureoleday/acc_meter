# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def calc_base(mb,mbc,opbs,opbe,opbc,bid_step):
    
    x = np.arange(opbs,opbe,bid_step)
    y = (x*opbc +mb*mbc)/(mbc+opbc)
    y = (y*0.6 + 88)*0.986
    
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

my_offer = 215
op_offer = 180

x,y = calc_base(my_bid,my_bid_cnt,op_bid_start,op_bid_end,op_bid_cnt,bid_step)
z = calc_score(my_offer,y)
z0 = calc_score(x,y)



#plt
fig = plt.figure()
ax = fig.add_subplot(211)
bx = fig.add_subplot(212)

#ax = plt.subplot2grid((7,1),(0,0),rowspan=3)
#bx = plt.subplot2grid((7,1),(1,0),rowspan=3)

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

left, bottom, width, height = 0.15, 0.01, 0.7, 0.02
xlims = [0, 1]
slider_ax = plt.axes([left, bottom, width, height])
target_bid = Slider(slider_ax, 'my_offer', valmin=op_bid_start, valmax=op_bid_end, valinit=xlims[1])

left, bottom, width, height = 0.15, 0.035, 0.7, 0.02
slider_bx = plt.axes([left, bottom, width, height])
av_bid = Slider(slider_bx, 'my_avg_bid', valmin=op_bid_start, valmax=op_bid_end, valinit=xlims[1])

#left, bottom, width, height = 0.1, 0.02, 0.7, 0.02
#xlims = [0, 1]
#slider_bx = plt.axes([left, bottom, width, height])
#slider1 = Slider(slider_bx, 'op_offer', valmin=op_bid_start, valmax=op_bid_end, valinit=xlims[1])

def update(event):
    fig.canvas.draw_idle()
    x,y = calc_base(av_bid.val,my_bid_cnt,op_bid_start,op_bid_end,op_bid_cnt,bid_step)
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
plt.show()

#ax.set_title("my_bid:%0.2f@%d, op_cnt:%d" % (my_bid,my_cnt,op_cnt))
#ax.plot(bvo,bid_do,label='opponent')
#ax.plot(bvo,bid_ds,label='self')
#ax.set_xlabel('op avg bid')
#ax.set_ylabel('score')
#ax.legend()
#ax.grid()


#def calc_adiff(bvo,bvo_c,bvs,bvs_c):
#    bva = (bvo*bvo_c + bvs*bvs_c)/(bvo_c+bvs_c)
#    bva = bva*0.6 + 88.0
#    bid_do = (bvo - bva)*100/bva
##    bid_do = (bvo - bva)
#    for i in range(len(bid_do)):
#        if bid_do[i] < 0.0:
#            bid_do[i] *= -0.8
#    bid_do = (100 - bid_do)*0.6
#    
#    
#    bid_ds = (bvs - bva)*100/bva
##    bid_ds = (bvs - bva)
#    for i in range(len(bid_ds)):
#        if bid_ds[i] < 0.0:
#            bid_ds[i] *= -0.8    
#    bid_ds = (100 - bid_ds)*0.6
#    return bid_do,bid_ds,bva
#
#
#my_bid = 216.0
#op_cnt = 3
#my_cnt = 3
#precision = 0.01
#start = 200.0
#end = 220.0
#
#
#bvo = np.arange(start,end,precision)
#bvo_c = np.zeros(bvo.shape[0]) + op_cnt
#bvs = np.zeros(bvo.shape[0]) + my_bid
#bvs_c = np.zeros(bvo.shape[0]) + my_cnt
#
#bid_do,bid_ds,bva = calc_adiff(bvo,bvo_c,bvs,bvs_c)
#
##plt
#fig = plt.figure()
#ax = fig.add_subplot(211)
#bx = fig.add_subplot(212)
#cx = fig
#
#
#ax.set_xlabel('op avg bid')
#ax.set_ylabel('score')
#lineax0, = ax.plot(bvo,bid_do,label='opponent')
#lineax1, = ax.plot(bvo,bid_ds,label='self')
#ax.grid()
#bx.set_xlabel('op avg bid')
#bx.set_ylabel('optimal call bid')
#linebx0, = bx.plot(bvo,bva,label='call',color='red')
#bx.grid()
##ax.set_title("my_bid:%0.2f@%d, op_cnt:%d" % (my_bid,my_cnt,op_cnt))
##ax.plot(bvo,bid_do,label='opponent')
##ax.plot(bvo,bid_ds,label='self')
##ax.set_xlabel('op avg bid')
##ax.set_ylabel('score')
##ax.legend()
##ax.grid()
##
##bx.set_title("my_bid:%0.2f@%d, op_cnt:%d" % (my_bid,my_cnt,op_cnt))
##bx.plot(bvo,bva,label='call',color='red')
##bx.set_xlabel('op avg bid')
##bx.set_ylabel('optimal call bid')
##bx.grid()
##print(np.argmax(bid_do)*precision+start)
##print(np.argmax(bid_ds)*precision+start)
##print(max(bid_do))
#
#left, bottom, width, height = 0.15, 0.02, 0.7, 0.02
#xlims = [0, 1]
#slider_ax = plt.axes([left, bottom, width, height])
#slider = Slider(slider_ax, 'x-limits', valmin=180, valmax=220.0, valinit=xlims[1])
#
#
#
#def update(val):
#    fig.canvas.draw_idle()
#    bvo = np.arange(start,end,precision)
#    bvo_c = np.zeros(bvo.shape[0]) + op_cnt
#    bvs = np.zeros(bvo.shape[0]) + val
#    bvs_c = np.zeros(bvo.shape[0]) + my_cnt    
#    bid_do,bid_ds,bva = calc_adiff(bvo,bvo_c,bvs,bvs_c)
#    ax.set_title("my_bid:%0.2f@%d, op_cnt:%d" % (my_bid,my_cnt,op_cnt))
#    lineax0.set_ydata(bid_do)
#    lineax1.set_ydata(bid_ds)
#    ax.set_ylim(np.min(bid_do),np.max(bid_do))
##    ax.plot(bvo,bid_do)
##    ax.plot(bvo,bid_ds)    
#    
#    
#    bx.set_title("my_bid:%0.2f@%d, op_cnt:%d" % (my_bid,my_cnt,op_cnt))
#    linebx0.set_ydata(bva)
#    bx.set_ylim(np.min(bva),np.max(bva))
##    bx.plot(bvo,bva,label='call',color='red')
#
#    print(np.argmax(bid_do)*precision+start)
#    print(np.argmax(bid_ds)*precision+start)
#    
#
#
#slider.on_changed(update)
#plt.show()








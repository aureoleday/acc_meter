#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:28:57 2021

@author: aureoleday
"""

import numpy as np
import operator
import matplotlib.pyplot as plt 
import pandas as pd
from enum import Enum
import time
from functools import reduce

#  引入ccxt框架， 通过pip install ccxt 可以进行安装
#ccxt 的github地址为： https://github.com/ccxt/ccxt
import ccxt

np.set_printoptions(suppress=True,precision=0,threshold=np.inf)

def update_av(timeout):
    ex_list = ccxt.exchanges
    # ex_id = 'binance'
    av_list = []
    for idx in ex_list:
        ex_class = getattr(ccxt,idx)
        ex = ex_class({
            'timeout':timeout
            })    
        try:
            h = ex.fetch_ticker('BTC/USDT')
        except:
            print('NA for %s' %idx)
        else:
            av_list.append(idx)
            print('OK for %s' %idx)  
    pd.DataFrame(av_list).to_csv('available_ex.csv')
    return av_list

def get_av():    
    df = pd.read_csv('available_ex.csv')
    return np.array(df.iloc[:,1])
    # return dfFalse

def get_ex(ex_name,timeout):
    ex_class = getattr(ccxt,ex_name)
    ex = ex_class({
        'timeout':timeout
        })
    
    return 
  

def get_ohlcv(exchange,pair,freq,sin,lim):
    # limit = 1000  
    print(exchange)
    ex = eval('ccxt.%s()' %exchange)
    current_time =int( time.time()//60 * 60 * 1000)  # 毫秒
    sin_time = current_time - sin * 60 * 1000 * 60 * 24
    data = ex.fetch_ohlcv(symbol=pair,timeframe=freq,limit=lim,since=sin_time)
    print(np.array(data).shape[0])
    return data

# def fdir(data):
#     left = data[:2]
#     right = data[-2:]
#     l = reduce(lambda x,y:2*x+y,np.sign(left[0]-left[1]))
#     r = reduce(lambda x,y:2*x+y,np.sign(right[0]-right[1]))
#     return 2*l+r

def fractal(data,dist):
    dm = np.array(data)
    res = np.zeros(dm.shape[0])
    reg = res
    direct = 0
    i=0
    
    for x in dm:
        new = dm[i,2:4]
        if i == 0:
            reg[i] = new
            continue
        
        flag = reduce(lambda x,y:2*x+y,np.sign(reg[i-1]-new))
        if abs(flag)==3:
            
        i += 0
                          

d =get_ohlcv('binanceus','ETH/USDT','2h',300,2000)
tt = fractal(d,3)
dd = segment(tt)
# dd1 = segment(dd)


xh = np.nonzero(tt[:,-1]<0)
yh = list(map(lambda x:tt[x,3],xh))

xl = np.nonzero(tt[:,-1]>0)
yl = list(map(lambda x:tt[x,2],xl))

xs = np.nonzero(dd[:,-1]!=0)[0]
ys = list(map(lambda x:dd[x,2] if dd[x,-1]==1 else dd[x,3],xs))

# xs1 = np.nonzero(dd1[:,-1]!=0)[0]
# ys1 = list(map(lambda x:dd1[x,2] if dd1[x,-1]==1 else dd1[x,3],xs1))


plt.plot(np.arange(tt.shape[0]),tt[:,2],'r')
plt.plot(np.arange(tt.shape[0]),tt[:,3],'r')
plt.scatter(xh,yh,c='b')
plt.scatter(xl,yl,c='g')
plt.plot(xs,ys,c='purple')
# plt.plot(xs1,ys1,c='orange')




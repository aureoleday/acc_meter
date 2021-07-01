import numpy as np
import threading
import time
import operator
import matplotlib.pyplot as plt 
import pandas as pd
from enum import Enum
import time
from functools import reduce

#  引入ccxt框架， 通过pip install ccxt 可以进行安装
#ccxt 的github地址为： https://github.com/ccxt/ccxt
import ccxt

np.set_printoptions(suppress=False,precision=2,threshold=np.inf)

# def ccxt_available_get():
#     ex_list = ccxt.exchanges
#     av_list = []
#     for x in ex_list:
#         h = eval('ccxt.%s()' %x)
#         try:
#             h.fetchstatus()
#         except:
#             # print()
#             print('NA for %s' %h)
#         else:
#             av_list.append(x)
#             print('OK for %s' %h)
#     pd.DataFrame(av_list).to_csv('available_ex.csv')
#     return av_list

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

def cnt_ind(arr):
    cdict = dict()
    for x in arr:
        key = int(x) if x==x else 0
        if x in cdict:
            cdict[key] += 1
        else:
            cdict[key] = 1
    p = [(k,cdict[k]) for k in sorted(cdict.keys())]
    return np.array(p)
    # return cdict

def cont_static(arr):
    res = np.zeros(arr.size)
    pre = 0
    for i in np.arange(arr.size):        
        if arr[i]==0:
            res[i] = pre/2
            pre = 0
        else:
            pre += arr[i]
            # pre += 1
    return res    

def monot(arr):
    df_mono = pd.DataFrame(arr).rolling(2).apply(lambda x:np.sign(float(x.tail(1))-float(x.head(1))))
    # df_mono_ls = df_mono.rolling(2).apply(np.cumsum)
    r = df_mono.rolling(2).apply(sum).apply(cont_static)
    return r

def get_ohlcv_til(exchange,pair,freq,sin,lim): 
    # print(pair)
    cdata = np.array([])
    gap = 0
    ex = eval('ccxt.%s()' %exchange)
    current_time =int( time.time()//60*60* 1000)  # 毫秒    
    # print(pd.to_datetime(current_time, unit='ms'))
    sin_time = current_time - sin * 60 * 60 * 24 * 1000 - gap
    while 1:        
        # print(pd.to_datetime(sin_time, unit='ms'))
        # print('.',end="")
        data = np.array(ex.fetch_ohlcv(symbol=pair,timeframe=freq,limit=lim,since=sin_time))
        if data.shape[0] == 0:
            return cdata
        else:
            cdata = data if cdata.shape[0] == 0 else np.r_[cdata,data]
            gap = data.shape[0]*(cdata[-1,0]-cdata[-2,0])
            sin_time = int(sin_time + gap)
    return np.array(cdata)

def get_ohlcv(exchange,pair,freq,sin,lim):
    # limit = 1000  
    print(pair)
    ex = eval('ccxt.%s()' %exchange)
    current_time =int( time.time()//60 * 60 * 1000)  # 毫秒
    sin_time = current_time - sin * 60 * 1000 * 60 * 24
    data = ex.fetch_ohlcv(symbol=pair,timeframe=freq,limit=lim,since=sin_time)
    print(np.array(data).shape[0])
    return data

def get_markets(exchange):
    ex = eval('ccxt.%s()' %exchange)
    return ex.load_markets().keys()

# def 

# #  初始化bitme对象
# bitmex = ccxt.binanceus()

# # 请求的candles个数
# limit = 200  

# #  当前时间
# current_time =int( time.time()/data/60 * 60 * 1000)  # 毫秒
# print(current_time)

# # 获取请求开始的时间
# since_time = current_time - limit * 60 * 1000 * 60 * 24
 
# data = bitmex.fetch_ohlcv(symbol='BTC/USDT',timeframe='1d', limit=500, since=since_time)
# df = pd.DataFrame(data)
# df = df.rename(columns={0: 'open_time', 1: 'odirpen', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})

# # 时间转换成北京时间
# df['open_time'] = pd.to_datetime(df['open_time'], unit='ms') + pd.Timedelta(hours=8)

# # 设置index
# df = df.set_index('open_time', drop=True)

# 保存成csv文件
# df.to_csv('bitmex_data.csv')  # comma seperate Value
# print(df)


# mdf = monot(df)
# res = cnt_ind(mdf['close'])
# # print(mdf)
# print(res)

# d =get_ohlcv('binanceus','BTC/USDT','30m',400,2000)
# df = pd.DataFrame(d)
# df = df.rename(columns={0: 'open_time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
# df['open_time'] = pd.to_datetime(df['open_time'], unit='ms') + pd.Timedelta(hours=8)
# df = df.set_index('open_time', drop=True)
# print(df)
# res = cnt_ind(monot(d).iloc[:,4])
# print(res)


def fdir(data):
    res = 0
    left = data[:2]
    right = data[-2:]
    l = reduce(lambda x,y:2*x+y,np.sign(left[0]-left[1]))
    r = reduce(lambda x,y:2*x+y,np.sign(right[0]-right[1]))
    if l==-3 and r==3:
        res = 1
    elif l==3 and r==-3:
        res = -1
    return res          
    
                
def fractal(data,dist):
    dm = np.array(data)
    res = np.zeros(dm.shape[0])
    arg_min = 0
    arg_max = 0
    direct = 0
    cnt = 0
    i=0
    pre_x = np.zeros(2)
    for x in dm:
        d_min = x[3]
        d_max = x[2]
        pd_min = dm[arg_min,3]
        pd_max = dm[arg_max,2]
        # print(arg_min,arg_max,direct,cnt)
        if direct == 0:
            if d_min < pd_min:
                arg_min = i
            
            if d_max > pd_max:
                arg_max = i
                
            if (arg_max - arg_min) > dist:
                res[arg_min] = -1
                arg_min = arg_max
                direct = 1
                cnt = 0
            elif (arg_max - arg_min) < -dist:
                res[arg_max] = 1
                arg_max = arg_min
                direct = -1
                cnt = 0
        elif direct == 1:
            # if pd_min >= d_min:
            arg_min = i
            if(pd_max <= d_max):
                arg_max = i
                cnt = 0
            else:
                flag = reduce(lambda x,y:2*x+y,np.sign(pre_x[2:4]-x[2:4]))
                # print('prex:',pre_x)
                # print('x:',x)
                # print('d1 flag:%d,%d' %(flag,i))
                if abs(flag) == 1:
                    cnt += 1
                    # pre_x[2]

                if (arg_min - arg_max) > (dist+cnt):
                    res[arg_max] = 1
                    arg_max = arg_min
                    direct = -1
                    cnt = 0
        else:
            # if pd_max <= d_max:
            arg_max = i
            if(pd_min >= d_min):
                arg_min = i
                cnt = 0
            else:
                flag = reduce(lambda x,y:2*x+y,np.sign(pre_x[2:4]-x[2:4]))#embedded condition
                if flag == -1:
                    cnt += 1

                if (arg_max - arg_min) > (dist+cnt):
                    res[arg_min] = -1
                    arg_min = arg_max
                    direct = 1
                    cnt = 0
        pre_x = x
        i += 1
    return np.c_[dm,res]

def get_bins(data,col):
    nz = np.nonzero(data[:,col]!=0)[0]
    ret = list(map(lambda x:data[x,2] if data[x,col]==1 else data[x,3],nz))
    return np.array(ret)

def pivot_cnt(pdata,cdata):
    cnt = 0
    if np.array(pdata).shape[0]==0:
        ppd = [cdata[-1,1],cdata[-1,2]]
    else:
        ppd = [pdata[1],pdata[2]]
    ccd = [cdata[0,1],cdata[0,2]]
    # print('ppd:')
    # print(ppd)
    pdmm = [0,0]
    if (ccd[0]-ppd[1])>=0:
        for x in cdata:
            cdmm = [x[1],x[2]]            
            if pdmm[0]==0:
                pdmm = cdmm
                cnt += 1
            else:
                if cdmm[1] < pdmm[0]:                    
                    # print(pdmm,cdmm)
                    pdmm = cdmm
                    cnt += 1
            
            if cdmm[0]<ppd[1]:
                return cnt
            
    elif (ppd[0]-ccd[1])>=0:
        for x in cdata:
            cdmm = [x[1],x[2]]            
            if pdmm[0]==0:
                pdmm = cdmm
                cnt -= 1
            else:
                if cdmm[0] > pdmm[1]:                    
                    # print(pdmm,cdmm)
                    pdmm = cdmm
                    cnt -= 1
            if cdmm[1]>ppd[0]:
                return cnt
    return cnt

def overlap(data):
    zd = zip(data[:-1],data[1:])
    pv = np.zeros(2)
    cnt = 0
    for x in list(zd):
        if cnt==0:
            pv = np.array([min(x),max(x)])
        else:            
            if (max(pv)-min(x))*(min(pv)-max(x))<0:                
                pv[0] = max(pv[0],min(x))
                pv[1] = min(pv[1],max(x))
            else:
                return np.zeros(2)        
        cnt += 1
    return pv

def pivot(data,col):
    bins = get_bins(data,col)
    pv = np.zeros(4)
    ret = np.array([])
    for i in np.arange(bins.shape[0]):
        x = np.array(bins)[::-1][i:i+5]
        if x.shape[0] < 5:
            return ret.reshape(-1,4)
        ov = overlap(x)
        if ov[1] != 0:
            pv[1:3] = ov
            if list(ret[-3:-1]) != list(pv[1:3]):
                ret = np.append(ret,pv)
    return ret.reshape(-1,4)

def pvr(raw):
    pv = np.zeros(4)
    ret = np.array([])
    for i in np.arange(np.array(raw).shape[0]):
        x = np.array(raw)[::-1][i:i+5]
        if x.shape[0] < 5:
            return ret.reshape(-1,4)
        ov = overlap(x)
        if ov[1] != 0:
            pv[1:3] = ov
            if list(ret[-3:-1]) != list(pv[1:3]):
                ret = np.append(ret,pv)
    return ret.reshape(-1,4)

#3,down
#-3,up
#-1,expand
#1,shrink
def gdir(arr):
    
    flag = reduce(lambda x,y:2*np.sign(x)+np.sign(y),[arr[-4]-arr[-2],arr[-3]-arr[-1]])
    if (arr[-4]-arr[-3]) < 0 and abs(flag) < 3:
        flag = -flag
    return flag

def segment(data):
    res = np.zeros(np.array(data).shape[0])
    dm = np.array([])
    i = 0
    direct = 0
    min_max = [-1,-1]
    flag = 0
    for x in data:
        if x[-1] != 0:           
            # print("%d,%d,%d,%d,%d" %(i,direct,flag,min_max[0],min_max[1]))
            dm = np.r_[dm,i]
            if dm.size>3:
                seq_id = dm[-4:]
                seq = list(map(lambda t: data[int(t)][2] if data[int(t)][-1]>0 else data[int(t)][3],seq_id))
                if direct == 0:
                    if abs(flag) == 1:
                        if seq[-1]>data[min_max[1]][2] and gdir(seq)==-3:#up break
                            direct = 1
                            res[min_max[0]] = -1
                            min_max[1] = int(seq_id[-1])
                            min_max[0] = int(seq_id[-1])
                            flag = -3
                            # print("up break from d0")
                        elif seq[-1]<data[min_max[0]][3] and gdir(seq)==3:#down break
                            direct = -1
                            res[min_max[1]] = 1
                            min_max[0] = int(seq_id[-1])
                            min_max[1] = int(seq_id[-1])
                            flag = 3
                            # print("down break from d0")
                        else:
                            if seq[-1]<data[min_max[0]][3]:
                                min_max[0] = int(seq_id[-1])
                            elif seq[-1]>data[min_max[1]][2]:
                                min_max[1] = int(seq_id[-1])
                    else:
                        flag = gdir(seq)
                        min_max = [int(seq_id[np.argmin(seq)]),int(seq_id[np.argmax(seq)])]
                        if flag == 3:
                            direct = -1
                            res[min_max[1]] = 1
                            # print("down extend from d0,%d" %i)
                        elif flag == -3:
                            direct = 1
                            res[min_max[0]] = -1
                            # print("up extend from d0")
                        else:
                            direct = 0
                elif direct == 1:
                    if abs(flag) == 1:  
                        if flag == 1:
                            min_max[0] = int(seq_id[np.argmin(seq[-2:])])
                        # if seq[-1]>data[min_max[1]][2] and gdir(seq)==-3:#up break
                        if seq[-1]>data[min_max[1]][2] and seq[-2]>data[min_max[0]][3]:#up break
                            flag = -3
                            min_max[1] = int(seq_id[np.argmax(seq)])
                        # elif seq[-1]<data[min_max[0]][3] and gdir(seq)==3:#down break
                        elif seq[-1]<data[min_max[0]][3] and seq[-2]<data[min_max[1]][2]:#down break 
                            direct = -1
                            res[min_max[1]] = 1
                            flag = 3
                            min_max[0] = int(seq_id[-1])
                            min_max[1] = int(seq_id[-1])
                            # print('down break from flag1,%d' %i)
                        else:
                            if min(seq)<data[min_max[0]][3]:
                                min_max[0] = int(seq_id[np.argmin(seq)])
                            elif max(seq)>data[min_max[1]][2]:
                                min_max[1] = int(seq_id[np.argmax(seq)]) 
                    else:
                        flag = gdir(seq)
                        if flag == 3:
                            direct = -1
                            res[min_max[1]] = 1
                            min_max[0] = int(seq_id[-1])
                            min_max[1] = int(seq_id[-1])
                            # print('down break from flag3,%d' %i)
                        elif flag == -3:
                            direct = 1
                            min_max[1] = int(seq_id[np.argmax(seq)])
                        else:
                            min_max[0] = int(seq_id[np.argmin(seq*(seq_id<=min_max[0]))])
                            min_max[1] = int(seq_id[-1]) if seq[-1]>data[min_max[1]][2] else min_max[1]
                elif direct == -1:                 
                    if abs(flag) == 1:
                        if flag == 1:
                                min_max[1] = int(seq_id[np.argmax(seq[-2:])])
                        if seq[-1]>data[min_max[1]][2] and seq[-2]>data[min_max[0]][3]:#up break
                            direct = 1
                            res[min_max[0]] = -1
                            min_max[0] = int(seq_id[-1])
                            min_max[1] = int(seq_id[-1])
                            # print('up break from flag1,%d' %i)
                            flag = -3
                        elif seq[-1]<data[min_max[0]][3] and seq[-2]<data[min_max[1]][2]:#down break
                            min_max[0] = int(seq_id[np.argmin(seq)])
                            flag = 3
                        else:
                            if min(seq)<data[min_max[0]][3]:
                                min_max[0] = int(seq_id[np.argmin(seq)])
                            elif max(seq)>data[min_max[1]][2]:
                                min_max[1] = int(seq_id[np.argmax(seq)]) 
                    else:
                        flag = gdir(seq)                        
                        if flag == 3:
                            direct = -1
                            min_max[0] = int(seq_id[np.argmin(seq)])                         
                            # res[min_max[1]] = 1
                        elif flag == -3:
                            direct = 1
                            res[min_max[0]] = -1
                            min_max[0] = int(seq_id[-1])
                            min_max[1] = int(seq_id[-1])
                            # print('up break from flag-3,%d' %i)
                        else:
                            min_max[1] = int(seq_id[np.argmax(seq*(seq_id>=min_max[1]))])
                            min_max[0] = int(seq_id[-1]) if seq[-1]<data[min_max[0]][3] else min_max[0]
                            # print(min_max)
                    
            else:
                0               
        else:
            0
        i += 1
    return np.c_[data,res]

def plotsegs(data,pair,show):
    fracto = fractal(data,3)
    seg = segment(fracto)
    seg1 = segment(seg)
    
    if show == 1:
        xh = np.nonzero(fracto[:,-1]<0)
        yh = list(map(lambda x:fracto[x,3],xh))
        
        xl = np.nonzero(fracto[:,-1]>0)
        yl = list(map(lambda x:fracto[x,2],xl))
        
        xs = np.nonzero(seg[:,-1]!=0)[0]
        ys = list(map(lambda x:seg[x,2] if seg[x,-1]==1 else seg[x,3],xs))
        
        xs1 = np.nonzero(seg1[:,-1]!=0)[0]
        ys1 = list(map(lambda x:seg1[x,2] if seg1[x,-1]==1 else seg1[x,3],xs1))
        
        plt.figure(pair)
    
        plt.vlines(x=np.arange(fracto.shape[0]),ymin=fracto[:,3],ymax=fracto[:,2],color='r')
        plt.scatter(xh,yh,c='b',marker='.')
        plt.scatter(xl,yl,c='g',marker='.')
        plt.plot(xs,ys,c='purple')
        plt.plot(xs1,ys1,c='green')
    return seg1

def cpos_calc(pvt,price):
    if np.array(pvt).shape[0] != 4:
        return None
    res = 0
    if price > pvt[2]:
        res = 1
    elif price < pvt[1]:
        res = -1
    else:
        res = 0
    return res

def pvcnt(seg):
    cpos = np.zeros(3)   
    cnt = np.zeros(3)
    d1 = pivot(seg,-1)
    # print(d1)
    d2 = pivot(seg,-2)
    d3 = pivot(seg,-3)
    
    # print('level-0')
    if np.array(d2).shape[0]>0:
        cnt[2] = pivot_cnt(d2[0],d3)
    else:
        print('d2 null')
        # return
    # print('level-1')
    if np.array(d1).shape[0]>0:
        cnt[1] = pivot_cnt(d1[0],d2)
    else:
        print('d1 null')
        # return
    # print('level-2')
    if np.array(d1).shape[0]>0:
        cnt[0] = pivot_cnt(d1[-1], d1)
    else:
        print('d1 null')
        # return
    
    price = seg[-1,4]
    # print(d1[0],price)
    cpos[0] = cpos_calc(d1[0],price)
    cpos[1] = cpos_calc(d2[0],price)
    cpos[2] = cpos_calc(d3[0],price)
    return np.array([cnt,cpos])

def mult_t(mk):
    try:
        d =get_ohlcv_til(ex,mk,'1h',400,720)
        seg1 = plotsegs(d,mk,0)
        cnt = pvcnt(seg1)
        print(mk)
        print(cnt)
    except:
        print('skip')
    

ex = 'binanceus'
# mk = 'BAT/USDT'
mks = get_markets(ex)
for item in filter(lambda x: 'BNB' in x,mks):
    t = threading.Thread(target=mult_t,args=[item])
    t.start()
    time.sleep(1)


# d =get_ohlcv_til(ex,mk,'1h',400,720)
# print(mk)
# seg1 = plotsegs(d,mk,1)
# cnt = pvcnt(seg1)
# print(cnt)


# mks = get_markets(ex)
# for mk in mks:
#     try:
#         d =get_ohlcv_til(ex,mk,'1h',400,720)
#         seg1 = plotsegs(d,mk,0)
#         print("\n")
#         try:
#             print(pvcnt(seg1))
#         except:
#             print('null')
#     except:
#         print('skip')
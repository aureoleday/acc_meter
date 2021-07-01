#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 23:37:37 2021

@author: aureoleday
"""

import ccxt
import pandas as pd

def get_es(ex_id,timeout):
    ex_class = getattr(ccxt,ex_id)
    ex = ex_class({
        'timeout':timeout
        })
    return ex.fetchStatus()

def get_data(ex_id,timeout):
    ex_class = getattr(ccxt,ex_id)
    ex = ex_class({
        'timeout':timeout
        })
    return ex.fetch_ticker('BTC/USDT')


def get_av(timeout):
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
    
h = get_av(3000)
    


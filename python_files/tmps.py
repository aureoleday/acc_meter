# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:08:52 2019

@author: Administrator
"""

def calc_temp(D2,C5,C6):
    dT = D2 - (C5<<8)
    ret = 2000 + ((dT*C6)>>23)    
    return dT,ret

def calc_off(dT,C2,C4):

    OFF = ((C2<<16) + (C4*dT>>7))
    
    return(OFF)

def calc_all(D1,D2,C1,C2,C3,C4,C5,C6):
    dT = D2 - (C5<<8)
    TEMP = 2000 + ((dT*C6)>>23)
    OFF = ((C2<<16) + (C4*dT>>7))
    SENSE = (C1<<15) + ((C3*dT)>>8)
    P = ((D1*SENSE)/2**21 - OFF)/(2**13)
    return TEMP,OFF,SENSE,P
#    OFF = ((C2<<16) + (C4*dT>>7))
#    SENSE = (C1<<15 + (C3*dT)>>8
#    print("dT:%d,TEMP:%d,OFF:%d,SENSE:%d\n",dT,TEMP,OFF,SENSE);

def calc_pres(D1,SEN,OFF):
    ret = (((D1*SEN)>>21) - OFF)>>13    
    return ret

calc_temp(7119033,26891,26383)


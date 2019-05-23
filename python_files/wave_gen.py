# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:17:53 2019

@author: Administrator
"""

import wave
import numpy as np
import struct

def wave_gen(sample_rate=8000,freq=500,duration=100,volume = 1000):
    x = np.linspace(0, duration, num=duration*sample_rate)
    y = np.sin(2 * np.pi * freq * x) * volume
    
    wf = wave.open("sine.wav", 'wb')
    wf.setnchannels(1)
    wf.setframerate(sample_rate)
    wf.setsampwidth(2)
    for i in y:
        data = struct.pack('<h', int(i))
        wf.writeframesraw(data)
    wf.close() 
    
if __name__ == '__main__':
    wave_gen(freq=700)



import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 

fs = 2e6            //sample frequency
rb = 1e3            //bit rate 
fc = 0.5e6          //carrier frequency
t_sim = 1e-2        //simlation time

src_bits = 1-2*np.random.randint(0,2,t_sim*rb)


class mod_iq:
    def __init__(self,fc,fs,mod_type=2,phase=0,alpha=0.5,rep=1):
        self.fc = fc 
        self.fs = fs  
        self.pahse = phase 
        self.mod_type = mod_type 
    
    def mod_gen(self,di,dq):
        i = 0
        while True:
            d_iq = np.exp(1j*(2*np.pi*(self.fc/self.fs*i+self.phase)))
            yield d_iq
            i += 1

	


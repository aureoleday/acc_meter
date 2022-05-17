#RDC algorithym simulation

import numpy as np
import matplotlib.pyplot as plt
from fxpmath import Fxp
from functools import reduce

def AGWN(Ps,snr):
    SNR = 10**(snr/10)
    # print(SNR)
    # Ps = reduce(lambda x,y:x+y,map(lambda x:x**2,sin))/sin.size
    # print(Ps)
    Pn = Ps/SNR
    # print(Pn)
    agwn = np.random.randn(1)[0]*(Pn**0.5)
    return agwn

def my_fft(din,fft_size):
    temp = din[:fft_size]
    fftx = np.fft.rfft(temp)/fft_size
    xfp = np.abs(fftx)*2
    return xfp

class LoopFilter(object):
    def __init__(self, gain, Bn, zeta,dt_in='S1.15',dt_out='S1.15'):
        self.kp = Fxp((1/gain)*(4*zeta/(zeta+1/(4*zeta)))*Bn,dtype='U0.16')
        self.ki = Fxp((1/gain)*(4/(zeta+1/(4*zeta))**2)*(Bn**2),dtype='U0.16')
        self.dt_in = dt_in
        self.dt_out = dt_out
        self.integrater = Fxp(0,dtype='S1.15')
        self.lf_out = Fxp(0,dtype='S1.15')
        print("kp:%f, ki:%f" %(self.kp,self.ki))
        
    def advance_filter(self, phase_difference):
        self.integrater += (self.ki*Fxp(phase_difference,dtype=self.dt_in)).resize(dtype=self.dt_out)
        self.lf_out= (self.integrater+self.kp*Fxp(phase_difference,dtype=self.dt_out)).resize(dtype=self.dt_out)

        return self.lf_out

class PhaseDetector(object):
    def __init__(self,mode = 'pll',dt_in='S1.15',dt_out='S1.15'):
        self.phase_difference = Fxp(0)
        self.mode = mode
        self.dt_in = dt_in
        self.dt_out = dt_out
        self.temp = Fxp(0,dtype=dt_in)*Fxp(0,dtype=dt_in)

    def pd(self, d_iq, vco):
        if(self.mode == 'costas'):
            self.temp = Fxp(vco,dtype=self.dt_in)*Fxp(d_iq,dtype=self.dt_in)
            self.phase_difference = Fxp(self.temp.real,dtype=self.dt_in) * Fxp(self.temp.imag,dtype=self.dt_in).resize(dtype=self.dt_in)
        elif(self.mode == 'pll'):
            self.phase_difference = (np.conjugate(Fxp(d_iq,dtype=self.dt_in))*Fxp(vco,dtype=self.dt_in)).imag.resize(dtype=self.dt_out)
        else:
            #self.phase_difference = (np.conjugate(d_iq)*vco).imag
            self.phase_difference = (np.conjugate(Fxp(d_iq,dtype=self.dt_in))*Fxp(vco,dtype=self.dt_in)).imag.resize(dtype=self.dt_out)
        
        # print("pd:",self.phase_difference.info())
        return self.phase_difference.resize(dtype=self.dt_in)
		
class PLL(object):
    def __init__(self,fs,fc, lf_gain, lf_bandwidth, lf_damping,lf_delay = 1,pd_type='pll'):
        self.n = 0
        self.delay = lf_delay 
        self.fs = fs
        self.fc = fc
        self.phase_estimate = 0.0
        self.vco = np.exp(0j)
        self.phase_difference = 0.0
        self.loop_filter = LoopFilter(lf_gain, lf_bandwidth, lf_damping)
        self.phase_detector = PhaseDetector(pd_type)
        self.omega = 2*np.pi*self.fc/self.fs 
        self.loop_reg = np.zeros(32) 
        print("%s omega:%f" %(pd_type,self.omega))
        
    def update_phase_estimate(self):
        self.n += 1
        self.phase_estimate += self.loop_reg[self.delay]
        self.vco = np.exp(1j*(self.omega*(self.n)+self.phase_estimate))
        self.loop_reg[1:] = self.loop_reg[:-1]
        self.loop_reg[0] = self.loop_filter.lf_out

    def step(self, d_iq):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        self.phase_difference = self.phase_detector.pd(self.vco,d_iq)
        self.loop_filter.advance_filter(self.phase_difference)
        self.update_phase_estimate()
        # print("ced:%f,cef:%f,cpe:%f" %(self.phase_difference,self.loop_filter.ef(),self.phase_estimate))

class FPLL(object):
    def __init__(self,fs,fc, lf_gain, lf_bandwidth, lf_damping,lf_delay = 1,pd_type='pll',dt_in='S1.15',dt_out='S1.15'):
        self.n = 0
        self.delay = lf_delay 
        self.fs = fs
        self.fc = fc
        self.phase_estimate = Fxp(0.0,dtype=dt_in)
        self.phase_difference = Fxp(0.0,dtype=dt_in)
        self.vco = Fxp(np.exp(0j),dtype=dt_in)
        self.loop_filter = LoopFilter(lf_gain, lf_bandwidth, lf_damping)
        self.phase_detector = PhaseDetector(pd_type)
        self.omega = Fxp(2*np.pi*self.fc/self.fs,dtype=dt_in)
        self.dt_in = dt_in
        self.dt_out = dt_out
        self.loop_reg = np.zeros(32) 
        print("%s omega:%f" %(pd_type,2*np.pi*fc/fs))

    def update_phase_estimate(self):
        self.n += 1
        self.phase_estimate += Fxp(self.loop_reg[self.delay],dtype=self.dt_in)
        self.vco = Fxp(np.exp(1j*(self.omega*(self.n)+self.phase_estimate)),dtype=self.dt_in)
        self.loop_reg[1:] = self.loop_reg[:-1]
        self.loop_reg[0] = self.loop_filter.lf_out
        
    def step(self, d_iq):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        self.phase_difference = self.phase_detector.pd(self.vco,d_iq)
        self.loop_filter.advance_filter(self.phase_difference)
        self.update_phase_estimate()
        # print("ced:%f,cef:%f,cpe:%f" %(self.phase_difference,self.loop_filter.ef(),self.phase_estimate))
        
class COMB(object):
    def __init__(self,fs,fc,fm, lf_gain, lf_bandwidth, lf_damping, lf_delay=1):
        self.costas = PLL(fs,fc,0.5, 0.02, 0.707,lf_delay,pd_type='costas')
        self.delay = lf_delay
        self.demod = 0
        self.phase_detector = PhaseDetector('pll')
        self.n = 0
        self.fs = fs
        self.fm = fm
        self.phase_estimate = 0.0
        self.rad = 0.0
        self.vco = np.exp(0j)
        self.phase_difference = 0.0
        self.loop_filter = LoopFilter(lf_gain, lf_bandwidth, lf_damping)
        self.loop_reg = np.zeros(32)
        self.N = 16384
        self.nco = np.exp(1j*2*np.pi*np.arange(self.N)/self.N)

    def update_phase_estimate(self):
       self.n += 1
       self.phase_estimate += self.loop_reg[self.delay]
       self.rad = self.phase_estimate
       self.vco = np.exp(1j*self.rad)
       self.loop_reg[1:] = self.loop_reg[:-1]
       self.loop_reg[0] = self.loop_filter.lf_out

    def step(self, d_iq):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        self.phase_difference = self.phase_detector.pd(d_iq,self.vco)
        self.costas.step(d_iq.real)
        self.demod = self.costas.vco.real*self.phase_difference
        self.loop_filter.advance_filter(self.demod)
        self.update_phase_estimate()

#motor angle frequency
fm = 3000
#RDC sample frequency
fs = 200000
#RDC stimulus carrier frequency
fc = 20000
#simlulation time stamp
N = 5000
#equivalent quantatization noise
SNR = 73.76
Tf = 0.001
#loop gain
Kd = 1
#equivalent noise bandwidth
Bn = 0.02
#damping ratio
Zeta = 0.707
ts = np.arange(N-1)/fs

def costasf_tb():
    pll = FPLL(fs,fc,0.5, 0.02, 0.707,lf_delay = 1,pd_type='costas')
    phi = np.pi*(0.6)
    sig_fc = []
    out = []
    ed = []
    ef = []
    pe = []
    mod = []
    alpha = []
    beta = []
    gamma = []
    for i in range(0, N - 1):
        sig_fc.append(np.cos(2*np.pi*fc/fs*i + phi))
        in_sig = np.cos(2*np.pi*fc/fs*i + phi)*np.cos(2*np.pi*fm/fs*i)
        pll.step(in_sig)
        mod.append(in_sig)
        out.append(pll.vco.imag)
        ed.append(pll.phase_difference)
        ef.append(pll.loop_filter.lf_out)
        pe.append(pll.phase_estimate)

    plt.close('all')
    plt.figure()
    ax = plt.subplot(411)
    # ax.plot(ref,label='sig_in')
    ax.plot(ts,sig_fc,label='carrier')
    ax.plot(ts,out,label='out')
    plt.legend()
    
    ax = plt.subplot(412)
    ax.plot(ts,ed,label='ed')
    plt.legend()
    ax = plt.subplot(413)
    ax.plot(ts,ef,label='ef')
    plt.legend()
    ax = plt.subplot(414)
    ax.plot(ts,pe,label='pe')
    plt.legend()
    plt.show()

    # plt.figure()
    # ax = plt.subplot(111)
    # ax.plot(ts,list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(beta)-np.array(gamma))),label='phase_error')
    # # ax.plot(np.array(beta)-np.array(gamma),label='phase_error')
    # plt.grid(1)
    # plt.show()
    # err_a = list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(beta)-np.array(gamma)))
    # err_arr = np.array(err_a[int(N*0.67):])
    # std = np.sqrt(np.sum(np.square(err_arr-np.mean(err_arr)))/err_arr.size)
    # print("rms: ",std)
    # print("enob: ",np.log(1/std)/np.log(2)-1.76)
    
def costas_tb():
    pll = PLL(fs,fc,0.5, 0.02, 0.707,lf_delay = 1,pd_type='costas')
    phi = np.pi*(0.6)
    sig_fc = []
    out = []
    ed = []
    ef = []
    pe = []
    mod = []
    alpha = []
    beta = []
    gamma = []
    for i in range(0, N - 1):
        sig_fc.append(np.cos(2*np.pi*fc/fs*i + phi))
        in_sig = np.cos(2*np.pi*fc/fs*i + phi)*np.cos(2*np.pi*fm/fs*i)
        pll.step(in_sig)
        mod.append(in_sig)
        out.append(pll.vco.imag)
        ed.append(pll.phase_difference)
        ef.append(pll.loop_filter.lf_out)
        pe.append(pll.phase_estimate)

    plt.close('all')
    plt.figure()
    ax = plt.subplot(411)
    # ax.plot(ref,label='sig_in')
    ax.plot(ts,sig_fc,label='carrier')
    ax.plot(ts,out,label='out')
    plt.legend()
    
    ax = plt.subplot(412)
    ax.plot(ts,ed,label='ed')
    plt.legend()
    ax = plt.subplot(413)
    ax.plot(ts,ef,label='ef')
    plt.legend()
    ax = plt.subplot(414)
    ax.plot(ts,pe,label='pe')
    plt.legend()
    plt.show()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(ts,list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(beta)-np.array(gamma))),label='phase_error')
    # ax.plot(np.array(beta)-np.array(gamma),label='phase_error')
    plt.grid(1)
    plt.show()
    err_a = list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(beta)-np.array(gamma)))
    err_arr = np.array(err_a[int(N*0.67):])
    std = np.sqrt(np.sum(np.square(err_arr-np.mean(err_arr)))/err_arr.size)
    print("rms: ",std)
    print("enob: ",np.log(1/std)/np.log(2)-1.76)
   
def pll_tb():
    pll = PLL(fs,fm,0.5, 0.05, 0.707)
    phi = np.pi/1.1

    ref = []
    diff = []
    demod = []
    for i in range(0, N - 1):
        d_iq = np.exp(1j*(2*np.pi*fm/fs*i+phi))
        pll.step(d_iq)
        ref.append(d_iq.imag)
        demod.append(pll.vco.imag)
        diff.append(pll.loop_filter.lf_out)

    plt.figure()
    ax = plt.subplot(311)
    ax.plot(ref,label='sig_in')
    ax.plot(demod,label='demod')
    plt.legend()
    ax = plt.subplot(312)
    ax.plot(diff,label='diff')
    plt.legend()
    ax = plt.subplot(313)
    ax.plot(demod,label='demod')
    plt.legend()   
    plt.show()   

def comb_tb():
    pll = COMB(fs,fc,fm,Kd, Bn, Zeta,lf_delay=1)
    # pll = COMB(fs,fc,fm,0.2, 0.01, 0.707,lf_delay=1)
    #phase delay in range
    # phi = 0.1*np.pi
    #phase delay out of range
    phi = 0.4*np.pi
    theta = 0.0*np.pi

    raw = []    
    iq_dr = []
    iq_d = []
    mix = []
    pll_vco = []
    costas_vco = []
    
    ed = []
    ed_c = []
    ef = []
    ef_c = []
    pe = []
    pe_c = []
    alpha = []
    beta = []
    gamma = []
    
    for i in range(0, N - 1):
        # clockwise rotation
        d_iq = np.exp(1j*(2*np.pi*fm/fs*i+theta))
        # anti-clockwise rotation
        #d_iq = complex(d_iq.imag,d_iq.real)
        raw.append(d_iq)

        iq_dr.append(d_iq+complex(AGWN(0.5,SNR),AGWN(0.5,SNR)))
        alpha.append(np.arctan2(iq_dr[-1].imag,iq_dr[-1].real))
        gamma.append(np.arctan2(raw[-1].imag,raw[-1].real))
        beta.append((pll.rad+np.pi)%(2*np.pi)-np.pi)

        mix.append(np.sin(2*np.pi*fc/fs*i + phi)*iq_dr[-1])
        pll.step(mix[-1])
        iq_d.append(d_iq)
        pll_vco.append(pll.vco)
        costas_vco.append(pll.costas.vco)
        
        ed.append(pll.phase_difference)
        ed_c.append(pll.costas.vco.real)
        ef.append(pll.loop_filter.lf_out)
        ef_c.append(pll.demod)
        pe.append(pll.phase_estimate)
        pe_c.append(pll.costas.phase_estimate)
    
    plt.close('all')
    plt.figure()
    ax = plt.subplot(411)
    ax.plot(ts,list(map(lambda x:x.real,mix)),label='I')
    ax.plot(ts,list(map(lambda x:x.imag,mix)),label='Q')
    ax.set_title('input signal')
    plt.legend()

    ax = plt.subplot(412)
    ax.plot(ts,list(map(lambda x:x.imag,costas_vco)),label='costas_q')
    #ax.plot(list(map(lambda x:x.real,costas_vco)),label='costas_i')
    ax.plot(ts,list(map(lambda x:x.imag,iq_dr)),label='input q')
    ax.plot(ts,list(map(lambda x:x.imag,pll_vco)),label='recovered q')
    #ax.plot(list(map(lambda x:x.real,pll_vco)),label='pll_i')
    ax.set_title('VCO')
    plt.legend()
    

    ax = plt.subplot(413)
    # ax.plot(pe,label='pe')
    ax.plot(ts,pe_c,label='epe_c')
    ax.plot(ts,ef_c,label='ef_c')
    ax.plot(ts,ef,label='ef')
    ax.set_title('phase error')
    plt.legend()
    plt.grid(1)
    
    ax = plt.subplot(414)    
    ax.plot(ts,alpha,label='raw data')
    ax.plot(ts,beta,label='recovered')
    ax.set_title('phase compare')
    plt.legend()
    plt.show()
    print("ef:%f,pe_c:%f,pe:%f" %(ef[-1],pe_c[-1]/(np.pi),pe[-1]))
    print("ts:",-np.log(Tf*np.sqrt(1-Zeta**2))/(Zeta*Bn*fs*np.pi))

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(ts,list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(beta)-np.array(gamma))),label='phase_error')
    # ax.plot(np.array(beta)-np.array(gamma),label='phase_error')
    plt.grid(1)
    plt.show()
    err_a = list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(beta)-np.array(gamma)))
    err_arr = np.array(err_a[int(N*0.67):])
    std = np.sqrt(np.sum(np.square(err_arr-np.mean(err_arr)))/err_arr.size)
    print("rms: ",std)
    print("enob: ",np.log(1/std)/np.log(2)-1.76)
    # print("std: ",np.std(err_arr))

    # plt.figure()
    # plt.plot(pe_c,label='pe_c')
    # plt.plot(pe,label='pe')
    # plt.legend()
    
    # plt.figure()
    # hs_costas = my_fft(costas_vc,N-1)
    # hs_pll = my_fft(pll_vc,N-1)
    # hs_id = my_fft(i_d,N-1)
    # # print(costas_vc)
    
    # ax = plt.subplot(111)
    # ax.plot(hs_costas,label='carrier')
    # ax.plot(hs_pll,label='velocity')
    # ax.plot(hs_id,label='velocity modulation')
    # plt.legend()
    
    # ax = plt.subplot(212)
    # ax.plot(hs_id,label='velocity modulation')
    # # ax.plot(hs_id)
    # plt.legend()
    
    # ax = plt.subplot(212)
    # ax.plot(i_dr,label='i')
    # ax.plot(q_dr,label='q')
    # plt.legend()
    # return i_d

# costas_tb()
# pll_tb()
# comb_tb()
costasf_tb()

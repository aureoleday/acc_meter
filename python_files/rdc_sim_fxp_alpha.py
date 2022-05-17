#RDC algorithym simulation

import numpy as np
from scipy import signal
from fxpmath import Fxp
import matplotlib.pyplot as plt
from functools import reduce

RAW = Fxp(None,dtype='S1.15')
DATA = Fxp(None,dtype='S1.15')
ANGLE = Fxp(None,dtype='S4.20')
PANGLE = Fxp(None,dtype='S1.23',overflow='wrap')
CONST = Fxp(None,dtype='U0.18')
ADDER = Fxp(None,dtype='S4.24')
PADDER = Fxp(None,dtype='S1.23',overflow='wrap')
MULTER = Fxp(None,dtype='S1.15')
COMEGA = Fxp(None,dtype='U0.3',overflow='wrap')
CANGLE = Fxp(None,dtype='S4.6',overflow='wrap')
UANGLE = Fxp(None,dtype='S1.9',overflow='wrap')

INV_PI_CONST = Fxp(1/(2*np.pi)).like(CONST)

def AGWN(Ps,snr):
    SNR = 10**(snr/10)
    # print(SNR)
    # Ps = reduce(lambda x,y:x+y,map(lambda x:x**2,sin))/sin.size
    # print(Ps)
    Pn = Ps/SNR
    # print(Pn)
    np.random.seed(0)
    agwn = np.random.randn(1)[0]*(Pn**0.5)
    return agwn

def my_fft(din,fft_size):
    temp = din[:fft_size]
    fftx = np.fft.rfft(temp)/fft_size
    xfp = np.abs(fftx)*2
    return xfp

class my_filter:
    def __init__(self,N_ord,filt_zone=[0.1],filt_type='lowpass'):
        self.b,self.a = signal.butter(N_ord, filt_zone, filt_type)
        self.z = np.zeros(max(len(self.a),len(self.b))-1)
        # print("filter coefs a:",self.a)
        # print("filter coefs b:",self.b)
        
    def filt(self,din):
        dout, self.z = signal.lfilter(self.b, self.a, din, zi=self.z)
        return dout

class LoopFilter(object):
    def __init__(self, gain, Bn, zeta,opf='R'):
        self.kp = (1/gain)*(4*zeta/(zeta+1/(4*zeta)))*Bn
        self.ki = (1/gain)*(4/(zeta+1/(4*zeta))**2)*(Bn**2) 
        self.integrater = 0
        self.lf_out = 0
        self.opf = opf
        if opf=='Q':
            self.kp = Fxp(self.kp).like(CONST)
            self.ki = Fxp(self.ki).like(CONST)
            self.integrater = Fxp(self.integrater).like(ADDER)
            self.lf_out = Fxp(self.lf_out).like(ADDER)
        print("kp:%f, ki:%f" %(self.kp,self.ki))
        
    def advance_filter(self, phase_difference):
        if self.opf=='R':
            self.integrater += self.ki*phase_difference
            self.lf_out = self.integrater + self.kp*phase_difference
        else:
            self.integrater(self.integrater + self.ki*phase_difference)
            self.lf_out(self.integrater + self.kp*phase_difference)
        return self.lf_out

class PhaseDetector(object):
    def __init__(self,mode = 'pll',opf='R'):
        self.phase_difference = 0
        self.temp = 0
        if opf=='Q':
            self.phase_difference = Fxp(self.phase_difference).like(DATA)
            self.temp = Fxp(self.temp).like(MULTER)
        self.mode = mode;
        self.monitor = 0
        self.pdf0 = 0
        self.pdf1 = 0
        self.lpf0 = my_filter(3,[0.08])
        self.lpf1 = my_filter(3,[0.08])
        self.mon = 0
        self.opf = opf

    def pd(self, d_iq, vco):
        if(self.mode == 'costas'):
            if self.opf=='R':
                self.temp = vco*d_iq.real
                self.phase_difference = self.temp.real * self.temp.imag                
            else:
                self.temp(vco*Fxp(d_iq.real).like(RAW))
                self.phase_difference(self.temp.real * self.temp.imag)
                
            self.pdf0 = 2*self.lpf0.filt([vco.imag*d_iq.real])
            self.pdf1 = 2*self.lpf1.filt([vco.imag*d_iq.imag])
            self.mon = self.pdf0**2 + self.pdf1**2

        elif(self.mode == 'pll'):
            if self.opf=='R':
                self.phase_difference = (np.conjugate(d_iq)*vco).imag
            else:
                self.phase_difference((Fxp(np.conjugate(d_iq)).like(RAW)*vco).imag)
        else:
            if self.opf=='R':
                self.phase_difference = (np.conjugate(d_iq)*vco).imag
            else:
                self.phase_difference((Fxp(np.conjugate(d_iq)).like(RAW)*vco).imag)

        return self.phase_difference

class PLL(object):
    def __init__(self,fs,fc, lf_gain, lf_bandwidth, lf_damping,lf_delay = 1,pd_type='pll',opf='R'):
        self.delay = lf_delay 
        self.fs = fs
        self.fc = fc
        self.phase_estimate = 0.0        
        self.phase_difference = 0.0
        self.vco = np.exp(0j)
        self.omega_const = self.fc/self.fs 
        self.omega = self.fc/self.fs 
        self.loop_reg = np.zeros(32)
        if opf == 'Q':
            self.phase_difference = Fxp(self.phase_difference).like(DATA)
            self.phase_estimate = Fxp(self.phase_estimate).like(PANGLE)
            self.vco = Fxp(self.vco).like(DATA)
            self.omega_const = Fxp(self.omega).like(COMEGA)
            self.omega = Fxp(self.omega).like(COMEGA)
            self.ph_sum = Fxp(self.omega).like(UANGLE)
        self.loop_filter = LoopFilter(lf_gain, lf_bandwidth, lf_damping,opf=opf)
        self.phase_detector = PhaseDetector(pd_type,opf=opf)        
        self.opf = opf
        print("%s omega:%f" %(pd_type,self.omega))
        
    def update_phase_estimate(self):     
        if self.opf == 'R':
            self.phase_estimate += self.loop_reg[self.delay]
            self.omega += self.omega_const
            self.vco = np.exp(1j*(2*np.pi*self.omega + self.phase_estimate))
        else:
            self.phase_estimate(self.phase_estimate + self.loop_reg[self.delay]*INV_PI_CONST)
            self.omega(self.omega+self.omega_const)
            self.ph_sum(self.omega+self.phase_estimate)
            self.vco(np.exp(1j*(2*np.pi*(self.ph_sum.get_val()))))
        self.loop_reg[1:] = self.loop_reg[:-1]
        self.loop_reg[0] = self.loop_filter.lf_out

    def step(self, d_iq):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        if self.opf == 'R':
            self.phase_difference = self.phase_detector.pd(d_iq,self.vco)
        else:
            self.phase_difference(self.phase_detector.pd(d_iq,self.vco))
        self.loop_filter.advance_filter(self.phase_difference)
        self.update_phase_estimate()
        
        
class COMB(object):
    def __init__(self,fs,fc,fm, lf_gain, lf_bandwidth, lf_damping, lf_delay=1,opf='R'):
        self.costas = PLL(fs,fc,0.5, 0.02, 0.707,lf_delay,pd_type='costas',opf=opf)
        self.delay = lf_delay
        self.demod = 0
        self.monitor = 0
        self.phase_detector = PhaseDetector('pll',opf)
        self.fs = fs
        self.fm = fm
        self.phase_estimate = 0.0
        self.rad = 0.0
        self.vco = np.exp(0j)
        self.phase_difference = 0.0
        if opf == 'Q':
            self.phase_difference = Fxp(self.phase_difference).like(ANGLE)
            self.phase_estimate = Fxp(self.phase_estimate).like(PADDER)
            self.vco = Fxp(self.vco).like(DATA)
            self.demod = Fxp(self.demod).like(ANGLE)
        self.loop_filter = LoopFilter(lf_gain, lf_bandwidth, lf_damping,opf=opf)
        self.loop_reg = np.zeros(32)
        self.opf = opf

    def update_phase_estimate(self):
        if self.opf == 'R':
            self.phase_estimate += self.loop_reg[self.delay]
            self.rad = self.phase_estimate
        else:
            self.phase_estimate(self.phase_estimate + self.loop_reg[self.delay]*INV_PI_CONST)
            self.rad = 2*np.pi*self.phase_estimate.get_val()
       
        if self.opf == 'R':
            self.vco = np.exp(1j*self.phase_estimate)
        else:
            self.vco(np.exp(1j*2*np.pi*self.phase_estimate.get_val()))
            
        self.loop_reg[1:] = self.loop_reg[:-1]
        self.loop_reg[0] = self.loop_filter.lf_out

    def step(self, d_iq):
        if self.opf == 'R':
            self.phase_difference = self.phase_detector.pd(d_iq,self.vco)
        else:
            self.phase_difference(self.phase_detector.pd(d_iq,self.vco))
            
        self.costas.step(d_iq)
        
        if self.opf == 'R':
            self.demod = self.costas.vco.real*self.phase_difference
        else:
            self.demod(self.costas.vco.real*self.phase_difference)
            
        self.loop_filter.advance_filter(self.demod)
        self.update_phase_estimate()

#max motor angle freq
fmax = 20000
#motor angle frequency
fm = 2000
#acceleration
a = 000000
#frequency deviation
fd = 0.00
#dc offset 
dc = 0.0*(1-1j)
#RDC sample frequency
fs = 160000
#RDC stimulus carrier frequency
fc = 20000
#simlulation time stamp
N = 4000
#equivalent quantatization noise
# SNR = 120.76
SNR = 73.76
Tf = 0.001
#loop gain
Kd = 1
#equivalent noise bandwidth
Bn = 0.02
# damping ratio
Zeta = 0.707
ts = np.arange(N-1)/fs
usr_opf = 'R'

def costas_tb():
    pll = PLL(fs,fc,0.5, 0.02, 0.707,lf_delay = 1,pd_type='costas')
    phi = np.pi*(0)
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
    ax.set_ylabel('V')
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
    print("data mode:",usr_opf)
    print("freq deviation:",fd)
    print("accelleration:",a)
    print("simulation points:",N)
    print("AGWN:%fdB" %(SNR))
    print("======================================================")
    pll = COMB(fs,fc,fm,Kd, Bn, Zeta,lf_delay=1,opf=usr_opf)
    # pll = COMB(fs,fc,fm,0.2, 0.01, 0.707,lf_delay=1)
    #phase delay in range
    # phi = 0.1*np.pi
    #phase delay out of range
    phi = 0.0*np.pi
    theta = 1.3*np.pi

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
    mo0 = []
    mo1 = []
    mon = []
    pe_c = []
    alpha = []
    beta = []
    gamma = []
    at = 0
    at_reg = []
    for i in range(0, N - 1):
        if(fm+a/fs*i)<fmax:            
            k = i
            d_iq = np.exp(1j*(2*np.pi*(fm/fs*i+0.5*a/fs/fs*i*i)+theta))
        else:
            d_iq = np.exp(1j*(2*np.pi*(fmax/fs*(i-k)+fm/fs*k+0.5*a/fs/fs*k*k)+theta))
        
        # clockwise rotation
        # if(i<(N/2)):
        #     d_iq = np.exp(1j*(2*np.pi*(fm+0.5*0*i)/fs*i+theta))
        # else:
        #     d_iq = np.exp(1j*(2*np.pi*(fm+0.5*a*i)/fs*i+theta))
        # d_iq = np.exp(1j*(2*np.pi*(fm+0.5*a*i)/fs*i+theta))
        # anti-clockwise rotation
        #d_iq = complex(d_iq.imag,d_iq.real)
        raw.append(d_iq)

        iq_dr.append(d_iq+complex(AGWN(0.5,SNR),AGWN(0.5,SNR))+dc)
        alpha.append(np.arctan2(iq_dr[-1].imag,iq_dr[-1].real))
        gamma.append(np.arctan2(raw[-1].imag,raw[-1].real))
        beta.append((pll.rad+np.pi)%(2*np.pi)-np.pi)

        mix.append(np.sin(2*np.pi*fc/fs*(1+fd)*i + phi)*iq_dr[-1])
        pll.step(mix[-1])
        iq_d.append(d_iq)
        pll_vco.append(pll.vco.get_val() if usr_opf=='Q' else pll.vco)
        costas_vco.append(pll.costas.vco.get_val()  if usr_opf=='Q' else pll.costas.vco)
        # print(costas_vco)
        # print(pll.costas.vco)
        
        ed.append(pll.phase_difference)
        ed_c.append(pll.costas.vco.real)
        ef.append(pll.loop_filter.lf_out.get_val() if usr_opf=='Q' else pll.loop_filter.lf_out)
        ef_c.append(pll.demod.get_val() if usr_opf=='Q' else pll.demod)
        pe.append(pll.phase_estimate.get_val() if usr_opf=='Q' else pll.phase_estimate)
        pe_c.append(pll.costas.phase_estimate.get_val() if usr_opf=='Q' else pll.costas.phase_estimate)
        # mo.append(pll.phase_detector.monitor)
        mo0.append(pll.costas.phase_detector.pdf0)
        mo1.append(pll.costas.phase_detector.pdf1)
        mon.append(pll.costas.phase_detector.mon)
    print("end angle freq:%f,last freq:%f" %(at,at))
    
    plt.close('all')
    plt.figure()
    ax = plt.subplot(311)
    ax.plot(ts,list(map(lambda x:x.real,mix)),label='I')
    ax.plot(ts,list(map(lambda x:x.imag,mix)),label='Q')
    ax.set_ylabel('Amplitude')
    ax.set_title('input signal')
    plt.legend()

    # ax = plt.subplot(412)
    # ax.plot(ts,list(map(lambda x:x.imag,costas_vco)),label='costas_q')
    # # print("costas_q:",costas_vco[:10])
    # #ax.plot(list(map(lambda x:x.real,costas_vco)),label='costas_i')
    # ax.plot(ts,list(map(lambda x:x.imag,iq_dr)),label='input q')
    # ax.plot(ts,list(map(lambda x:x.imag,pll_vco)),label='recovered q')
    # #ax.plot(list(map(lambda x:x.real,pll_vco)),label='pll_i')
    # ax.set_title('VCO')
    # plt.legend()
    

    ax = plt.subplot(312)
    ax.plot(ts,(np.array(pe_c)+np.pi)%(2*np.pi)-np.pi,label='epe_c') 
    ax.plot(ts,ef_c,label='ef_c')
    ax.plot(ts,ef,label='ef')
    # ax.plot(ts,np.array(ef)*fs/(2*np.pi),label='ef')
    ax.set_ylabel('velocity(rps)')
    # ax.plot(ts,at_reg,label='at')
    # ax.plot(ts,pe,label='pe')
    ax.set_title('phase error')
    plt.legend()
    plt.grid(1)
    
    ax = plt.subplot(313)    
    ax.plot(ts,alpha,label='input data')
    ax.plot(ts,beta,label='recovered')
    ax.set_ylabel('angle(rad)')
    ax.set_title('phase compare')
    plt.legend()
    plt.show()
    print("ef:%f,pe_c:%f,pe:%f" %(ef[-1],pe_c[-1]/(np.pi),pe[-1]))
    print("ts:",-np.log(Tf*np.sqrt(1-Zeta**2))/(Zeta*Bn*fs*np.pi))
    print("angular freq:%f(rps)" %(ef[-1]*fs/(2*np.pi)))
    print("costas phase(scaled):",pe_c[-1]/(np.pi))

    plt.figure()
    ax = plt.subplot(211)
    ax.plot(ts,list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(beta)-np.array(gamma))),label='phase_error')
    # ax.plot(np.array(beta)-np.array(gamma),label='phase_error')
    # ax.plot(ts,np.unwrap(np.array(beta)-np.array(gamma)),label='phase_error')
    
    plt.grid(1)
    ax = plt.subplot(212)
    # m0 = list(map(lambda x:x[0],mo))SNR
    # m1 = list(map(lambda x:x[1],mo))
    ax.plot(ts,mo0,label='cos')
    ax.plot(ts,mo1,label='sin')
    ax.plot(ts,mon,label='mon')
    # ax.plot(ts,mo[1,:],label='monitor')
    plt.legend()
    plt.show()
    err_a = list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(beta)-np.array(gamma)))
    err_arr = np.array(err_a[int(N*0.5):])
    std = np.sqrt(np.sum(np.square(err_arr-np.mean(err_arr)))/err_arr.size)
    print("phase indicators:")
    print("mean: %f,rms: %f,range:%f" %(np.mean(err_arr),std,np.ptp(err_arr)))
    print("enob: ",np.log(1/std)/np.log(2)-1.76)
    
    err_arr = np.array(ef[int(N*0.5):])
    std = np.sqrt(np.sum(np.square(err_arr-np.mean(err_arr)))/err_arr.size)
    v_mean = np.mean(err_arr)*fs/(2*np.pi)
    print("velocity indicators:")
    print("mean: %f,rms: %f,range:%f" %(v_mean,std*fs/(2*np.pi),np.ptp(err_arr)*fs/(2*np.pi)))
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

comb_tb()

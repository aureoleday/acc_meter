#RDC algorithym simulation

import numpy as np
from scipy import signal
from fxpmath import Fxp
import matplotlib.pyplot as plt
# from functools import reduce

class cfg(object):
    class com(object):
        #RDC sample frequency
        fs = 160000
        #RDC stimulus carrier frequency
        fc = 20000
        #simlulation time stamp
        N = 4000
        #signal to noise ratio
        SNR = 61.96
        # SNR = 86
        ts = np.arange(N-1)/fs
        usr_opf = 'R'
    class src(object):
        #max motor angle freq
        fmax = 4000
        #motor angle frequency
        fm = 100
        #rotor direction@0:clockwise;1:anti-clockwise
        r_dir = 0
        #acceleration
        a = 00000
        #frequency deviation (~5%)
        fd_err = 0.00
        #dc offset 
        dc_err = 0.0*(1+1j)
        #gain error (~20%)
        ac_err = 0.00
        #differential phase shift 
        ps_err = 0.00*2*np.pi
        #differential phase shift compensation
        # dpsc = 1+ps_err*1.05
        # dpsc = 1-ps_err*1.2
        dpsc = 1-0.0
        # step response
        step_thresh = 2000
        step_value = 0/360*2*np.pi
        #carrier initial phase 
        phi = 0.0*np.pi
        #revolver initial phase 
        theta = 0*np.pi
    class pll(object):
        #loop gain
        Kd = 1
        #equivalent noise bandwidth(1%~5%)
        Bn = 0.02
        # damping ratio
        Zeta = 0.707
    class costas(object):
        #loop gain
        Kd = 0.5
        #equivalent noise bandwidth
        Bn = 0.01
        # damping ratio
        Zeta = 0.707
    class iir(object):
        iir_bypass = 0
        b1_iir = 0.998
        
# cfg = rdc_cfg()

RAW = Fxp(None,dtype='S1.15')
DATA = Fxp(None,dtype='S1.15')
ANGLE = Fxp(None,dtype='S2.16')
PANGLE = Fxp(None,dtype='S1.17',overflow='wrap')
CONST = Fxp(None,dtype='U0.18')
ADDER = Fxp(None,dtype='S1.24')
PADDER = Fxp(None,dtype='S1.23',overflow='wrap')
MULTER = Fxp(None,dtype='S1.15')
COMEGA = Fxp(None,dtype='U0.3',overflow='wrap')
CANGLE = Fxp(None,dtype='S4.6',overflow='wrap')
UANGLE = Fxp(None,dtype='S1.9',overflow='wrap')


INV_PI_CONST = Fxp(1/(2*np.pi)).like(CONST)

def AGWN(Ps,snr):
    SNR = 10**(snr/10)
    Pn = Ps/SNR
    # np.random.seed(9)
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
        
    def filt(self,din):
        dout, self.z = signal.lfilter(self.b, self.a, din, zi=self.z)
        return dout
    
class iir_filter(object):
    def __init__(self,k):
        self.k = k
        self.x = np.exp(0j)
        self.y = np.exp(0j)
    
    def filt(self,d_iq):
        y = d_iq - self.x+ self.k * self.y
        self.x = d_iq
        self.y = y 
        return y

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
    def __init__(self,mode = 'pll',opf='R',dpsc=1):
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
        self.dpsc=dpsc

    def phd(self, d_iq, vco):
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
                self.phase_difference = d_iq.real*vco.imag - self.dpsc*d_iq.imag*vco.real
                # self.phase_difference = (np.conjugate(d_iq)*vco).imag
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
        self.pd_type = pd_type
        self.opf = opf
        print("%s omega:%f" %(pd_type,self.omega))
        
    def update_phase_estimate(self):
        if self.pd_type == 'costas':
            if self.opf == 'R':
                self.vco = np.exp(1j*(2*np.pi*self.omega + self.phase_estimate))
                self.phase_estimate += self.loop_reg[self.delay]
                # self.omega += self.omega_const                
                self.omega += self.omega_const
            else:
                self.phase_estimate(self.phase_estimate + self.loop_reg[self.delay]*INV_PI_CONST)
                self.omega(self.omega+self.omega_const)
                self.ph_sum(self.omega+self.phase_estimate)
                self.vco(np.exp(1j*(2*np.pi*(self.ph_sum.get_val()))))            
        else:
            if self.opf == 'R':
                self.phase_estimate += self.loop_reg[self.delay]
                self.rad = self.phase_estimate
                self.vco = np.exp(1j*self.phase_estimate)
            else:
                self.phase_estimate(self.phase_estimate + self.loop_reg[self.delay]*INV_PI_CONST)
                self.rad = 2*np.pi*self.phase_estimate.get_val()
                self.vco(np.exp(1j*2*np.pi*self.phase_estimate.get_val()))
        self.loop_reg[1:] = self.loop_reg[:-1]
        self.loop_reg[0] = self.loop_filter.lf_out

    def step(self, d_iq):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        if self.opf == 'R':
            self.phase_difference = self.phase_detector.phd(d_iq,self.vco)
        else:
            self.phase_difference(self.phase_detector.phd(d_iq,self.vco))
        self.loop_filter.advance_filter(self.phase_difference)
        self.update_phase_estimate()
        
class COMB(object):
    def __init__(self,
                  fs=cfg.com.fs,
                  fm=cfg.src.fm,
                  fc=cfg.com.fc, 
                  lf_gain=cfg.pll.Kd, 
                  lf_bandwidth=cfg.pll.Bn, 
                  lf_damping = cfg.pll.Zeta, 
                  clf_gain=cfg.costas.Kd, 
                  clf_bandwidth=cfg.costas.Bn, 
                  clf_damping = cfg.costas.Zeta, 
                  lf_delay=1,
                  opf=cfg.com.usr_opf):
        self.costas = PLL(fs,fc,clf_gain, clf_bandwidth, clf_damping,lf_delay,pd_type='costas',opf=opf)
        self.delay = lf_delay
        self.demod = 0
        self.monitor = 0
        self.phase_detector = PhaseDetector('pll',opf,dpsc=cfg.src.dpsc)
        self.fs = fs
        self.fm = fm
        self.phase_estimate = 0.0
        self.rad = 0.0
        self.vco = np.exp(0j)
        self.phase_difference = 0.0
        if opf == 'Q':
            self.phase_difference = Fxp(self.phase_difference).like(DATA)
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
            self.vco = np.exp(1j*self.phase_estimate)
        else:
            self.phase_estimate(self.phase_estimate + self.loop_reg[self.delay]*INV_PI_CONST)
            self.rad = 2*np.pi*self.phase_estimate.get_val()
            self.vco(np.exp(1j*2*np.pi*self.phase_estimate.get_val()))
            
        self.loop_reg[1:] = self.loop_reg[:-1]
        self.loop_reg[0] = self.loop_filter.lf_out

    def step(self, d_iq):
        if self.opf == 'R':
            self.phase_difference = self.phase_detector.phd(d_iq,self.vco)
        else:
            self.phase_difference(self.phase_detector.phd(d_iq,self.vco))
            
        self.costas.step(d_iq)
        
        if self.opf == 'R':
            self.demod = self.costas.vco.real*self.phase_difference
        else:
            self.demod(self.costas.vco.real*self.phase_difference)
            
        self.loop_filter.advance_filter(self.demod)
        self.update_phase_estimate()
        

def sig_gen(
        fs=cfg.com.fs,
        fc=cfg.com.fc,
        fm=cfg.src.fm,
        fmax=cfg.src.fmax,
        phi=cfg.src.phi,
        theta=cfg.src.theta,
        a=cfg.src.a,
        step_thresh=cfg.src.step_thresh,
        step_value=cfg.src.step_value,
        r_dir=cfg.src.r_dir,

        ac_err=cfg.src.ac_err,
        dc_err=cfg.src.dc_err,
        fd_err=cfg.src.fd_err,
        ps_err=cfg.src.ps_err,        
        ):
    
    print("======source configuration============================")
    print("dc offset err :",dc_err)
    print("ac gain err :",ac_err)
    print("freq deviation:",fd_err)
    print("carrier phase diff:",ps_err)
    print("======================================================")
    
    i = 0
    while True:
        if(fm+a/fs*i)<fmax:            
            k = i            
            if i>step_thresh:
                raw = np.exp(1j*(2*np.pi*(fm/fs*i+0.5*a/fs/fs*i*i)+theta+step_value))
            else:
                raw = np.exp(1j*(2*np.pi*(fm/fs*i+0.5*a/fs/fs*i*i)+theta))
            # print(3)
        else:
            raw = np.exp(1j*(2*np.pi*(fmax/fs*(i-k)+fm/fs*k+0.5*a/fs/fs*k*k)+theta))

        # anti-clockwise rotation
        if r_dir == 1:
            raw = complex(raw.imag,raw.real)
            
        d_iq = complex(np.sin(2*np.pi*fc/fs*(1+fd_err)*i + phi)*raw.real*(1-ac_err),np.sin(2*np.pi*fc/fs*(1+fd_err)*i + phi+ps_err)*raw.imag) + dc_err
        # d_iq = complex(np.sin(2*np.pi*fc/fs*(1+fd_err)*i + phi)*raw.real*(1-ac_err),np.sin(2*np.pi*fc/fs*(1+fd_err)*i + phi)*raw.imag) + dc_err
       
        # dd_iq = complex(d_iq.real,(d_iq*np.exp(ps_err*1j)).imag)
        yield raw,d_iq
        i += 1

def add_kv(dic,k,v):
    if k not in dic.keys():
        dic[k] = []
    dic[k].append(v)
    

def comb_tb():
    # print("data mode:",cfg.com.usr_opf)
    # print("hpf bypass:",iir_bypass)
    # print("simulation points:",N)
    # print("AGWN:%fdB" %(SNR))
    # print("======================================================")
    my_iir = iir_filter(cfg.iir.b1_iir)
    pll = COMB()

    dic = {}

    mo0 = []
    mo1 = []
    mon = []
    
    rdc_in = sig_gen()
    for i in range(0, cfg.com.N - 1):
        t1,t2 = next(rdc_in)
            
        add_kv(dic,'raw',t1)
        add_kv(dic,'iq_dr',t2)
        add_kv(dic,'original_ph',np.arctan2(dic['raw'][-1].imag,dic['raw'][-1].real))
        add_kv(dic,'recovered_ph',(pll.rad+np.pi)%(2*np.pi)-np.pi)
        add_kv(dic,'mix',t2+complex(AGWN(0.5,cfg.com.SNR),AGWN(0.5,cfg.com.SNR)))        
        add_kv(dic,'mix_hpf',my_iir.filt(dic['mix'][-1]))
        
        pll.step(dic['mix'][-1] if cfg.iir.iir_bypass==1 else dic['mix_hpf'][-1])
        
        add_kv(dic,'pll_vco',pll.vco.get_val() if cfg.com.usr_opf=='Q' else pll.vco)        
        add_kv(dic,'costas_vco',pll.costas.vco.get_val()  if cfg.com.usr_opf=='Q' else pll.costas.vco)        
        add_kv(dic,'ed',pll.phase_difference.get_val()  if cfg.com.usr_opf=='Q' else pll.phase_difference)
        add_kv(dic,'ef',pll.loop_filter.lf_out.get_val() if cfg.com.usr_opf=='Q' else pll.loop_filter.lf_out)
        add_kv(dic,'ef_c',pll.demod.get_val() if cfg.com.usr_opf=='Q' else pll.demod)       
        add_kv(dic,'pe',pll.phase_estimate.get_val() if cfg.com.usr_opf=='Q' else pll.phase_estimate)        
        add_kv(dic,'pe_c',pll.costas.phase_estimate.get_val() if cfg.com.usr_opf=='Q' else pll.costas.phase_estimate)
               
        mo0.append(pll.costas.phase_detector.pdf0)
        mo1.append(pll.costas.phase_detector.pdf1)
        mon.append(pll.costas.phase_detector.mon)
    
    plt.close('all')
    plt.figure()
    ax = plt.subplot(311)
    ax.plot(cfg.com.ts,list(map(lambda x:x.real,dic['mix'])),label='I')
    ax.plot(cfg.com.ts,list(map(lambda x:x.imag,dic['mix'])),label='Q')
    ax.set_ylabel('Amplitude')
    ax.set_title('input signal')
    plt.legend()

    ax = plt.subplot(312)
    ax.plot(cfg.com.ts,(np.array(dic['pe_c'])+np.pi)%(2*np.pi)-np.pi,label='phase_c') 
    ax.plot(cfg.com.ts,dic['ef_c'],label='accelleration')
    ax.plot(cfg.com.ts,dic['ef'],label='velocity')
    ax.set_ylabel('velocity(rps)')
    ax.set_title('phase error')
    plt.legend()
    plt.grid(1)
    
    ax = plt.subplot(313)    
    ax.plot(cfg.com.ts,dic['original_ph'],label='input data')
    ax.plot(cfg.com.ts,dic['recovered_ph'],label='recovered')
    ax.plot(cfg.com.ts,list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(dic['recovered_ph'])-np.array(dic['original_ph']))),label='phase_error')

    ax.set_ylabel('angle(rad)')
    ax.set_title('phase compare')
    plt.legend()
    plt.show()
    print("ef:%f,pe_c:%f,pe:%f" %(dic['ef'][-1],dic['pe_c'][-1]/(np.pi),dic['pe'][-1]))
    print("angular freq:%f(rps)" %(dic['ef'][-1]*cfg.com.fs/(2*np.pi)))
    print("costas phase(scaled):",dic['pe_c'][-1]/(np.pi))

    plt.figure()
    ax = plt.subplot(211)
    ax.plot(cfg.com.ts,list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(dic['recovered_ph'])-np.array(dic['original_ph']))),label='phase_error')
    
    plt.grid(1)
    ax = plt.subplot(212)
    ax.plot(cfg.com.ts,mo0,label='cos')
    ax.plot(cfg.com.ts,mo1,label='sin')
    ax.plot(cfg.com.ts,mon,label='mon')
    plt.legend()
    plt.show()
    err_a = list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(dic['recovered_ph'])-np.array(dic['original_ph'])))
    err_arr = np.array(err_a[int(cfg.com.N*0.5):])
    std = np.sqrt(np.sum(np.square(err_arr))/err_arr.size)
    print("phase indicators:")
    print("mean: %f,rms: %f,range:%f" %(np.mean(err_arr),std,np.ptp(err_arr)))
    print("enob: ",np.log(2*np.pi/std)/np.log(2)-1.76)
    
    err_arr = np.array(dic['ef'][int(cfg.com.N*0.5):])
    std = np.sqrt(np.sum(np.square(err_arr-np.mean(err_arr)))/err_arr.size)
    v_mean = np.mean(err_arr)*cfg.com.fs/(2*np.pi)
    print("velocity indicators:")
    print("mean: %f,rms: %f,range:%f" %(v_mean,std*cfg.com.fs/(2*np.pi),np.ptp(err_arr)*cfg.com.fs/(2*np.pi)))
    print("enob: ",np.log(1/std)/np.log(2)-1.76)


comb_tb()

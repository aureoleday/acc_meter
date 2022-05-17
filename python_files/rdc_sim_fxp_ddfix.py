#RDC algorithym simulation
import json
from types import SimpleNamespace
import numpy as np
from scipy import signal
from fxpmath import Fxp
import matplotlib.pyplot as plt
import pandas as pd
# from functools import reduce

# class cfg(object):
#     class com(object):
#         #RDC sample frequency
#         fs = 160000
#         #RDC stimulus carrier frequency
#         fc = 20000
#         #simlulation time stamp
#         N = 6000
#         #signal to noise ratio
#         SNR = 61.96
#         # SNR = 1000
#         usr_opf = 'R'
#     class src(object):
#         #max motor angle freq
#         fmax = 4000
#         #motor angle frequency
#         fm = 50
#         #rotor direction@0:clockwise;1:anti-clockwise
#         r_dir = 0
#         #acceleration
#         a = 00000
#         #frequency deviation (~5%)
#         fd_err = 0.00
#         #dc offset 
#         dc_err = 0.00
#         #gain error (~20%)
#         ac_err = 0.000
#         #differential phase shift (uinified rad)
#         ps_err = 0.000
#         #shaft phase error (uinified rad)
#         sp_err = 0.00
#         #differential phase shift compensation
#         # dpsc = 1+ps_err*1.05
#         # dpsc = 1-ps_err*1.2
#         dpsc = 1
#         # step response
#         step_thresh = 2000
#         # step angle
#         step_value = 0
#         #carrier initial phase (uinified rad)
#         phi = 0.5
#         #revolver initial phase (uinified rad)
#         theta = 0.8
#     class pll(object):
#         #loop gain
#         Kd = 1
#         #equivalent noise bandwidth(1%~5%)
#         Bn = 0.02
#         # damping ratio
#         Zeta = 0.707
#     class costas(object):
#         #loop gain
#         Kd = 0.5
#         #equivalent noise bandwidth
#         Bn = 0.02
#         # damping ratio
#         Zeta = 0.707
#     class iir(object):
#         iir_bypass = 1
#         b1_iir = 0.998

cfg = json.load(open("rdc_cfg.json"), object_hook=lambda d: SimpleNamespace(**d))

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
    def __init__(self,k,opf='R'):
        self.k = k
        self.x = np.exp(0j)
        self.y = np.exp(0j)
        self.opf = opf
        self.out = np.exp(0j)
        if opf=='Q':
            self.k = Fxp(self.k).like(DATA)
            self.x = Fxp(self.x).like(DATA)
            self.y = Fxp(self.y).like(DATA)
            self.out = Fxp(self.out).like(DATA)
    
    def filt(self,d_iq):
        if self.opf=='R':
            self.out = d_iq - self.x+ self.k * self.y
            self.x = d_iq
            self.y = self.out
        else:
            self.out(Fxp(d_iq).like(DATA) - self.x+ self.k * self.y)
            self.x(d_iq)
            self.y(self.out.get_val())
        return self.y

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
        self.mode = mode
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

        elif(self.mode == 'pll'):
            if self.opf=='R':
                self.phase_difference = d_iq.real*vco.imag - self.dpsc*d_iq.imag*vco.real
            else:
                self.phase_difference((Fxp(np.conjugate(d_iq)).like(RAW)*vco).imag)
            
        else:
            if self.opf=='R':
                self.phase_difference = (np.conjugate(d_iq)*vco)
            else:
                self.phase_difference((Fxp(np.conjugate(d_iq)).like(RAW)*vco))

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
            self.phase_difference = Fxp(self.phase_difference).like(ANGLE)
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
                self.omega += self.omega_const
            else:
                self.vco(np.exp(1j*(2*np.pi*(self.ph_sum.get_val()))))  
                self.phase_estimate(self.phase_estimate + self.loop_reg[self.delay]*INV_PI_CONST)
                self.omega(self.omega+self.omega_const)
                self.ph_sum(self.omega+self.phase_estimate)                          
        else:
            if self.opf == 'R':
                self.vco = np.exp(1j*self.phase_estimate)
                self.phase_estimate += self.loop_reg[self.delay]
                self.rad = self.phase_estimate
                
            else:
                self.vco(np.exp(1j*2*np.pi*self.phase_estimate.get_val()))
                self.phase_estimate(self.phase_estimate + self.loop_reg[self.delay]*INV_PI_CONST)
                self.rad = 2*np.pi*self.phase_estimate.get_val()                
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
                  hpf_bypass=cfg.iir.iir_bypass,
                  b1_iir=cfg.iir.b1_iir,
                  opf=cfg.com.usr_opf):
        self.hpf_bypass = hpf_bypass
        self.hpf = iir_filter(b1_iir,opf)
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
        self.lpf = my_filter(3,[0.05])
        self.lpf_vco = my_filter(3,[0.05])
        self.lpf_dm = my_filter(3,[0.05])
        self.vco_filt = np.exp(0j)
        self.mon_iq = np.exp(0j)
        self.mon = 0
        self.demon = 0
        if opf == 'Q':
            self.phase_difference = Fxp(self.phase_difference).like(DATA)
            self.phase_estimate = Fxp(self.phase_estimate).like(PADDER)
            self.vco = Fxp(self.vco).like(DATA)
            self.demod = Fxp(self.demod).like(ANGLE)
            self.mon = Fxp(self.mon).like(DATA)
            self.mon_iq = Fxp(self.mon_iq).like(DATA)
            self.vco_filt = Fxp(self.vco_filt).like(DATA)
            self.demon = Fxp(self.demon).like(DATA)
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
        self.costas.step(d_iq) 
        if self.opf == 'R':
            if self.hpf_bypass==0:
                self.phase_difference = self.phase_detector.phd(self.hpf.filt(d_iq),self.vco)
            else:
                self.phase_difference = self.phase_detector.phd(d_iq,self.vco)
            # self.mon = d_iq*self.costas.vco.imag
        else:
            if self.hpf_bypass==0:
                self.phase_difference(self.phase_detector.phd(self.hpf.filt(d_iq),self.vco))
            else:
                self.phase_difference(self.phase_detector.phd(d_iq,self.vco))
            # self.mon(d_iq*self.costas.vco.imag)
        
        if self.opf == 'R':
            self.mon = d_iq*self.costas.vco.imag
        else:
            self.mon(d_iq*self.costas.vco.imag)
                        
        # self.costas.step(d_iq) 
                
        if self.opf == 'R':
            self.demod = self.costas.vco.real*self.phase_difference
            self.mon_iq = 2*self.lpf.filt([self.mon])
            self.vco_filt = self.lpf_vco.filt([self.vco])
            self.demon = (np.conjugate(self.mon_iq)*self.vco_filt).real
        else:
            self.demod(self.costas.vco.real*self.phase_difference)
            self.mon_iq(2*self.lpf.filt([self.mon]))
            self.vco_filt(self.lpf_vco.filt([self.vco]))
            self.demon((np.conjugate(self.mon_iq)*self.vco_filt).real)
            
        self.loop_filter.advance_filter(self.demod)
        self.update_phase_estimate()
        

def sig_gen(
        fs=cfg.com.fs,
        fc=cfg.com.fc,
        fm=cfg.src.fm,
        fmax=cfg.src.fmax,
        phi=cfg.src.phi*np.pi,
        theta=cfg.src.theta*np.pi,
        a=cfg.src.a,
        step_thresh=cfg.src.step_thresh,
        step_value=cfg.src.step_value/360*2*np.pi,
        r_dir=cfg.src.r_dir,

        ac_err=cfg.src.ac_err,
        dc_err=cfg.src.dc_err*(1+1j),
        fd_err=cfg.src.fd_err,
        ps_err=cfg.src.ps_err*2*np.pi,
        sp_err=cfg.src.sp_err*2*np.pi,
        ):
    
    print("======source configuration============================")
    print("dc offset err :",dc_err)
    print("ac gain err :",ac_err)
    print("freq deviation:",fd_err)
    print("carrier phase diff:",ps_err)
    print("shaft phase err:",sp_err)
    print("======================================================")
    
    i = 0
    while True:
        if(fm+a/fs*i)<fmax:            
            k = i            
            if i>step_thresh:
                raw = np.exp(1j*(2*np.pi*(fm/fs*i+0.5*a/fs/fs*i*i)+theta+step_value))
            else:
                raw = np.exp(1j*(2*np.pi*(fm/fs*i+0.5*a/fs/fs*i*i)+theta))
        else:
            raw = np.exp(1j*(2*np.pi*(fmax/fs*(i-k)+fm/fs*k+0.5*a/fs/fs*k*k)+theta))

        # anti-clockwise rotation
        if r_dir == 1:
            raw = complex(raw.imag,raw.real)
        
        raw = complex(raw.real,(raw*np.exp(1j*sp_err)).imag)
            
        d_iq = complex(np.sin(2*np.pi*fc/fs*(1+fd_err)*i + phi)*raw.real*(1-ac_err),np.sin(2*np.pi*fc/fs*(1+fd_err)*i + phi+ps_err)*raw.imag) + dc_err

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
    pll = COMB()
    dic = {}
    ts = np.arange(cfg.com.N-1)/cfg.com.fs
    
    rdc_in = sig_gen()
    for i in range(0, cfg.com.N - 1):
        t1,t2 = next(rdc_in)
            
        add_kv(dic,'raw',t1)
        add_kv(dic,'iq_dr',t2)
        add_kv(dic,'original_ph',np.arctan2(dic['raw'][-1].imag,dic['raw'][-1].real))
        add_kv(dic,'recovered_ph',(pll.rad+np.pi)%(2*np.pi)-np.pi)
        add_kv(dic,'mix',t2+complex(AGWN(0.5,cfg.com.SNR),AGWN(0.5,cfg.com.SNR))) 
        
        pll.step(dic['mix'][-1])
        
        add_kv(dic,'pll_vco',pll.vco.get_val() if cfg.com.usr_opf=='Q' else pll.vco)        
        add_kv(dic,'costas_vco',pll.costas.vco.get_val()  if cfg.com.usr_opf=='Q' else pll.costas.vco)        
        add_kv(dic,'ed',pll.phase_difference.get_val()  if cfg.com.usr_opf=='Q' else pll.phase_difference)
        add_kv(dic,'v',pll.loop_filter.lf_out.get_val() if cfg.com.usr_opf=='Q' else pll.loop_filter.lf_out)
        add_kv(dic,'acc',pll.demod.get_val() if cfg.com.usr_opf=='Q' else pll.demod)       
        add_kv(dic,'angle',pll.phase_estimate.get_val() if cfg.com.usr_opf=='Q' else pll.phase_estimate)        
        add_kv(dic,'angle_c',pll.costas.phase_estimate.get_val() if cfg.com.usr_opf=='Q' else pll.costas.phase_estimate)
        
        add_kv(dic,'mon',pll.mon.get_val() if cfg.com.usr_opf=='Q' else pll.mon)
        add_kv(dic,'mon_iq',pll.mon_iq.get_val() if cfg.com.usr_opf=='Q' else pll.mon_iq)
        add_kv(dic,'demon',pll.demon.get_val() if cfg.com.usr_opf=='Q' else pll.demon)

    df = pd.DataFrame(dic)
    df.to_csv('rdc_data.csv')
    
    plt.close('all')
    plt.figure()
    ax = plt.subplot(311)
    ax.plot(ts,list(map(lambda x:x.real,dic['mix'])),label='I')
    ax.plot(ts,list(map(lambda x:x.imag,dic['mix'])),label='Q')
    ax.set_ylabel('Amplitude')
    ax.set_title('input signal')
    plt.legend()

    ax = plt.subplot(312)
    ax.plot(ts,(np.array(dic['angle_c'])+np.pi)%(2*np.pi)-np.pi,label='phase_c') 
    ax.plot(ts,dic['acc'],label='accelleration')
    ax.plot(ts,dic['v'],label='velocity')
    ax.set_ylabel('phase(rad)')
    ax.set_title('state signals')
    plt.legend()
    plt.grid(1)
    
    ax = plt.subplot(313)    
    ax.plot(ts,dic['original_ph'],label='input data')
    ax.plot(ts,dic['recovered_ph'],label='recovered')
    ax.plot(ts,list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(dic['recovered_ph'])-np.array(dic['original_ph']))),label='phase_error')

    ax.set_ylabel('angle(rad)')
    ax.set_title('phase compare')
    plt.legend()
    plt.show()
    print("v:%f,angle_c:%f,angle:%f" %(dic['v'][-1],dic['angle_c'][-1]/(np.pi),dic['angle'][-1]))
    print("angular freq:%f(rps)" %(dic['v'][-1]*cfg.com.fs/(2*np.pi)))
    print("costas phase(scaled):",dic['angle_c'][-1]/(np.pi))

    plt.figure()
    ax = plt.subplot(211)
    ax.plot(ts,list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(dic['recovered_ph'])-np.array(dic['original_ph']))),label='phase_error')
    
    plt.grid(1)
    ax = plt.subplot(212)
    ax.plot(ts,list(map(lambda x:x.real,dic['mon_iq'])),label='cos')
    ax.plot(ts,list(map(lambda x:x.imag,dic['mon_iq'])),label='sin')
    ax.plot(ts,dic['demon'],label='demon')
    plt.legend()
    plt.show()
    err_a = list(map(lambda x: x+2*np.pi if x<-np.pi else (x-2*np.pi if x>np.pi else x),np.array(dic['recovered_ph'])-np.array(dic['original_ph'])))
    err_arr = np.array(err_a[int(cfg.com.N*0.5):])
    std = np.sqrt(np.sum(np.square(err_arr))/err_arr.size)
    print("phase indicators:")
    print("mean: %f,rms: %f,range:%f" %(np.mean(err_arr),std,np.ptp(err_arr)))
    print("enob: ",np.log(2*np.pi/std)/np.log(2)-1.76)
    
    err_arr = np.array(dic['v'][int(cfg.com.N*0.5):])
    err_acc = np.array(dic['acc'][int(cfg.com.N*0.5):])
    std = np.sqrt(np.sum(np.square(err_arr-np.mean(err_arr)))/err_arr.size)
    v_mean = np.mean(err_arr)*cfg.com.fs/(2*np.pi)
    v_acc = np.mean(err_acc)*(cfg.com.fs**2)*pll.loop_filter.ki/(2*np.pi)
    # v_acc = np.mean(err_acc)
    print("velocity indicators:")
    print("mean: %f,rms: %f,range:%f,acc:%f" %(v_mean,std*cfg.com.fs/(2*np.pi),np.ptp(err_arr)*cfg.com.fs/(2*np.pi),v_acc))
    print("enob: ",np.log(1/std)/np.log(2)-1.76)


comb_tb()

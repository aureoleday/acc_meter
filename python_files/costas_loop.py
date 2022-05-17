#costas loop
import numpy as np
import matplotlib.pyplot as plt

def AGWN(Ps,snr):
    SNR = 10**(snr/10)
    # print(SNR)
    # Ps = reduce(lambda x,y:x+y,map(lambda x:x**2,sin))/sin.size
    # print(Ps)
    Pn = Ps/SNR
    # print(Pn)
    agwn = np.random.randn(1)[0]*(Pn**0.5)
    return agwn

# fft_size =512

def choose_windows(name='Hanning', N=20): # Rect/Hanning/Hamming 
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Rect':
        window = np.ones(N) 
    return window

def my_fft(din,fft_size):
    # temp = din[:fft_size]*choose_windows(name='Rect',N=fft_size)
    temp = din[:fft_size]
    fftx = np.fft.rfft(temp)/fft_size
    xfp = np.abs(fftx)*2
    return xfp

class LoopFilter(object):
    def __init__(self, gain, Bn, zeta):
        self.kp = (1/gain)*(4*zeta/(zeta+1/(4*zeta)))*Bn
        self.ki = (1/gain)*(4/(zeta+1/(4*zeta))**2)*(Bn**2) 
        self.integrater = 0
        self.lf_out = 0
        print("kp:%f, ki:%f" %(self.kp,self.ki))
        
    def advance_filter(self, phase_difference):
        self.integrater += self.ki*phase_difference
        self.lf_out = self.integrater + self.kp*phase_difference
        # print(self.lf_out)
        
    def ef(self):
        return self.lf_out

class COSTAS(object):
    # def __init__(self,lf_gain, lf_bandwidth, lf_damping):
    def __init__(self,fs,fc, lf_gain, lf_bandwidth, lf_damping):
        self.n = 0
        self.fs = fs
        self.fc = fc
        self.phase_estimate = 0.0
        self.phase_estimate_d = 0.0
        self.vs = np.sin(self.phase_estimate)
        self.vc = np.cos(self.phase_estimate)
        # self.ve = np.cos(self.phase_estimate)
        self.phase_difference = 0.0
        self.loop_filter = LoopFilter(lf_gain, lf_bandwidth, lf_damping)
        
    def update_phase_estimate(self):
        self.n += 1
        self.phase_estimate_d = self.phase_estimate
        self.phase_estimate += self.loop_filter.ef()
        self.vs = 2*np.sin(2*np.pi*self.fc/self.fs*(self.n) + self.phase_estimate_d)
        self.vc = 2*np.cos(2*np.pi*self.fc/self.fs*(self.n) + self.phase_estimate_d)

    def update_phase_difference(self, in_sig):
        self.v_i = self.vc*in_sig
        self.v_q = self.vs*in_sig
        self.phase_difference = self.v_i*self.v_q

    def step(self, in_sig):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        self.update_phase_difference(in_sig)
        self.loop_filter.advance_filter(self.phase_difference)
        self.update_phase_estimate()
        # print("ced:%f,cef:%f,cpe:%f" %(self.phase_difference,self.loop_filter.ef(),self.phase_estimate))

class PLL(object):
    # def __init__(self,lf_gain, lf_bandwidth, lf_damping):
    def __init__(self,fs,fc, lf_gain, lf_bandwidth, lf_damping):
        self.n = 0
        self.fs = fs
        self.fc = fc
        self.phase_estimate = 0.0
        self.phase_estimate_d = 0.0
        self.vs = np.sin(self.phase_estimate)
        self.vc = np.cos(self.phase_estimate)
        self.phase_difference = 0.0
        self.loop_filter = LoopFilter(lf_gain, lf_bandwidth, lf_damping)
        
    def update_phase_estimate(self):
        self.n += 1
        self.phase_estimate_d = self.phase_estimate
        self.phase_estimate += self.loop_filter.ef()
        self.vs = -2*np.sin(2*np.pi*self.fc/self.fs*(self.n) + self.phase_estimate_d)
        self.vc = 2*np.cos(2*np.pi*self.fc/self.fs*(self.n) + self.phase_estimate_d)

    def update_phase_difference(self, d_i,d_q):
        self.phase_difference = d_i*self.vs - d_q*self.vc

    def step(self, d_i,d_q):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        self.update_phase_difference(d_i,d_q)
        self.loop_filter.advance_filter(self.phase_difference)
        self.update_phase_estimate()
        # print("ced:%f,cef:%f,cpe:%f" %(self.phase_difference,self.loop_filter.ef(),self.phase_estimate))

class COMB(object):
    def __init__(self,fs,fc,fm, lf_gain, lf_bandwidth, lf_damping):
        self.costas = COSTAS(fs,fc,0.5, 0.02, 0.707)
        self.n = 0
        self.fs = fs
        self.fc = fc
        self.fm = fm
        self.phase_estimate = 0.0
        self.phase_estimate_d = 0.0
        self.rad = 0.0
        self.vs = np.sin(self.phase_estimate)
        self.vc = np.cos(self.phase_estimate)
        self.phase_difference = 0.0
        self.loop_filter = LoopFilter(lf_gain, lf_bandwidth, lf_damping)
        
    def update_phase_estimate(self):
        self.n += 1
        self.phase_estimate_d = self.phase_estimate
        self.phase_estimate += self.loop_filter.ef()
        self.rad = 2*np.pi*self.fm/self.fs*(self.n) + self.phase_estimate_d
        self.vs = 2*np.sin(self.rad)
        self.vc = 2*np.cos(self.rad)
        

    def update_phase_difference(self, d_i,d_q):
        self.phase_difference = (d_i*self.vs - d_q*self.vc)
        # self.phase_difference = (d_q*self.vc - d_i*self.vs)

    def step(self, d_i,d_q):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        self.update_phase_difference(d_i,d_q)
        # self.costas.step(self.phase_difference)
        self.costas.step(d_i)
        self.loop_filter.advance_filter(self.costas.vc*self.phase_difference)
        self.update_phase_estimate()
        # print("ced:%f,cef:%f,cpe:%f" %(self.phase_difference,self.loop_filter.ef(),self.phase_estimate))

fm = 10
fs = 1000
fc = 100
N = 400
SNR = 60

def costas_tb():    
    pll = COSTAS(fs,fc,0.5, 0.04, 0.707)
    # pll = PLL(fs,fm,0.5, 0.02, 0.707)
    phi = np.pi*(0.5)
    
    sig_fc = []
    out = []
    
    ed = []
    ef = []
    pe = []
    mod = []
    for i in range(0, N - 1):
        sig_fc.append(np.cos(2*np.pi*fc/fs*i + phi))
        in_sig = np.cos(2*np.pi*fc/fs*i + phi)*np.cos(2*np.pi*fm/fs*i)
        pll.step(in_sig)
        # ref.append(in_sig)
        # demod.append(pll.v_i)
        # diff.append(pll.loop_filter.ef())
        mod.append(in_sig)
        out.append(0.5*pll.vs)
        ed.append(pll.phase_difference)
        ef.append(pll.loop_filter.ef())
        pe.append(pll.phase_estimate)

    plt.close('all')
    plt.figure()
    ax = plt.subplot(411)
    # ax.plot(ref,label='sig_in')
    ax.plot(sig_fc,label='carrier')
    ax.plot(out,label='out')
    plt.legend()
    # ax = plt.subplot(312)
    # ax.plot(diff,label='diff')
    # plt.legend()
    # ax = plt.subplot(313)
    # ax.plot(demod,label='demod')
    # plt.legend()
    
    # plt.figure()
    ax = plt.subplot(412)
    ax.plot(ed,label='ed')
    plt.legend()
    ax = plt.subplot(413)
    ax.plot(ef,label='ef')
    plt.legend()
    ax = plt.subplot(414)
    ax.plot(pe,label='pe')
    plt.legend()
    plt.show()
    
    plt.figure()
    # hs_costas = my_fft(out,N-1)
    # hs_pll = my_fft(pll_vc,N-1)
    hs_vc = my_fft(out,N-1)
    hs_id = my_fft(mod,N-1)
    # print(costas_vc)
    
    # ax = plt.subplot(211)
    # ax.plot(hs_costas,label='carrier')
    # ax.plot(hs_pll,label='velocity')
    # plt.legend()
    
    ax = plt.subplot(111)
    ax.plot(hs_vc,label='fc')
    ax.plot(hs_id,label='velocity modulation')
    # ax.plot(hs_id)
    plt.legend()
    plt.show()

def pll_tb():
    # costas = COSTAS(fs,fc,0.5, 0.02, 0.707)
    pll = PLL(fs,fm,0.5, 0.05, 0.707)
    phi = np.pi/2

    ref = []
    # out = []
    diff = []
    demod = []
    for i in range(0, N - 1):
        d_i = np.cos(2*np.pi*fm/fs*i+phi)
        d_q = np.sin(2*np.pi*fm/fs*i+phi)
        pll.step(d_i,d_q)
        ref.append(d_i)
        demod.append(pll.vc)
        diff.append(pll.loop_filter.ef())

    plt.figure()
    ax = plt.subplot(311)
    ax.plot(ref,label='sig_in')
    # ax.plot(out,label='vc')
    plt.legend()
    ax = plt.subplot(312)
    ax.plot(diff,label='diff')
    plt.legend()
    ax = plt.subplot(313)
    ax.plot(demod,label='demod')
    plt.legend()   

def comb_tb():
    pll = COMB(fs,fc,fm,0.5, 0.01, 1)
    # phi = np.pi
    # theta = np.pi
    phi = 0.9*np.pi
    theta = 0.0*np.pi

    i_dr = []
    q_dr = []
    i_d = []
    q_d = []
    costas_vc = []
    pll_vc = []
    pll_vs = []
    
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
        i_dr.append(np.cos(2*np.pi*fm/fs*i + theta)+AGWN(0.5,SNR))
        q_dr.append(np.sin(2*np.pi*fm/fs*i + theta)+AGWN(0.5,SNR))
        # i_dr.append(np.sin(2*np.pi*fm/fs*i + theta)+AGWN(0.5,SNR))
        # q_dr.append(np.cos(2*np.pi*fm/fs*i + theta)+AGWN(0.5,SNR))
        
        # i_dr.append(np.cos(2*np.pi*fm/fs*i + theta))
        # q_dr.append(np.sin(2*np.pi*fm/fs*i + theta))
        d_i = np.sin(2*np.pi*fc/fs*i + phi)*i_dr[-1]
        d_q = np.sin(2*np.pi*fc/fs*i + phi)*q_dr[-1]
        gamma.append(np.arctan2(d_q,d_i))
        alpha.append(np.arctan2(q_dr[-1],i_dr[-1]))
        pll.step(d_i,d_q)
        i_d.append(d_i)
        q_d.append(d_q)
        pll_vc.append(pll.vc)
        pll_vs.append(pll.vs)
        costas_vc.append(pll.costas.vc)
        
        ed.append(pll.phase_difference)
        ed_c.append(pll.costas.v_i)
        ef.append(pll.loop_filter.ef())
        ef_c.append(pll.costas.loop_filter.ef())
        pe.append(pll.phase_estimate)
        pe_c.append(pll.costas.phase_estimate)
        # beta.append((pll.rad)%(2*np.pi)-np.pi)
        beta.append(np.arctan2(pll.vs,pll.vc))
       

    plt.close('all')
    plt.figure()
    ax = plt.subplot(411)
    ax.plot(i_d,label='I')
    ax.plot(q_d,label='Q')
    ax.set_title('input signal')
    plt.legend()

    ax = plt.subplot(412)
    ax.plot(costas_vc,label='costas_vco')
    ax.plot(pll_vc,label='pll_vc',color='b')
    ax.plot(pll_vs,label='pll_vs',color='r')
    ax.set_title('VCO')
    plt.legend()

    # ax = plt.subplot(613)
    # ax.plot(i_dr,label='i',color='b')
    # ax.plot(q_dr,label='q',color='r')
    # plt.legend()

    # ax = plt.subplot(613)
    # ax.plot(ed,label='ed')
    # ax.plot(ed_c,label='ed_c')
    # plt.legend()
    
    # ax = plt.subplot(413)
    # # ax.plot(ed,label='ed')
    # ax.plot(ef,label='ef')
    # ax.plot(ef_c,label='ef_c')
    # ax.set_title('frequency error')
    # plt.legend()
    ax = plt.subplot(413)
    ax.plot(pe,label='pe')
    ax.plot(pe_c,label='epe_c')
    ax.set_title('phase error')
    plt.legend()
    ax = plt.subplot(414)    
    # ax.plot(gamma,label='signal_in')
    ax.plot(alpha,label='raw data')
    ax.plot(beta,label='recovered')
    ax.set_title('phase compare')
    plt.legend()
    plt.show()
    print("ef:%f,pe_c:%f,pe:%f" %(ef[-1],pe_c[-1]/(np.pi),pe[-1]))
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
    # plt.show()
    # return i_d
    

# costas_tb()
# pll_tb()
comb_tb()

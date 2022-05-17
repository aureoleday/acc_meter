#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 18:15:46 2022

@author: aureoleday
"""

from __future__ import division
from numpy.random import rand, randn
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import math

qfunc = lambda x: 0.5 * scipy.special.erfc(x/np.sqrt(2))

sf_range = [7,8,9,10,11,12]
EbN0dB_range = range(-10,15)

itr = len(EbN0dB_range)
itr_sf = len(sf_range)

## BPSK
bpsk_ber = [None for _ in range(itr)] # BER
bpsk_SNR_dB = [None for _ in range(itr)]  # Signal to noise ratio in dB
bpsk_M = 2               # Modulation Order
bpsk_k = np.log2(bpsk_M) # No of bits per symbol (BPSK is 1 bit per symbol)
bpsk_Rfec = 1
bpsk_Rs = 1
bpsk_BWn = 1  # Noise bandwidth

## LoRa
lora_ber = [[None for _ in range(itr)] for _ in range(itr_sf)]
lora_SNR_dB = [[None for _ in range(itr)] for _ in range(itr_sf)]  # Signal to noise ratio in dB
lora_Rs = [None for _ in range(itr_sf)]
lora_Rb = [None for _ in range(itr_sf)]
lora_Rfec = 1
lora_bw = 125e3
lora_BWn = 125e3

for s in range (0, itr_sf):
  sf = sf_range[s]
  lora_Rs[s] = ( lora_bw / ( 2 ** sf ) )
  lora_Rb[s] = lora_Rs[s] * sf

for n in range (0, itr):
  EbNOdB = EbN0dB_range[n]

  # For a matched-filter BPSK system, BWn = Rs
  bpsk_SNR_dB[n] = EbNOdB  + 10 * np.log10(bpsk_Rs) + 10 * np.log10(bpsk_k) + 10 * np.log10(bpsk_Rfec) - 10 * np.log10(bpsk_BWn)

  EbN0 = 10.0 ** (EbNOdB / 10.0)
  bpsk_ber[n] = qfunc(np.sqrt(2 * EbN0)) # or ber[n] = Pb(np.sqrt(EbN0))

  # LoRa
  for s in range (0, itr_sf):
    sf = sf_range[s]

    # Convert EbN0 to SNR
    # We consider matched transmit and receive filters (lora_Rs == BWn)
    lora_SNR_dB[s][n] = EbNOdB + 10 * np.log10(sf)
    # lora_SNR_dB[s][n] = EbNOdB + 10 * np.log10(lora_Rs[s]) + 10 * np.log10(sf) + 10 * np.log10(lora_Rfec) - 10 * np.log10(lora_BWn)
    # Calculate BER for LoRa CSS
    lora_ber[s][n] = qfunc(( log12(sf) / np.sqrt(2) ) * EbN0)

plt.figure()
plt.plot(EbN0dB_range, bpsk_ber, '-o', label='BPSK')
for i,b in enumerate(css_ber):
  plt.plot(EbN0dB_range, b,'-o', label='LoRa CSS, SF {}'.format(sf_range[i]))
axes = plt.gca()
axes.set_xlim([EbN0dB_range[0], EbN0dB_range[-1]])
axes.set_ylim([1e-4, 1e-0])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('$E_{b}/N_{0}$ (dB)')
plt.ylabel('Bit error rate, BER')
plt.grid(True)
plt.legend(loc = 'best')

plt.figure()
plt.plot(bpsk_SNR_dB, bpsk_ber, '-o', label='BPSK')
for i,b in enumerate(lora_ber):
    plt.plot(lora_SNR_dB[i], lora_ber[i], '-o', label='LoRa (sf {})'.format(sf_range[i]))
axes = plt.gca()
axes.set_ylim([1e-4, 1e-0])
plt.xscale('linear')
plt.yscale('log')
plt.ylabel('Bit error rate')
plt.xlabel('Signal to noise ratio, SNR (dB)')
plt.grid(True)
plt.legend(loc = 'best')
plt.show()
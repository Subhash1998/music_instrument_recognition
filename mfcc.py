#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:01:54 2019

@author: subhash
"""
###################################import libraries################################
import fnmatch
import numpy as np
import scipy.io.wavfile
import scipy.fftpack
import matplotlib.pyplot as plt
################################import from libraries##############################
from os import walk
from sklearn.preprocessing import LabelEncoder
from scipy.fftpack import dct


def find_mfcc(location,label):
    
    sample_rate,signal = scipy.io.wavfile.read(location)
    signal = signal[0:int(3.5 * sample_rate)]
    
    #plt.plot(signal)
    #plt.show()
    
    pre_emphasis = 0.97
    
    emphasized_signal = np.append(signal[0],signal[1:] - pre_emphasis*signal[:-1])
    signal_length=len(emphasized_signal)
    #plt.plot(emphasized_signal)
    #plt.show()
    
    frame_size=0.025
    frame_stride=0.01
    
    frame_length=int(round(frame_size*sample_rate))
    frame_step=int(round(frame_stride*sample_rate))
    
    num_frames = int(np.floor(float(np.abs(signal_length-frame_length))/frame_step)) + 1
    
    ######################    padding done to equalize number of samples in each frame   #######################
    pad_signal_length = num_frames*frame_step + frame_length
    
    z = np.zeros(pad_signal_length-signal_length)
    
    pad_signal = np.append(emphasized_signal,z)
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    
    frames = pad_signal[indices.astype(np.int32,copy=False)]
    
    frames *= np.hamming(frame_length)
    
    #plt.plot(frames)
    #plt.show()
    
    NFFT=512
    
    mag_frames = np.absolute(np.fft.rfft(frames,NFFT))
    
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    
    ###################################   triangular filters #################################
    
    nfilt = 40
    
    low_freq_mel = 0
    high_freq_mel = (2595*np.log10(1+(sample_rate/2)/700))
    mel_points = np.linspace(low_freq_mel,high_freq_mel,nfilt+2)
    hz_points=(700*(10**(mel_points/2595)-1))
    bin = np.floor((NFFT+1)*hz_points/sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    
    
    num_ceps = 12
    cep_lifter = 22
    
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    final_mfcc=[]
    final_mfcc = np.mean(mfcc,axis=0)
    
    ans = []
    for i in final_mfcc:
        ans.append(i)
    ans.append(label)
    
    return ans


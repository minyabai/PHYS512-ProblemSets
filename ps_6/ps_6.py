## PHYS512 - Problem 6
## ==========================
## Minya Bai (260856843)

import numpy as np
import json
from matplotlib import pyplot as plt
from simple_read_ligo import *
import scipy.signal as sig
from scipy.interpolate import interp1d as interp1d

## Functions
## ==========================

def smooth(data):
    k = np.arange(len(data))
    g = np.fft.fft(np.exp(-0.5*k**2))
    h = np.abs(np.fft.ifft(np.fft.fft(data)*g)) # convolve with gaussian
    return h

def whiten(strain, noise, freq, dt):
    Nt = len(strain)
    noise_interp = interp1d(freq, noise, kind='linear') # interpolate data
    fft_dat = np.fft.fft(strain*sig.get_window(window='tukey',Nx=Nt)) # apply window function
    fft_freq = np.fft.fftfreq(Nt, dt) # get freqs
    dat = fft_dat/np.sqrt(noise_interp(np.abs(fft_freq))) # whiten data
    white_dat = np.fft.irfft(dat,Nt)
    
    return white_dat, fft_freq

def match_filter(w_data, template, dt):
    fs = 1/dt
    fft_dat = np.fft.fft(w_data) / fs # fft of whiten data 
    fft_template = np.fft.fft(template) / fs # fft of template
        
    MF = fft_dat * fft_template.conjugate() # apply match filter
    MF = np.fft.irfft(MF,len(w_data))

    t = np.arange(len(w_data) * dt, step=dt)

    return MF, t

def snr(w_strain, template, w_template, freq, dt):
    Nt = len(w_strain)
    fs = 1/dt
    fft_wdat = np.fft.fft(w_strain) / fs # fft of whiten data
    fft_wtemplate = np.fft.fft(w_template) / fs 
    fft_template = np.fft.fft(template) / fs
    
    df = np.abs(freq[1] - freq[0]) # freq step
    noise = np.interp(np.abs(freq), freq, fft_template) # cal noise

    fft_SNR = fft_wdat * fft_wtemplate.conjugate() / np.sqrt(noise) # calculate fft of signal to noise ratio
    SNR = 2 * np.fft.irfft(fft_SNR,Nt) * fs # want signal to noise ratio
    sigma = np.sqrt(np.abs(1*(fft_wtemplate * fft_wtemplate.conjugate() / noise).sum() * df)) # normalization
    SNR = SNR/sigma # snr normalized

    shift = Nt // 2
    SNR = np.abs(np.roll(SNR,shift)) # shift to center data

    return SNR

def mid_freq(data, freq):
    pdf = np.cumsum(np.abs(data)) # cumulative probability density 
    return freq[(pdf < max(pdf)/2)][-1] # want where the probability is half

## Reading Data
## ===========================

data_loc = 'LOSC_Event_tutorial/'

with open(data_loc+'BBH_events_v3.json') as file:
    json_dat = json.load(file)

list_events = list(json_dat.keys())

N_events = len(list_events)
H1 = [] # Hanford data
L1 = [] # Livingston data
template_H = []
template_L = []

for i in range(N_events):
    event_name = str(list_events[i])
    fn_H1 = json_dat[event_name]['fn_H1']
    fn_L1 = json_dat[event_name]['fn_L1']
    fn_temp = json_dat[event_name]['fn_template']
    H1.append(read_file(data_loc+fn_H1))
    L1.append(read_file(data_loc+fn_L1))
    th,tl = read_template(data_loc+fn_temp)
    template_H.append(th)
    template_L.append(tl)

file.close()

## Initialize Arrays

plot = False 

noise_H = [] # noise model for Hanford
noise_L = [] # noise model for Livingston
freq_H = []
freq_L = []

## Part a)
for i in range(len(H1)):
    H_strain, H_dt = H1[i][:2]
    L_strain, L_dt = L1[i][:2]
    freqs_H1, noise_spec_H = sig.welch(H_strain,fs=1/H_dt,nperseg=1/H_dt,window='tukey')
    freqs_L1, noise_spec_L = sig.welch(L_strain,fs=1/L_dt,nperseg=1/L_dt,window='tukey')
    # print(len(noise_spec_H))
    noise_H.append(smooth(noise_spec_H))
    noise_L.append(smooth(noise_spec_L))
    freq_H.append(freqs_H1)
    freq_L.append(freqs_L1)

    if plot:
        plt.loglog(noise_H[i])
        plt.loglog(noise_L[i])
        plt.show()

## Part b)
wH = [] # 0 - strain, 1 - strain freq, 2 - template, 3 - template freq
wL = []

mfH = [] # match filter data for Hanford
mfL = [] # for Livingston

Htshift = []
Ltshift = []

# whiten_data = whiten(H1[0][0],noise_H[0])
for i in range(4):
    # whiten the data
    w_H,f_H = whiten(H1[i][0],noise_H[i],freq_H[i],H1[i][1])
    w_Ht,f_Ht = whiten(template_H[i],noise_H[i],freq_H[i],H1[i][1])
    w_L,f_L = whiten(L1[i][0],noise_L[i],freq_L[i],L1[i][1])
    w_Lt,f_Lt = whiten(template_L[i],noise_L[i],freq_L[i],L1[i][1])
    wH.append([w_H,f_H,w_Ht,f_Ht])
    wL.append([w_L,f_L,w_Lt,f_Lt])

    # apply match filter
    mf_H,Ht = match_filter(w_H,w_Ht,H1[i][1])
    mf_L,Lt = match_filter(w_L,w_Lt,L1[i][1])
    mfH.append(mf_H)
    mfL.append(mf_L)

    # shift data to center signal
    Ht_shift = np.fft.fftshift(Ht)
    Lt_shift = np.fft.fftshift(Lt)
    Htshift.append(Ht_shift)
    Ltshift.append(Lt_shift)
    
    if plot:
        plt.plot(Ht_shift,mf_H,'red',alpha=0.4)
        plt.plot(Lt_shift,mf_L,'blue',alpha=0.6)
        plt.show()

## Part c)
for i in range(4):
    H_noise = np.std(mfH[i][10000:40000]) # cal std in region of noise
    H_snr = np.max(np.abs(mfH[i])/H_noise) # find max (signal) and divide noise to get snr
    L_noise = np.std(mfL[i][10000:40000])
    L_snr = np.max(np.abs(mfL[i])/L_noise)
    HL_snr = np.sqrt(H_snr**2 + L_snr**2)
    print("Hanford: {}, Livingston: {}, Combined: {}".format(H_snr,L_snr,HL_snr))
    
## Part d)
for i in range(4):
    w_H,f_H,w_Ht,f_Ht = wH[i] # whiten data
    w_L,f_L,w_Lt,f_Lt = wL[i]

    H_SNR = snr(w_H,template_H[i],w_Ht,f_Ht,H1[i][1]) # calculate snr using noise model
    L_SNR = snr(w_H,template_L[i],w_Lt,f_Lt,L1[i][1])

    maxH_SNR = H_SNR[np.argmax(H_SNR)] # find peak for Hanford
    maxL_SNR = L_SNR[np.argmax(L_SNR)] # for Livingston

    HL_SNR = np.sqrt(maxH_SNR**2 + maxL_SNR**2) # combined data

    print("Hanford: {}, Livingston: {}, Combined: {}".format(maxH_SNR,maxL_SNR,HL_SNR))

    if plot:
        plt.plot(H_SNR)
        plt.plot(L_SNR)
        plt.show()

## Part e)
# freqs = []

for i in range(4):
    w_H,f_H,w_Ht,f_Ht = wH[i]
    w_L,f_L,w_Lt,f_Lt = wL[i]

    Hfreq = np.fft.rfftfreq(len(w_Ht),H1[i][1])
    Lfreq = np.fft.rfftfreq(len(w_Lt),L1[i][1])
    Htemp = np.fft.rfft(w_Ht)
    Ltemp = np.fft.rfft(w_Lt)
    
    f_H = mid_freq(Htemp,Hfreq)
    f_L = mid_freq(Ltemp,Lfreq)
    # freqs.append([f_H,f_L])
    print("Hanford: {}, Livingston: {}".format(f_H,f_L))

## Part f)
dt = [] # time differences between two detectors
for i in range(4):
    Ht = Htshift[i][np.argmax(mfH[i])] # time from Hanford
    Lt = Ltshift[i][np.argmax(mfL[i])] # time from Livingston
    dt.append(np.abs(Ht-Lt)) # calculate the difference between the two

del_t = np.mean(dt) # average time difference
del_p = del_t * (3*10**8) # d = v * t

print("Uncertainty in time: {}s".format(del_t))
print("Uncertainty in position: {}m".format(del_p))
    
    

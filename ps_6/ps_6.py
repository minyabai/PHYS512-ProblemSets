## PHYS512 - Problem 6
## ==========================
## Minya Bai (260856843)

import numpy as np
import json
from matplotlib import pyplot as plt
from simple_read_ligo import *
import scipy.signal as sig
        
def smooth(data):
    k = np.arange(len(data))
    g = np.fft.fft(np.exp(-k**2))
    h = np.abs(np.fft.ifft(np.fft.fft(data)*g))
    return h

def whiten(data, noise, freq, template, dt):
    fft_dat1 = np.fft.fft(data*sig.get_window(window='tukey', Nx=len(data)))
    fft_dat2 = np.fft.fft(template*sig.get_window(window='tukey', Nx=len(template)))
    fft_freq = np.fft.fftfreq(len(template),dt)
    noise_dat = np.interp(np.abs(fft_freq),freq,noise)
    spectr1 = fft_dat1/np.sqrt(noise_dat)
    spectr2 = fft_dat2/np.sqrt(noise_dat)
    white1 = np.fft.ifft(spectr1)
    white2 = np.fft.ifft(spectr2)

    return white1, white2, freq

def match_filter(data, noise, freq, template, dt):
    fft_dat = np.fft.fft(data) * dt 
    fft_spectr = np.fft.fft(template) * dt
    fft_freq = np.fft.fftfreq(len(template)) * dt
        
    MF = np.conjugate(fft_spectr) * fft_dat
    MF = np.real(np.fft.ifft(MF))

    scale = np.abs(MF).sum() * dt

    t = np.arange(len(data) * dt, step=dt)

    return MF/scale, t

def snr(data, noise, freq, template, dt):
    fft_freq = np.fft.fftfreq(len(template)) / dt
    df = fft_freq[1] - fft_freq[0]
    noise_dat = np.interp(np.abs(fft_freq), fft_freq, data)
    SNR = np.conjugate(template) * data / noise_dat
    sigma = np.sqrt(np.abs(SNR).sum() / noise_dat.sum() / dt)
    SNR = np.real(np.fft.ifft(SNR)) / dt
    SNR = SNR/sigma
    shift = int(len(data)/2)
    SNR = np.abs(np.roll(SNR,shift))

    return SNR

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
    H1_fname = json_dat[event_name]['fn_H1']
    L1_fname = json_dat[event_name]['fn_L1']
    fn_temp = json_dat[event_name]['fn_template']
    H1.append(read_file(data_loc+H1_fname))
    L1.append(read_file(data_loc+L1_fname))
    th,tl = read_template(data_loc+fn_temp)
    template_H.append(th)
    template_L.append(tl)

file.close()

noise_H = []
noise_L = []
freq_H = []
freq_L = []

# noise_H = get_noise(H1)
# noise_L = get_noise(L1)

## Part a)
for i in range(len(H1)):
    H_strain, H_dt = H1[i][:2]
    L_strain, L_dt = L1[i][:2]
    # noise_spec_H = get_noise(H_strain)
    # noise_spec_L = get_noise(L_strain)
    freqs_H1, noise_spec_H = sig.welch(H_strain,fs=1/H_dt,nperseg=2048,window='tukey')
    freqs_L1, noise_spec_L = sig.welch(L_strain,fs=1/L_dt,nperseg=2048,window='tukey')
    # print(len(noise_spec_H))
    noise_H.append(smooth(noise_spec_H))
    noise_L.append(smooth(noise_spec_L))
    freq_H.append(freqs_H1)
    freq_L.append(freqs_L1)

plt.loglog(noise_H[-1])
plt.loglog(noise_L[-1])
plt.show()

## Part b)
mfH = []
mfL = []

Htshift = []
Ltshift = []

# whiten_data = whiten(H1[0][0],noise_H[0])
for i in range(4):
    w_H,w_Ht,w_Hfreq = whiten(H1[i][0],noise_H[i],freq_H[i],template_H[i],H1[i][1])
    w_L,w_Lt,w_Lfreq = whiten(L1[i][0],noise_L[i],freq_L[i],template_L[i],L1[i][1])
    mf_H,Ht = match_filter(w_H,noise_H[i],w_Hfreq,w_Ht,H1[i][1])
    mf_L,Lt = match_filter(w_L,noise_L[i],w_Lfreq,w_Lt,L1[i][1])
    mfH.append(mf_H)
    mfL.append(mf_L)
    Ht_shift = np.fft.fftshift(Ht)
    Lt_shift = np.fft.fftshift(Lt)
    plt.plot(Ht_shift,mf_H,'red',alpha=0.4)
    plt.plot(Lt_shift,mf_L,'blue',alpha=0.6)
    plt.show()

## Part c)
for i in range(4):
    w_H,w_Ht,w_Hfreq = whiten(H1[i][0],noise_H[i],freq_H[i],template_H[i],H1[i][1])
    w_L,w_Lt,w_Lfreq = whiten(L1[i][0],noise_L[i],freq_L[i],template_L[i],L1[i][1])
    snr_H = snr(w_H,noise_H[i],w_Hfreq,w_Ht,H1[i][1])
    snr_L = snr(w_L,noise_L[i],w_Lfreq,w_Lt,L1[i][1])
    plt.plot(snr_H)
    plt.plot(snr_L)
    plt.show()

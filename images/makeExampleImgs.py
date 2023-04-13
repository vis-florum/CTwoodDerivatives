#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Export synthetic example images for later analysis

M = 100
N = 100
r = N/3	# radius of circle
delta = r/7 # transition width around circle
growthRingWidth = 10 # pixels
tanWaveStrength = 0.02 # strength of tangential wavyness
tanWavePeaks = 5 # number of peaks in tangential wavyness

I = np.zeros((M,N))

## Homogeneous Circle
mx = int(np.ceil(N/2))
my = int(np.ceil(M/2))
x,y = np.meshgrid(np.arange(1,N+1),np.arange(1,M+1))
R = np.zeros((M,N,2))
R[:,:,0] = (mx - x)
R[:,:,1] = (my - y)
RR = np.hypot(R[:,:,0],R[:,:,1])
I[RR < r] = 1
transition = (RR >= r) & (RR < r+delta)
I[transition] = np.exp((r - RR[transition])/2)
plt.imsave('circle.png',I) # export as png

## Diagonal Wave
k = 2*np.pi/(M/5)
m = 2*np.pi/(M/5)
x,y = np.meshgrid(np.arange(1,N+1),np.arange(1,M+1))
I = np.sin(k*x + m*y)
plt.imsave('diagWaves.png',I) # export as png

# Circular homogeneous waves around centre
k = 2*np.pi/growthRingWidth
x,y = np.meshgrid(np.arange(1,N+1),np.arange(1,M+1))
R = np.zeros((M,N,2))
R[:,:,0] = (mx - x)
R[:,:,1] = (my - y)
RR = np.hypot(R[:,:,0],R[:,:,1])
I = np.sin(k*RR)
transition = (RR >= r) & (RR < r+delta)
I[transition] = np.exp((r - RR[transition])/2)
I[RR >= r+delta] = 0
plt.imsave('circWaves.png',I) # export as png

## Noisy image
I = I + .1*np.random.randn(M,N)
plt.imsave('circWavesNoisy.png',I) # export as png


## Circular waves around centre with wavyness
k = 2*np.pi/growthRingWidth
x,y = np.meshgrid(np.arange(1,N+1),np.arange(1,M+1))
R = np.zeros((M,N,2))
R[:,:,0] = (mx - x)
R[:,:,1] = (my - y)
RR = np.hypot(R[:,:,0],R[:,:,1])
ang = np.arctan2(R[:,:,1],R[:,:,0])
RR = RR + tanWaveStrength*RR*np.sin(ang*tanWavePeaks)	# add tangential wavyness
I = np.sin(k*RR)
transition = (RR >= r) & (RR < r+delta)
I[transition] = np.exp((r - RR[transition])/2)
I[RR >= r+delta] = 0
plt.imsave('circWavesWavy.png',I) # export as png


## Plot
fig = plt.figure()
im = plt.imshow(I, animated=True)
plt.show()


#!/usr/bin/env python
from sigprocess import *
from calcmfcc import *
import scipy.io.wavfile as wav
import numpy

(rate,sig) = wav.read("0a7c2a8d_nohash_0.wav")
mfcc_feat = calcMFCC_delta_delta(sig,rate) 
print(mfcc_feat.shape)
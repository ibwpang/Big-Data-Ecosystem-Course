#!/usr/bin/env python
#!/usr/bin/python

import os, sys
import numpy as np 
import scipy as sp
import wave
import struct
import matplotlib.pylab as pl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('threshold', type = int, help = 'you need to provide an integer threshold')
args = parser.parse_args()
threshold = args.threshold

# Open a file
path = "/home/ubuntu/src/tf_p27/tensorflow/tensorflow/clouder/dataset/audio"
out = "/home/ubuntu/src/tf_p27/tensorflow/tensorflow/clouder/dataset/spectrogram"
dirs = os.listdir(path)

# This would print all the files and directories
for dir in dirs:
	files = os.listdir(path + "/" + dir)
	outs = os.listdir(out + "/" + dir)
	if len(files) == len(outs): continue
	print("start processing... " + dir)
	count = 0
	for file in files:
		if (file + ".png") in outs: continue
		if not file.endswith(".wav"): continue
		if count >= threshold: sys.exit(0)
		winsize=512
		shift=256
		fh=600 
		fl=60 
		filename = file
		wavefile = wave.open(path + "/" + dir + "/" + filename, 'r') # open for writing
		nchannels = wavefile.getnchannels()
		sample_width = wavefile.getsampwidth()
		framerate = wavefile.getframerate()
		numframes = wavefile.getnframes()
		# get wav_data
		wav_data = wavefile.readframes(-1)
		wav_data = np.fromstring(wav_data, 'Int16')

		Time=np.linspace(0, len(wav_data)/framerate, num=len(wav_data))

		pl.figure(1)
		pl.title('Signal Wave...')
		pl.plot(Time,wav_data)
		Fs = framerate
		pl.figure(2)
		pl.subplots_adjust(left=0,right=1,bottom=0,top=1)
		pl.specgram(wav_data, NFFT=1024, Fs=Fs, noverlap=512)
		pl.axis('off')
		pl.axis('tight')
		pl.savefig(out + "/" + dir + '/%s.png' % filename)
		count += 1
		print(dir + ": " + str(count) + " for this round")
	f = open("/home/ubuntu/src/tf_p27/tensorflow/tensorflow/clouder/log.txt","a")
	f.write("%s finished\n" % dir)
	f.close()

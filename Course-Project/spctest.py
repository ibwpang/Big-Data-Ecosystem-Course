#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np 
import scipy as sp
import wave
import struct
import matplotlib.pylab as pl



def getspectrogram(input):
	#********************参数设置********************%
	winsize=512;               #%%帧长设置为512  一般取20-50
	shift=256;                 #    %%帧移设置为256 一般取帧长的一半
	fh=600;                    #    %%设定最高基音频率
	fl=60;                     #    %%设定最低基音频率
	# 读取语音
	#filename = '0a9f9af7_nohash_0.wav'
	filename = input
	#subprocess.call(['ffmpeg', '-i', 'XXX.mp3', 'XXX.wav'])
	wavefile 
	try:
		wavefile = wave.open(filename, 'r') # open for writing
	except wave.Error as e:
	# if you get here it means an error happende, maybe you should warn the user
	# but doing pass will silently ignore it
		pass

	'''
	if not (wavefile.getname() == 'RIFF' or wavefile.getname() == 'WAVE'):
    	return
    '''

	#读取wav文件的四种信息的函数
	nchannels = wavefile.getnchannels()
	sample_width = wavefile.getsampwidth()
	framerate = wavefile.getframerate()
	numframes = wavefile.getnframes()


	print('nchannels:' + str(nchannels))
	print('sample_width:' +  str(sample_width))
	print('framerate:' +  str(framerate))
	print('numframes:' +  str(numframes))

	# get wav_data
	wav_data = wavefile.readframes(-1)
	wav_data = np.fromstring(wav_data, 'Int16')

	Time=np.linspace(0, len(wav_data)/framerate, num=len(wav_data))

	pl.figure(1)
	pl.title('Signal Wave...')
	pl.plot(Time,wav_data)
	#pl.show()

	#framerate就是16000, specgram！
	Fs = framerate
	pl.figure(2)
	pl.subplots_adjust(left=0,right=1,bottom=0,top=1)
	pl.specgram(wav_data, NFFT=1024, Fs=Fs, noverlap=512)
	pl.axis('off')
	pl.axis('tight')
	#pl.savefig("0a9f9af7_nohash_0.png")
	pl.savefig('/Users/gongman/Desktop/eightgram/%s.png' % input)
	pl.show()

#a = '0a9f9af7_nohash_0.wav'
#getspectrogram(a)
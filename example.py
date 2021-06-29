import preFunctions as pre
import librosa
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
#step1.载入音频
data,fs=librosa.load("P05-0013.WAV",sr=16000,mono=False)
if(data.shape[0]==2):
    left=data[0]
else:left=data
#step2.降采样
left_down=librosa.resample(left,fs,2000)
fs=2000
#step3.带通滤波
b,a=pre.bandPassFilter(8,fs,50,950)
left_filted=pre.filterSig(left_down,b,a)
#step4.归一化
left_normal=pre.normalization(left_filted)
#step5.分帧
#1.5S的窗，窗重叠为50%
frames=pre.enFrames(left_normal,1.5*fs,0.75*fs)
#汉明窗
hamming=pre.getHamming(int(1.5*fs))
#加窗
winedframes=pre.addWin(hamming,frames)
plt.plot(winedframes[5])
plt.show()
#恢复
reSig=pre.deFrames(winedframes,1.5*fs,0.75*fs,len(left_normal))
plt.subplot(2,1,1)
plt.plot(left_normal)
plt.subplot(2,1,2)
plt.plot(reSig)
plt.title("comparations between hanningwindow before and after")
plt.show()
pre.drawTimeFFt(left_down,2000)
pre.drawTimeFFt(reSig,2000)
import numpy as np
import pywt as wt
from numpy.fft import fft
from scipy import signal
import matplotlib.pyplot as plt
'''
    python-version>=3.6
    pip install pywavelets
    pip install numpy
    pip install scipy
    pip install matplotlib
'''
def cwt(data,fs,wavename,scale):
    #连续小波变换
    '''

    :param data: 原始信号
    :param fs: 采样率
    :param wavename: 小波名字
    :param scale: 尺度数量
    :return: 小波系数和对应中心频率
    '''
    fc=wt.central_frequency(wavename)
    cparam=2*fc*scale
    scales=cparam/np.arange(scale,1,-1)
    [coefs,freqs]=wt.cwt(data,scales,wavename,1.0/fs)
    return coefs,freqs
def getHamming(N):
    #返回长度为N的汉明窗
    return signal.windows.hanning(N,sym=False)
def bandPassFilter(N,fs,wl,wh):
    '''

    :param N: 滤波器阶数
    :param fs: 信号采样率
    :param wl: FL
    :param wh: FH
    :return: 滤波器参数b,a
    '''
    b,a=signal.butter(N,[2*wl/fs,2*wh/fs],'bandpass')
    return b,a
def filterSig(data,b,a):
    '''

    :param data: 原始信号
    :param b: 滤波器参数
    :param a: 滤波器参数
    :return: 滤波后信号
    '''
    return signal.filtfilt(b,a,data)
def normalization(signal):
    # Normalization
    #去除采样分辨率的影响，（量化电平通常16位）
    data_wav_norm = signal / (2 ** 15)
    #去除直流分量
    data_wav_norm -= data_wav_norm.mean()
    data_wav_norm /= abs(data_wav_norm).max() + 1e-10
    return data_wav_norm
def enFrames(x,wlen,inc):
    '''

    :param x: 信号
    :param wlen: 帧长
    :param inc: 帧移
    :return:
    '''
    #加窗分帧
    wlen=int(wlen)
    inc=int(inc)
    nx=len(x)
    if nx<=wlen:
        nf=1
    else:
        nf=int(np.ceil(1.0*nx-wlen+inc)/inc+1)
    padLength=int((nf-1)*inc+wlen)
    pad_signal=np.pad(x,(0,padLength-nx),'constant')
    indices=np.tile(np.arange(0,wlen),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(wlen,1)).T
    indices=np.array(indices,dtype=np.int32)
    frames=pad_signal[indices]
    return frames
def addWin(w,frames):
    '''

    :param w: 窗函数
    :param frames: 帧
    :return: 加过窗的帧
    '''

    newFrames=np.multiply(frames,w.T)
    return newFrames
def deFrames(frames,win,inc,N):
    '''

    :param frames: 分的帧
    :param win: 帧长
    :param inc: 帧移
    :param N: 原信号长度
    :return:恢复后的信号
    '''
    win=int(win)
    inc=int(inc)
    nf=frames.shape[0]
    nx=nf*inc-inc+win
    index=0
    signal=np.zeros(nx)
    for frame in frames:
        signal[index*inc:index*inc+win]+=frame
        index+=1
    #np.resize(signal,N)

    return signal[0:N]
def drawTF(signal,fs,N=64):
    '''
    绘制信号的连续小波变换时频图
    :param signal: 原始信号
    :param fs:采样率
    :param N: 小波变换阶数

    '''
    coefs, freqs = cwt(signal, fs, 'morl', N)
    t = np.arange(0, len(signal)/fs, 1.0 / fs)
    plt.pcolormesh(t,freqs,abs(coefs),cmap='jet',vmin=0,vmax=3)
    plt.xlabel('t/s')
    plt.ylabel("f/HZ")
    plt.colorbar()
    plt.show()
def drawTimeFFt(signal, sampling_rate):
    duration = len(signal) / float(sampling_rate)
    #signal=normalization(signal)
    plt.subplot(211)
    plt.plot(np.arange(0,duration,float(1/sampling_rate)),signal)
    plt.title('time domain')
    num_fft=int(sampling_rate*duration/2)
    fft_magnitude = abs(fft(signal))
    fft_magnitude = fft_magnitude[0:num_fft]
    plt.subplot(212)
    freq=np.arange(0,sampling_rate/2,sampling_rate/(2*num_fft))
    plt.plot(freq, fft_magnitude, 'black')
    plt.title('fft domain')
    plt.show()
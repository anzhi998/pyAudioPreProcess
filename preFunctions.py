import numpy as np
import pywt as wt
from numpy.fft import fft
from numpy import pi,polymul
from scipy import signal
import matplotlib.pyplot as plt

from scipy import fftpack
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
    plt.pcolormesh(t,freqs,abs(coefs),cmap=plt.cm.jet,vmin=0,vmax=3)
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
def getASFC(cwt1d,freqs):
    '''

    :param cwt1d:小波变换系数
    :param freqs: 对应的频率矩阵
    :return: ASF和ASC
    '''
    K=cwt1d.shape[0]
    N=cwt1d.shape[1]
    ASF=np.zeros(N)
    ASC=np.zeros(N)
    for index in range(0,N):
        temp=0.0
        up=0.0
        down=0.0
        for k in  range(0,K):
            up+=abs(cwt1d[k,index])*freqs[k]
            down+=abs(cwt1d[k,index])
            if(index==0):
                temp+=pow(abs(cwt1d[k,index]),2)
            else:
                temp+=pow((abs(cwt1d[k,index])-abs(cwt1d[k,index-1])),2)
        ASF[index]=np.sqrt(temp)
        ASC[index]=up/down
    return ASF,ASC
def getEnv(signal):
    hx=fftpack.hilbert(signal)
    return np.sqrt(signal**2+hx**2)
def A_weighting(fs):
    f1=20.598997
    f2=107.65265
    f3=737.86223
    f4=12194.217
    A1000=1.9997
    NUMs=[(2*pi*f4)**2*(10**(A1000/20)),0,0,0,0]
    DENs=polymul([1,4*pi*f4,(2*pi*f4)**2],[1,4*pi*f1,(2*pi*f1)**2])
    DENs=polymul(polymul(DENs,[1,2*pi*f3]), [1,2*pi*f2])
    return signal.bilinear(NUMs,DENs,fs)
def A_Weighted(x,fs):
    b,a=A_weighting(fs)
    y=signal.lfilter(b,a,x)
    return y
def lpc(s,p):
    n=len(s)
    Rp=np.zeros(p)
    for i in range(p):
        Rp[i]=np.sum(np.multiply(s[i+1:n],s[:n-i-1]))
    Rp0=np.matmul(s,s.T)
    Ep=np.zeros((p,1))
    k=np.zeros((p,1))
    a=np.zeros((p,p))
    Ep0=Rp0
    k[0]=Rp[0]/Rp0
    a[0,0]=k[0]
    Ep[0]=(1-k[0]**2)*Ep0
    if p>1:
        for i in range(1,p):
            k[i]=(Rp[i]-np.sum(np.multiply(a[:i,i-1],Rp[i-1::-1])))/Ep[i-1]
            a[i,i]=k[i]
            Ep[i]=(1-k[i]**2)*Ep[i-1]
            for j in range(i-1,-1,-1):
                a[j,i]=a[j,i-1]-k[i]*a[i-j-1,i-1]
    ar=np.ones(p+1)
    ar[1:]=-a[:,p-1]
    G=np.sqrt(Ep[p-1])
    return np.array(ar)
def Bruit_enhance(x,fs,p):
    N=len(x)
    x_dct=fftpack.dct(x)
    funit=float(fs/N)
    # w1=0.05
    # w2=0.5
    # w3=0.3
    # w4=0.15
    # w1=0.5
    # w2=0.3
    # w3=0.15
    # w4=0.05
    Xdct_band1=list()
    Xdct_band2=list()
    Xdct_band3=list()
    Xdct_band4=list()
    for i in range(N):
        w=(i+1)*funit
        if(w>=25 and w<=225):
            Xdct_band1.append(x_dct[i])
        elif(w>=300 and w<=700):
            Xdct_band2.append(x_dct[i])
        elif(w>=650 and w<=900):
            Xdct_band3.append(x_dct[i])
        elif(w>900):
            Xdct_band4.append(x_dct[i])
    # for i in range(N):
    #     w=(i+1)*funit
    #     if(w>=0 and w<=40):
    #         Xdct_band1.append(x_dct[i])
    #     elif(w>=80 and w<=180):
    #         Xdct_band2.append(x_dct[i])
    #     elif(w>=150 and w<=300):
    #         Xdct_band3.append(x_dct[i])
    #     elif(w>300):
    #         Xdct_band4.append(x_dct[i])
    Xdct_band1=np.array(Xdct_band1)
    Xdct_band2=np.array(Xdct_band2)
    Xdct_band3=np.array(Xdct_band3)
    Xdct_band4=np.array(Xdct_band4)
    hn1=lpc(Xdct_band1,p)
    hn2=lpc(Xdct_band2,p)
    hn3=lpc(Xdct_band3,p)
    hn4=lpc(Xdct_band4,p)
    E_BEF=w1*hn1+w2*hn2+w3*hn3+w4*hn4
    #X_BEF=signal.lfilter(-E_BEF[1:],[1],x)
    X_BEF=simpleMul(x,E_BEF[1:])
    return X_BEF
def simpleMul(signal,E_bef):
    N1 = len(E_bef)
    N2 = len(signal)
    output = np.zeros(N2)
    for index in range(N2):
        if (index <= N2 - N1):
            temp = signal[index:index + N1]
            output[index] = np.dot(E_bef,temp)
        else:
            temp = signal[index:]
            temp=np.pad(temp,(0,N1-len(temp)))
            output[index] = np.dot(E_bef,temp)
    return output
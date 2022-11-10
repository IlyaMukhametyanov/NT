#!/usr/bin/env python3
 # -*- coding: utf-8 -*-

import numpy as np
import mne
import csv
import pandas
from time import strptime
from datetime import datetime
import matplotlib.pyplot as plt
import time
import math
from scipy import *
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq

#перевести время в секунды


# ввод данных
# распознование шумов
# очистка шумов
# анализ результатов

def csv_to_list(path:str):
    with open(path, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)

    #print(data)
    print(len(data))
    print(type(data))
    print(data.pop())
    return data

def input_data(path:str):
    data = csv_to_list('monoVladCSV.csv')

    col_names = data.pop(0)
    count = len(data)
    print(col_names)
    df = pandas.DataFrame(data, columns = col_names)
    df = df.drop(columns='BioRadio Event')
    df = df.drop(columns='SpO2 pulse')
    df = df.drop(columns='PPG pulse')
    df = df.drop(columns='Heart Rate pulse')
    print(df.head())
    #df.plot(x="Elapsed Time", y="EEG1")
    format = '%H:%M:%S'

    for i in range(100000):
        time = df.iloc[i]['Elapsed Time'].split(':')

        for i in range(len(time)):
            time[i] = int(float(time[i])*1000) if i==2 else int(time[i])
        print(time)



    #col_names.head()
    #raw = mne.io.read_raw_fif(path)
    #print(raw)
    #print(raw.info)
    #return raw

def testspec():
    tenf = np.array([1.4458e-02, 9.8201e-03, 8.8783e-03, 1.2674e-02, 8.1712e-03, 4.0293e-03,
                     5.0523e-03, 3.7394e-03, 2.1965e-02, 2.4790e-02, 8.5859e-03, 1.3081e-02,
                     6.4008e-03, 2.0409e-02, 3.0875e-02, 2.3617e-02, 3.3676e-02, 5.3449e-02,
                     2.5645e-02, 1.3666e-02, 9.4334e-03, 2.2892e-02, 2.7402e-02, 2.2909e-02,
                     4.3816e-02, 3.0607e-02, 7.8202e-04, 5.9660e-03, 1.3925e-02, 4.5966e-02,
                     2.8455e-02, 1.4542e-03, 3.9536e-02, 1.4115e-02, 1.9775e-02, 2.2408e-02,
                     5.0217e-03, 7.0494e-04, 3.1064e-03, 4.0066e-02, 9.3312e-03, 7.5034e-03,
                     2.2241e-02, 2.9770e-02, 3.2652e-02, 1.1621e-02, 1.1621e-02, 2.0524e-02,
                     1.5398e-02, 4.5158e-02, 3.9614e-02, 1.1103e-03, 1.3038e-03, 1.8396e-03,
                     2.3575e-02, 5.2928e-02, 1.7184e-02, 1.4979e-02, 1.5802e-02, 1.3214e-02,
                     1.5350e-02, 2.6884e-04, 1.9442e-02, 2.4871e-02, 2.6104e-02, 3.0045e-02,
                     2.8420e-02, 4.0301e-03, 3.9526e-03, 1.5045e-02, 2.8030e-02, 2.5798e-02,
                     2.5798e-02, 2.0892e-02, 2.9606e-02, 5.2186e-03, 1.7122e-03, 5.2519e-02,
                     2.6752e-02, 1.3373e-02, 8.5243e-03, 3.6422e-03, 7.7749e-03, 1.1167e-02,
                     1.4266e-02, 7.4657e-03, 7.8746e-03, 7.4677e-05, 1.1132e-03, 1.1489e-02,
                     1.2060e-02, 1.0775e-02, 5.4699e-02, 3.5796e-02, 6.7368e-03, 1.9504e-02,
                     2.6181e-02, 4.1793e-03, 3.9625e-03, 1.6326e-02, 2.2435e-02, 1.3888e-02,
                     1.2613e-02, 2.4543e-02, 1.3389e-02, 4.0636e-03, 1.1115e-02, 4.2193e-02,
                     1.4601e-02, 2.0686e-02, 4.4160e-03, 7.6157e-03, 1.5917e-02, 2.5257e-02,
                     1.9146e-02, 1.1246e-02, 2.8330e-04, 8.7970e-03, 9.1066e-03, 1.5478e-02,
                     1.1184e-02, 1.7199e-02, 3.2053e-02, 2.1027e-03, 2.8986e-03, 7.4544e-03,
                     9.6567e-03, 4.6971e-03, 5.3027e-02, 6.2249e-02, 2.4887e-02, 1.2462e-02,
                     1.9944e-02, 1.3925e-02, 1.0937e-02, 1.4913e-02, 1.0694e-02, 5.9187e-02,
                     5.9187e-02, 5.9187e-02, 3.2870e-03, 1.1995e-02, 1.7444e-02, 4.0331e-03,
                     1.6909e-02, 2.4710e-02, 1.1389e-02, 1.8584e-03, 1.6878e-02, 2.3752e-02,
                     2.3752e-02, 1.9340e-02, 3.9648e-02, 1.7670e-02, 3.4614e-02, 6.7582e-03,
                     1.1932e-02, 9.6085e-03, 4.9216e-03, 2.4596e-02, 4.6430e-03, 9.6705e-03,
                     3.1486e-02, 9.9361e-03, 1.6469e-02, 6.0242e-03, 1.4003e-02, 3.5460e-02,
                     1.1701e-02, 1.8429e-03, 1.0323e-02, 3.1715e-02, 3.6270e-02, 1.1964e-02,
                     1.9469e-02, 1.6741e-02, 2.2261e-02, 4.1393e-02, 2.9687e-02, 1.4913e-02],
                    dtype=float)
    stft1, stft2, stft3 = signal.stft(tenf)
    plt.pcolormesh(stft2, stft1, abs(stft3), shading='auto')
    plt.show()
# Press the green button in the gutter to run the script.
def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate*duration, endpoint=False)
    frequencies = x * freq
    # 2pi для преобразования в радианы
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

def f():
    SAMPLE_RATE = 44100  # Гц
    DURATION = 5  # Секунды

    x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
    plt.plot(x, y)
    plt.show()
    _, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
    _, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)

    noise_tone = noise_tone * 0.3
    mixed_tone = nice_tone + noise_tone
    normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

    plt.plot(normalized_tone[:1000])
    plt.show()
    # число точек в normalized_tone
    N = SAMPLE_RATE * DURATION

    yf = fft(normalized_tone)
    xf = fftfreq(N, 1 / SAMPLE_RATE)

    plt.plot(xf, np.abs(yf))
    plt.show()
    yf = rfft(normalized_tone)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    plt.plot(xf, np.abs(yf))
    plt.show()

if __name__ == '__main__':
    #input_data('D://projects//NT//monoVladCSV.csv')
    f()
    csv_to_list('MonoVladCSV.csv')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

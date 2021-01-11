import scipy.io.wavfile as wav
import scipy.signal as sg
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import numpy as np


def main():
    audio_signal = wav.read('untitled.wav')
    signal_mono = [sample[0] for sample in audio_signal[1]]
    I = 2
    D = 2
    x_16 = signal_mono
    x_32 = sg.resample(x_16, len(x_16) * I)
    x_8 = sg.decimate(x_16, D)

    get_ratios(x_8, x_16, x_32, 1, 2)
    get_ratios(x_8, x_16, x_32, 2, 3)
    get_ratios(x_8, x_16, x_32, 3, 4)


def get_ratios(x_8, x_16, x_32, freq_low, freq_high):
    psd_8 = fft.fft(x_8) * np.conj(fft.fft(x_8))
    psd_16 = fft.fft(x_16) * np.conj(fft.fft(x_16))
    psd_32 = fft.fft(x_32) * np.conj(fft.fft(x_32))

    E_8 = np.sum(psd_8)
    E_16 = np.sum(psd_16)
    E_32 = np.sum(psd_32)

    p_8 = psd_8[freq_low * 1000:freq_high * 1000].sum()
    p_16 = psd_16[freq_low * 1000:freq_high * 1000].sum()
    p_32 = psd_32[freq_low * 1000:freq_high * 1000].sum()

    print(f"Frequency range{freq_low}-{freq_high}")
    print(f"E_8/p_8:{E_8 / p_8}")
    print(f"E_16/p_16:{E_16 / p_16}")
    print(f"E_32/p_32:{E_32 / p_32}")


if __name__ == '__main__':
    main()

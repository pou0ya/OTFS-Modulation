import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt

def awgn(signal, snr):
    snr_linear = 10**(snr/10)
    power_signal = np.mean(np.abs(signal)**2)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.normal(size=signal.shape)
    return signal + noise


T = 50e-6
Delf = 20e3

M = int(input("Enter the value of M: "))
N = int(input("Enter the value of N: "))
P = int(input("Enter the value of P: "))


delay_axis = np.linspace(0, T, N, endpoint=False)
doppler_axis = np.linspace(-Delf/2, Delf/2, N, endpoint=False)
dd_xmit = np.zeros((N, M))


tx_symbol_n = [0]
tx_symbol_m = [1]

for i in range(len(tx_symbol_n)):
    n = N//2 - tx_symbol_n[i]
    m = M//2 - tx_symbol_m[i]
    M_ax, N_ax = np.meshgrid(np.arange(-M//2+m, M//2+m), np.arange(-N//2+n, N//2+n))
    dd_xmit += np.sinc(M_ax/(0.5*np.pi)) * np.sinc(N_ax/(0.5*np.pi))


X = fftshift(fft(fftshift(ifft(dd_xmit, axis=0), axes=0), axis=1), axes=1)
s_mat = ifft(X, M, axis=0) * np.sqrt(M)
s = s_mat.flatten()
r = np.zeros_like(s)

channel_coefficients = np.random.rand(P)  
channel_delay = np.random.randint(1, M, P)  
channel_doppler = np.random.randint(1, N, P) 

for i in range(len(channel_delay)):
    r += np.roll(s * np.exp(-1j*2*np.pi*channel_doppler[i]/(M*N) * np.arange(1, M*N+1)), -channel_delay[i])

Y_DD = np.zeros((M, N))
for i in range(P):
    Y_DD[channel_delay[i], channel_doppler[i]] = channel_coefficients[i]

Y_DD_equalized = Y_DD

snr = 30
r = awgn(r, snr)
time_axis = np.linspace(0, T, M*N, endpoint=False)
r_mat = np.reshape(r, (N, M))
Y = fft(r_mat, M, axis=0) / np.sqrt(M)
dd_rx = fftshift(ifft(fftshift(fft(Y, axis=1), axes=1), axis=0), axes=0)

def plot_3d_bar(data, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_data, y_data = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
    ax.bar3d(x_data.flatten(), y_data.flatten(), np.zeros_like(data.flatten()), 1, 1, data.flatten(), shade=True)
    ax.set_xlabel('Doppler Intervals')
    ax.set_ylabel('Delay Intervals')
    ax.set_zlabel('Magnitude')
    ax.set_title(title)
    plt.show()

def plot_3d_surface(data, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_data, y_data = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
    ax.plot_surface(x_data, y_data, data, cmap='viridis')
    ax.set_xlabel('Doppler Intervals')
    ax.set_ylabel('Delay Intervals')
    ax.set_zlabel('Magnitude')
    ax.set_title(title)
    plt.show()

def plot_3d(data, time_axis, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(time_axis, np.real(data), np.imag(data), marker='o', linestyle='-', markersize=4)
    ax.set_xlabel('Time')
    ax.set_ylabel('Real Part')
    ax.set_zlabel('Imaginary Part')
    ax.set_title(title)
    plt.show()


plot_3d_bar(np.abs(dd_xmit), 'Delay-Doppler Domain')
plot_3d_surface(np.abs(X), 'Time-Frequency Domain')
plot_3d(np.abs(s), time_axis, 'Time Domain Signal')

plot_3d_bar(Y_DD_equalized, 'Delay-Doppler Domain after Channel Equalization')

plot_3d(np.abs(r), time_axis, 'Time Domain Signal')
plot_3d_surface(np.abs(Y), 'Time-Frequency Domain')
plot_3d_bar(np.abs(dd_rx), 'Delay-Doppler Domain')

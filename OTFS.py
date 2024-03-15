import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt

# Function to add white Gaussian noise to a signal
def awgn(signal, snr):
    snr_linear = 10**(snr/10)
    power_signal = np.mean(np.abs(signal)**2)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.normal(size=signal.shape)
    return signal + noise

# Given parameters
T = 50e-6
Delf = 20e3

# User inputs for M, N, and P
M = int(input("Enter the value of M: "))
N = int(input("Enter the value of N: "))
P = int(input("Enter the value of P: "))

# Creating arrays for delay and Doppler axes
delay_axis = np.linspace(0, T, N, endpoint=False)
doppler_axis = np.linspace(-Delf/2, Delf/2, N, endpoint=False)
dd_xmit = np.zeros((N, M))

# Transmit symbol coordinates
tx_symbol_n = [0]
tx_symbol_m = [1]

# Generating the transmitted signal in the delay-Doppler domain
for i in range(len(tx_symbol_n)):
    n = N//2 - tx_symbol_n[i]
    m = M//2 - tx_symbol_m[i]
    M_ax, N_ax = np.meshgrid(np.arange(-M//2+m, M//2+m), np.arange(-N//2+n, N//2+n))
    dd_xmit += np.sinc(M_ax/(0.5*np.pi)) * np.sinc(N_ax/(0.5*np.pi))

# Performing FFT and IFFT operations to convert to time-frequency domain
X = fftshift(fft(fftshift(ifft(dd_xmit, axis=0), axes=0), axis=1), axes=1)
s_mat = ifft(X, M, axis=0) * np.sqrt(M)
s = s_mat.flatten()  # Flattening the matrix to get the transmitted signal

# Initializing received signal
r = np.zeros_like(s)

# Generating random channel coefficients, delays, and Doppler shifts
channel_coefficients = np.random.rand(P)
channel_delay = np.random.randint(1, M, P)
channel_doppler = np.random.randint(1, N, P)

# Generating the received signal considering channel effects
for i in range(len(channel_delay)):
    r += np.roll(s * np.exp(-1j*2*np.pi*channel_doppler[i]/(M*N) * np.arange(1, M*N+1)), -channel_delay[i])

# Creating the channel matrix
Y_DD = np.zeros((M, N))
for i in range(P):
    Y_DD[channel_delay[i], channel_doppler[i]] = channel_coefficients[i]

# Equalizing the channel matrix (in this case, it seems like no equalization is applied)
Y_DD_equalized = Y_DD

# Adding white Gaussian noise to the received signal
snr = 30
r = awgn(r, snr)

# Converting the received signal back to the time-frequency domain
time_axis = np.linspace(0, T, M*N, endpoint=False)
r_mat = np.reshape(r, (N, M))
Y = fft(r_mat, M, axis=0) / np.sqrt(M)
dd_rx = fftshift(ifft(fftshift(fft(Y, axis=1), axes=1), axis=0), axes=0)

# Function to plot a 3D bar graph
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

# Function to plot a 3D surface graph
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

# Function to plot a 3D graph
def plot_3d(data, time_axis, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(time_axis, np.abs(data) * np.cos(np.angle(data)), np.abs(data) * np.sin(np.angle(data)), marker='o', linestyle='-', markersize=4)
    ax.set_xlabel('Time')
    ax.set_ylabel('Real Part')
    ax.set_zlabel('Imaginary Part')
    ax.set_title(title)
    plt.show()

# Plotting the original transmitted signal in the delay-Doppler domain
plot_3d_bar(np.abs(dd_xmit), 'Delay-Doppler Domain')

# Plotting the time-frequency domain representation of the transmitted signal
plot_3d_surface(np.abs(X), 'Time-Frequency Domain')

# Plotting the transmitted signal in the time domain
plot_3d(s, time_axis, 'Time Domain Signal')

# Plotting the equalized channel matrix in the delay-Doppler domain
plot_3d_bar(Y_DD_equalized, 'Delay-Doppler Domain after Channel Equalization')

# Plotting the received signal in the time domain
plot_3d(r, time_axis, 'Time Domain Signal')

# Plotting the time-frequency domain representation of the received signal
plot_3d_surface(np.abs(Y), 'Time-Frequency Domain')

# Plotting the received signal in the delay-Doppler domain
plot_3d_bar(np.abs(dd_rx), 'Delay-Doppler Domain')

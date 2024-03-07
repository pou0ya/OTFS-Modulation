clc;
clear;
close all;
%--------------------------------------------------------------------------
%delay Doppler grid dimensions
%span of delay axis
T = 50e-6;
%span of Doppler axis
Delf = 20e3;
%number of cells along Doppler
M = input("Enter the value of M: ");
%Number of cells across Delay
N = input("Enter the value of N: ");
%channel paths
P = input("Enter the value of P: ");
%delay axis
delay_axis = 0:T/N:T-T/N;
%Doppler axis
doppler_axis = -Delf/2:Delf/N:Delf/2-Delf/N;
%% %transmit signal
dd_xmit = zeros(N,M);
%ideal pulse transmit signal in dd domain
%dd_xmit(1,2) = 1;
%dd_xmit(1,4) = 1;
%delay axis location of symbol
tx_symbol_n = 0;
%Doppler axis location of symbol
tx_symbol_m = 1;
for i = 1:length(tx_symbol_n)
    n=N/2-tx_symbol_n(i);m=M/2-tx_symbol_m(i);
    [M_ax,N_ax] = meshgrid((-M/2+m:M/2+m-1),(-N/2+n:N/2+n-1));
    dd_xmit = dd_xmit + sinc(M_ax/(0.5*pi)).*sinc(N_ax/(0.5*pi));
end
%% ISFFT to convert to time-frequency grid
X = fftshift(fft(fftshift(ifft(dd_xmit).',1)).',2); %Method 1
%X = fft(ifft(dd_xmit).').'/sqrt(M/N); %Method 2
%% Heisenberg transform to convert to time domain transmit signal
s_mat = ifft(X,M,1)*sqrt(M); % Heisenberg transform %Method 1
%s_mat = ifft(X.')*sqrt(M); %Method 2
s = s_mat(:);
%% channel equalization
r = zeros(size(s));

%channel_delay = [10,12,13]; %Method 1
%channel_doppler = [2,3,4]; %Method 1

channel_coefficients = rand(P, 1); % Channel coefficients
channel_delay = randi([1, M], P, 1); % Delay shifts %Method 2
channel_doppler = randi([1, N], P, 1); % Doppler shifts %Method 2

for i=1:length(channel_delay)
    r = r + circshift(s.*exp(-1j*2*pi*channel_doppler(i)/(M*N).*(1:M*N)'),-channel_delay(i));
end

Y_DD = zeros(M, N);
for i = 1:P
    Y_DD(channel_delay(i), channel_doppler(i)) = channel_coefficients(i);
end
Y_DD_equalized = Y_DD;

snr = 30;
r = awgn(r,snr);
time_axis = (0:1/(M*N):1-1/(M*N))*T;
%% Receiver
r_mat = reshape(r,N,M);
% Wigner transform to convert to time frequency grid
Y = fft(r_mat,M,1)/sqrt(M); % Wigner transform %Method 1

%Y = fft(r_mat)/sqrt(M); % Wigner transform %Method 2
%Y = Y.'; %Method 2

% SFFT to transform to DD domain again
dd_rx = fftshift(ifft(fftshift(fft(Y).')).'); %Method 1
%dd_rx = ifft(fft(Y).').'/sqrt(N/M); %Method 2
%% plots (Method 1)
%figure;
%title("Transmit side");
%subplot(3,1,1);
%bar3(doppler_axis,delay_axis, abs(dd_xmit))
%subplot(3,1,2);
%surf(doppler_axis,delay_axis, real(X))
%subplot(3,1,3);
%plot3(time_axis,abs(s).*cos(angle(s)),abs(s).*sin(angle(s)));
%grid on

%figure;
%bar3(Y_DD_equalized);
%colorbar;
%xlabel('Doppler Intervals');
%ylabel('Delay Intervals');
%title('Delay-Doppler Domain Representation of OTFS Symbols after Channel Equalization');

%figure;
%title("Receive side");
%subplot(3,1,1);
%plot3(time_axis,abs(r).*cos(angle(r)),abs(r).*sin(angle(r)));
%grid on
%subplot(3,1,2);
%surf(doppler_axis,delay_axis, real(Y))
%subplot(3,1,3);
%bar3(doppler_axis,delay_axis, abs(dd_rx))
%% plots (Method 2)
figure('Name','Transmit side');
bar3(abs(dd_xmit));
xlabel('Doppler Intervals');
ylabel('Delay Intervals');
zlabel('Magnitude');
title('Delay-Doppler Domain');
%--------------------------------------------------------------------------
figure('Name','Transmit side');
surf(abs(X));
xlabel('Doppler Intervals');
ylabel('Delay Intervals');
zlabel('Magnitude');
title('Time-Frequency Domain');
%--------------------------------------------------------------------------
figure('Name','Transmit side');
plot3(time_axis,abs(s).*cos(angle(s)),abs(s).*sin(angle(s)));
grid on
xlabel('Time');
ylabel('Magnitude');
title('Time Domain Signal');
%--------------------------------------------------------------------------
figure('Name','Channel Equalization');
bar3(Y_DD_equalized);
colorbar;
xlabel('Doppler Intervals');
ylabel('Delay Intervals');
title('Delay-Doppler Domain after Channel Equalization');
%--------------------------------------------------------------------------
figure('Name','Receive side');
plot3(time_axis,abs(r).*cos(angle(r)),abs(r).*sin(angle(r)));
grid on
xlabel('Time');
ylabel('Magnitude');
title('Time Domain Signal');
%--------------------------------------------------------------------------
figure('Name','Receive side');
surf(abs(Y))
xlabel('Doppler Intervals');
ylabel('Delay Intervals');
zlabel('Magnitude');
title('Time-Frequency Domain');
%--------------------------------------------------------------------------
figure('Name','Receive side');
bar3(abs(dd_rx))
xlabel('Doppler Intervals');
ylabel('Delay Intervals');
zlabel('Magnitude');
title('Delay-Doppler Domain');
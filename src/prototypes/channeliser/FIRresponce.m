fs = 1024;  % Sample rate (MHz)
dec = 32; % Decimation factor
taps = 2^8; % Number of taps

% Time array
t = 0:(1/fs):((1/fs)*(taps-1));

% Polyphase prototype filter
s = fir1(taps - 1, 1/dec);

% Plot the prototype filter
figure(3), plot(t,s);

% Zero pad prototype filter up by a factor of 100 and calc spectrum
S = fftshift(fft((s)));

% Frequency array of spectrum
f = ((-floor(100 * taps / 2): (ceil(100*taps/2)-1))/(100*taps))/(t(2)-t(1));

% Plot the spectrum
figure(1); plot(f, 20*log10(abs(S)));

figure(2), plot(f, angle(S.*exp(1i*2*pi*f*(127.5/1024)))*180/pi);
axis([-16 16 -180 180]);

pause

% Write spectrum to file
Sps = S.*exp(1i*2*pi*f*(127.5/1024));
Sps(angle(Sps) ~= 0) = abs(Sps(angle(Sps) ~= 0));

St = zeros(16, length(Sps));
for n = 1:16
    St(n,:) = circshift(Sps', (n-1)*floor(dec/f(2)-f(1)))';
end

St = St/max(max(St)); % normalise
figure(3)
plot(f/512, St); axis([0 1 0 1.01])
xlabel('Frequency (half-cycles/second)');
ylabel('Normalised Gain');
title('Matlab Prototype PFB');

figure(4)
plot(f, angle(St), 'linewidth', 5); axis([0 512, -10 10])
xlabel('Frequency (MHz)');
ylabel('Phase (\theta)')


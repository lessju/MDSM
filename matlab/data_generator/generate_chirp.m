function [ chirp ] = generate_chirp( params, dm, snr )
%GENERATE_CHIRP Add a chirp to the voltage stream to model
%               a dispersed pulse

fch1  = params.center_frequency / 1e6;
foff  = params.bandwidth / 1e6;
s     = 1 / params.bandwidth; %params.sampling_time;

% Calculate chirp length in samples
chirp_length = ceil(4.15e3 * (((fch1 - foff/2)^-2) ...
                           - ((fch1 + foff/2)^-2)) ...
                           * dm / s);

% Generate chirp
fch1 = fch1 + foff / 2;
coeff = 2 * pi * dm / 2.41e-10;
freq = -foff : foff/chirp_length : 0;
phase = coeff * freq.^2 ./ ((fch1 + freq) .* fch1^2);
chirp = complex(cos(phase) , sin(phase) );

% Inverse FFT chirp into the time domain
chirp = ifft(chirp);

% Normalie complex array
chirp = chirp ./ (sqrt(mean(abs(chirp) .* 2)) / snr);

end

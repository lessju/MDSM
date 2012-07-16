function [ chirp ] = generate_chirp( voltage, params, dm )
%GENERATE_CHIRP Add a chirp to the voltage stream to model
%               a dispersed pulse

fch1  = params.center_frequency / 1e6;
foff  = params.bandwidth / 1e6;
s     = params.sampling_time;

% Calculate chirp length in samples
chirp_length = ceil(4.15e3 * (((fch1 - foff/2)^-2) ...
                           - ((fch1 + foff/2)^-2)) ...
                           * dm / s);
                       
% Voltage stream must be able to fit the entire pulse                       
if (chirp_length > size(voltage, 2))
    disp('Chirp does not fit in voltage stream');
    return;
end

% Generate chirp
fch1 = fch1 + foff / 2;
coeff = 2 * pi * dm / 2.41e-10;
freq = -foff : foff/chirp_length : 0;
phase = coeff * freq.^2 ./ ((fch1 + freq) .* fch1^2);
chirp = exp(1i .* phase  ./ chirp_length);

% Inverse FFT chirp into the time domain
chirp = ifft(chirp);% ./ chirp_length;

% Normalise chirp
%chirp = chirp ./ max(chirp);

end

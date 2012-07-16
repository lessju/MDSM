function [ voltage ] = generate_channel_rfi( voltage, params, frequency, snr )
%GENERATE_CHANNEL_RFI

bw = params.bandwidth;
frequency = frequency - params.center_frequency + bw / 2;
disp([frequency, bw]);
rfi = cos(2 * pi * frequency * ([0:1:size(voltage,2)-1] .* 1/bw));
rfi = fft(rfi);
if (frequency < bw / 2)
    rfi(size(rfi,2)/2:end) = 0;
else
    rfi(1:size(rfi,2)/2) = 0;
end
rfi = ifft(rfi / size(rfi,2));
voltage = voltage + snr * rfi;

end


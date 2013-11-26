function [ channeliser_voltages ] = fft_channeliser( voltages, nchans )
%FFT_CHANNELISER Simple FFT channeliser, top frequency first

channeliser_voltages = zeros(nchans, size(voltages, 2) / nchans);
spectra_per_channel  = size(voltages, 2) / nchans;

for i=1:1:spectra_per_channel
    spectrum = fft(voltages( (i-1) * nchans + 1 : i * nchans));
    channeliser_voltages(:,i) = fliplr(spectrum);
end

end


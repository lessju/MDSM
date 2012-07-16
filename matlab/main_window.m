% Add paths to script
addpath('data_generator','dedispersers','post_processors', 'rfi_filters', 'channelisers');

%% Define and apply parameters

% Create simulation parameters structure 
center_frequency     = 120e6;   % Hz
bandwidth            = 8e6;     % Hz
sampling_time        = 1 / bandwidth;
observation_length   = 2;       % seconds
number_channels         = 128;

parameters = struct('center_frequency', center_frequency, ...
                    'bandwidth', bandwidth,               ...
                    'channel_bandiwdth', bandwidth,       ...
                    'sampling_time', sampling_time,       ...
                    'number_channels', 1,                    ...
                    'observation_length', observation_length);

% Script options - bandpass
apply_bandpass = 0;
freq_end   = 0.15;
freq_start = 0.20;
atten      = 20;
ripple     = 1;

% Script options - RFI

% Script options - Dispersed pulses
dm            = 15;
pulse_width   = 0.001;   % s
snr           = 0.1;     % Relative the voltage mean


%% Generate raw voltages
voltage = generate_voltage_stream(parameters);

%% Add dispersed pulses
chirp = generate_chirp(voltage, parameters, dm);
voltage(1e6:1e6+size(chirp,2)-1) = voltage(1e6:1e6+size(chirp,2)-1) + chirp;

%% Add channel RFI
% fcarr = 1.26e6;
% rfi = cos(2 * pi * fcarr * ([0:1:size(voltage,2)-1] .* 1/8e6));
% rfi = fft(rfi);
% if (fcarr < bandwidth / 2)
%     rfi(size(rfi,2)/2:end) = 0;
% else
%     rfi(1:size(rfi,2)/2) = 0;
% end
% rfi = ifft(rfi);
% voltage = voltage + 0.1 * rfi;
   
%% Add RFI spike
%voltage(4e6:4e6+1e5) = 1.8 .* voltage(4e6:4e6+1e5);

%% Apply bandpass filter, if required
if (apply_bandpass)

    d=fdesign.highpass('Fst,Fp,Ast,Ap', freq_end, freq_start, atten, ripple);
    Hd = design(d, 'equiripple');
    voltage = filter(Hd, voltage);
    
    d = fdesign.highpass();
end

%% Apply channeliser
channeliser_voltages = fft_channeliser(voltage, number_channels);

% Adjust parameters
parameters.number_channels   = number_channels;
parameters.sampling_time     = parameters.sampling_time * number_channels;
parameters.channel_bandwidth = parameters.bandwidth / parameters.number_channels;

%% Calculate power
power_series = abs(channeliser_voltages).^2;

%% Perform de-dispersion
dedispersed_series = brute_force_dedisperser(power_series, parameters, dm);
figure
subplot(2,1,1);
imagesc(dedispersed_series);
subplot(2,1,2);
plot(sum(dedispersed_series));
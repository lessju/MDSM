function [ voltage_stream ] = generate_voltage_stream( params )
%GENERATE_CLEAN_SIGNAL Generate raw voltage stream with provided parameters
%                      having a 0 mean

% Generate clean gaussian noise
num_samples    = round(params.observation_length / params.sampling_time);
voltage_stream = wgn(1, num_samples, 0, 'complex');

% Normalise signal
voltage_stream = voltage_stream ./  sqrt(mean(abs(voltage_stream) .* 2)); 

end


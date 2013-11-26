function [ delta_t ] = dispersion_delay( f1, f2, dm, sampling_time )
% Calculate the dispersion delay in samples for the provided input
% parameters

delta_t = 4148.741601 * (f1^-2 - f2^-2) * dm / sampling_time;

end


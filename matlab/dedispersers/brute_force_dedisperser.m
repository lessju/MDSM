function [ dedispersed_series ] = brute_force_dedisperser( power_series, params, dm )
%BRUTE_FORCE_DEDISPERSER Perform brute-force dedisperion 

    function [ delta_t ] = delay(f1, f2, dm, sampling_time)
        delta_t = 4148.741601 * (f1^-2 - f2^-2) * dm / sampling_time;
    end

fch1 = (params.center_frequency + params.bandwidth / 2) * 1e-6;
foff = params.channel_bandwidth * 1e-6;
dedispersed_series = zeros(size(power_series));

for i = 1:params.number_channels
    delta_t = round(delay(fch1 - foff * (i-1), fch1, dm, params.sampling_time));
    if delta_t ~= 0
        series = power_series(i,:);
        dedispersed_series(i,:) = [ series(delta_t : end) NaN(1, delta_t - 1) ];
    end
end


end


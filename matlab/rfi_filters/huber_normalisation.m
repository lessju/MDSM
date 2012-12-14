function [ clipped_data ] = huber_normalisation( data, L )
%ENERGY_CLIPPING Apply RFI Mitigation technique: energy clipping

% Initialise global parameters
clipped_data = zeros(size(data));
p = 2 / (size(data, 2) + 1);
q = 2 / (size(data, 2) + 1);

c = 1 - 2 * (L * normpdf(L, 0, 1) - (L^2 - 1) * normcdf(-L, 0, 1));

    function val = huber_threshold (value, threshold)
        if value < -threshold
            val = -threshold;
        else
            if value > threshold
                val = threshold;
            else
                val = value;
            end
        end
    end

% Loop over frequencies
for f = 1 : size(data, 1)
    
    % Initialise for each frequency (need to change to detect non-gaussian
    % spikes in the data .. skewness of the few samples?
    m  = mean(data(f,1:256));
    s2 = std(data(f,1:256)).^2;
    s = sqrt(s2);
    
    % Loop over time samples
    for t = 2 : size(data, 2)
        clipped_data(f, t) = huber_threshold( (data(f, t) - m) / s, L);
        m  = m + p * s * clipped_data(f, t);
        s2 = (1 - q) * s2 + (q / c) * s2 * clipped_data(f, t) .^ 2;
        s = sqrt(s2);
    end
end

if (size(data, 1) == 1)
    figure;
    subplot(3,1,1);
    plot(data);
    subplot(3,1,2);
    plot(clipped_data);
    subplot(3,1,3);
    plot(clipped_data - data);
end

end


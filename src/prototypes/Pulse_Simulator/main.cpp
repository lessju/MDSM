#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;

// Telescope-dependent parameters
float fch1  =  418; // MHz
float bw    =  20;  // MHz

// Observation-specific parameters
unsigned nchans = 1024;
float    foff   = bw / (float) nchans;  // MHz
float    tsamp  = 1.0 / (bw * 1e6 / (float) nchans);  // s
float    period = 1000;   // ms
float    dm     = 25;

// Simulator-specific parameters
unsigned num_reps = 10, num_widths = 10, num_snr = 10, pad_length = 20;
unsigned widths[10] = {1, 3, 5, 11, 21, 51, 101, 201, 501, 1001};
float    snr[10]    = {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5};

// Calculate frequency-dependent shift
inline float dmdelay(float F1, float F2)
{  return (4148.741601 * ((1.0 / F1 / F1) - (1.0 / F2 / F2))); }

// Generate gaussian pulse
inline float *generate_pulse(unsigned width, unsigned *pulse_width)
{
    // Create pulse buffer
    *pulse_width = width * 6 + 1;
    float *pulse = (float *) malloc(*pulse_width * sizeof(float));
    float constant = 1.0 / sqrt(2 * M_PI);

    // Generate pulse
    for(unsigned i = 0; i < *pulse_width; i++)
    {
        float x = i / (float) width - 3;
        pulse[i] = constant * exp(- (x * x) / 2.0);
    }

    // Normalise pulse
    float max_val = 0;
    for(unsigned i = 0; i < *pulse_width; i++) max_val = (max_val < pulse[i]) ? pulse[i] : max_val;
    for(unsigned i = 0; i < *pulse_width; i++) pulse[i] /= max_val;

    return pulse;
}

int main()
{
    // Create random number generator (uniform, mean = 0, stddev = 1)
    boost::mt19937 rng;
    boost::normal_distribution<> nd(0.0, 1.0);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > normal(rng, nd);

    // Calculate pulse length
    unsigned period_len = period * 1e-3 / tsamp;
    unsigned long samples = 0;

    // Generate dispersion delay shifts
    unsigned long shifts[nchans];
    for(unsigned i = 0; i < nchans; i++)
        shifts[i] = dmdelay(fch1 - foff * i, fch1) * dm / tsamp;

    // Create file
    FILE *fp = fopen("/data/simulated_pulses.fil", "wb");

    // Create pulses file
    FILE *pulses_file = fopen("generated_pulses.txt", "w");
    fprintf(pulses_file, "[");

    // Pad buffer with empty data
    float *data;

    // Re-create buffer for pulses
    data = (float *) malloc(period_len * nchans * sizeof(float));

    // Loop over weights
    for(unsigned w = 0; w < num_widths; w++)
    {
        // Generate pulse
        unsigned pulse_width;
        float *pulse = generate_pulse(widths[w], &pulse_width);

        // Loop over snrs
        for(unsigned s = 0; s < num_snr; s++)
        {
            float SNR = snr[s] / sqrt(nchans);

            // Repeate for num_reps times
            for(unsigned r = 0; r < num_reps; r++)
            {
                // Overwrite buffer data
                data = (float *) malloc(period_len * nchans * sizeof(float));

                for(unsigned i = 0; i < period_len * nchans; i++) data[i] = normal();

                // Calculate pulse position
                unsigned long pulse_position = period_len / 2;// - (shifts[nchans / 2] + widths[w] / 2);

                // Add pulse to data buffer
                for(unsigned i = 0; i < nchans; i++)
                    for(unsigned j = 0; j < pulse_width; j++)
                        data[(pulse_position + shifts[i] + j) * nchans + i] += pulse[j] * SNR;

                // Write pulse information to file
                printf("Width = %d, SNR = %f [%d]\n", widths[w], snr[s], r);
                fprintf(pulses_file, "{'dm' : %.2f, 'width' : %d, 'snr' : %.2f, 'pos' : %lf },",
                                     dm, widths[w], snr[s], (samples + pulse_position + widths[w] / 2) * tsamp);
                samples += period_len;

                // Write buffer to file
                fwrite(data, sizeof(float), period_len * nchans, fp);
                fflush(fp);
                free(data);
            }
        }

        free(pulse);
    }

    fseek(pulses_file, -1, SEEK_CUR); // Remove extra trailing comma
    fprintf(pulses_file, "]");
    fclose(pulses_file);

    // Pad buffer with 10s of empty data
    data = (float *) malloc(1.0 / tsamp * pad_length * nchans * sizeof(float));
    for(unsigned i = 0; i < 1.0 / tsamp * pad_length * nchans ; i++) data[i] = normal();
    fwrite(data, sizeof(float), 1.0 / tsamp * pad_length * nchans, fp);
    free(data);
}

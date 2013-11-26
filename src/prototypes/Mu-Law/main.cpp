#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include "time.h"
#include <map>
#include "file_handler.h"

#include <xmmintrin.h>
#include <x86intrin.h>

using namespace std;

char *filename = "/home/lessju/Medicina_Channel_RFI_and_Pulsar.dat";
unsigned nchans = 512, nsamp = 16384;

// ========================= CPU HELPER FUNCTIONS ===========================
void read_data(float *buffer, unsigned nsamp, unsigned nchans)
{
    // Read file
    float *tempBuff = (float *) malloc(nsamp * nchans * sizeof(float));
    FILE *fp = fopen(filename, "rb");

    // Read header
    read_header(fp);

    unsigned num_read = read_block(fp, 32, tempBuff, nchans * nsamp);
    fclose(fp);

    // Transpose data
    unsigned i, j;
    for(i = 0; i < nchans; i++)
        for(j = 0; j < nsamp; j++)
            buffer[i * nsamp + j] = tempBuff[j * nchans + i];

    free(tempBuff);

    if (num_read != nsamp * nchans)
    {
        printf("Seems there's not enough data in the file\n");
        exit(0);
    }
}

// ========================================================================

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;

    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-nchans"))
           nchans = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       i++;
    }
}

// ========================= 32-bit Quantisation =============================

// Perform mu-law quantization
float mu = 128;

// Perform mu-law quantisation using SSE intrinsics
void mulaw_encode(float *data, unsigned char *encodedData, unsigned numValues)
{
    // First data pass, calculate maximim and minimum values
    float maxValue = FLT_MIN;
    for(unsigned i = 0; i < numValues; i++)
        maxValue = (maxValue < data[i]) ? data[i] : maxValue;

    float Q = 1.0 / 255.0;
    float inverse_Q = 1.0 / Q;
    float log_one_plus_mu = 1.0 / log10(1 + mu);
    float inverse_maxValue = 1.0 / maxValue;

    // Start encoding data four items at a time (assuming 128-bit
    // wide CPU vectors)
    for(unsigned i = 0; i < numValues; i++)
    {
        // First normalise input value to 1
        float datum = data[i] * inverse_maxValue;

        // NOTE: Our values are ALWAYS POSITIVE (no need for fabs(datum))
        datum = log10(1 + mu * datum) * log_one_plus_mu;

        // Use linear quantiser to encode datum (our range is 0 -> 1)
        encodedData[i] = (unsigned char) (((int)(datum * inverse_Q) * Q + Q * 0.5) * 255.0);
    }
}

// Perofrm mu-law quantisation using a lookup table for the logarithm calculation
void mulaw_encode_32bit_log_lookuptable(float *data, unsigned char *encodedData, unsigned numValues)
{
    // First data pass, calculate maximim and minimum values
    float maxValue = FLT_MIN;
    for(unsigned i = 0; i < numValues; i++)
        maxValue = (maxValue < data[i]) ? data[i] : maxValue;

    // Create the lookup table and calculate the values
    // The lookup table's input range is 1 -> mu
    // Table is created only once using global maximum value
    unsigned size = 65536;
    float lookup_table[size];
    for(unsigned i = 0; i < size; i++)
        lookup_table[i] = log10(1 + i * mu / (float) size );

    // Define some initial values for fast processing
    float Q = 1.0 / 255.0;
    float inverse_Q = 1.0 / Q;
    float log_one_plus_mu = 1.0 / log10(1 + mu);
    float inverse_maxValue = 1.0 / maxValue;

    // Start encoding data
    for(unsigned i = 0; i < numValues; i++)
    {
        float datum = data[i] * inverse_maxValue * size;
        datum = lookup_table[(int) datum] * log_one_plus_mu;
        encodedData[i] = (((int)(datum * inverse_Q) * Q + Q * 0.5) * 255.0);
    }
}

// Perform mu-law decoding
void mulaw_32bit_decode(float *data, unsigned char *encodedData, unsigned numValues)
{
    float log_one_plus_mu = log10(1 + mu);
    float invserse_mu = 1.0 / mu;
    float quant_interval = 1.0 / 255.0;

    // Start decoding data
    for(unsigned i = 0; i < numValues; i++)
    {
        float datum = encodedData[i] * quant_interval;
        data[i] = (pow(10.0, log_one_plus_mu * datum) - 1) * invserse_mu;
    }
}

// ========================= 8-bit Quantisation =============================

// Perofrm mu-law quantisation using a lookup table for the logarithm calculation
void mulaw_encode_16bit_log_lookuptable(short *data, unsigned char *encodedData, unsigned numValues)
{
    // We have a fixed maximum value of 32768
    float lookup_table[32769];
    float maxValue = 32768;

    // Create the lookup table and calculate its values
    // We only need 32768 value to cover the entire range
    // for signed short values (the negative values are
    // just mirrored on the negative y-axis)
    for(unsigned i = 0; i < maxValue + 1; i++) lookup_table[i] = log10(1 + i * mu / maxValue);

    // Define some initial values for fast processing
    float Q = 1.0 / 8.0;
    float inverse_Q = 1.0 / Q;
    float log_one_plus_mu = 1.0 / log10(1 + mu);

    // Start encoding data
    for(unsigned i = 0; i < numValues; i++)
    {
        // Extract signs of real and imaginary parts
        char real_sign = (data[i*2]   < 0) ? -1 : 1;
        char imag_sign = (data[i*2+1] < 0) ? -1 : 1;

        // Encode the value using the log lookup table
        float real_datum = lookup_table[abs(data[i*2])]   * log_one_plus_mu;
        float imag_datum = lookup_table[abs(data[i*2+1])] * log_one_plus_mu;

        // Quantise values to 3 bits (+ sign bit)
        unsigned char real_quant = ((char) (real_datum * inverse_Q) * Q + Q * 0.5) * 8;
        unsigned char imag_quant = ((char) (imag_datum * inverse_Q) * Q + Q * 0.5) * 8;

        // Combine values and sign bits to for 8-bit representation:
        // [rsign rX rX rX isign iX iX iX]
        unsigned char value = (real_sign  & 0x80)   |
                              ((real_quant & 0x07) << 4)  |
                              ((imag_sign  & 0x80) >> 4)  |
                              (imag_quant  & 0x07);
        encodedData[i] =  value;

       // printf("%d = %d, %d = %d = %d\n", (int) data[i*2], (int) real_quant, (int) data[i*2+1], (int) imag_quant, (int) value);
    }
}

// Perform mu-law decoding
void mulaw_16bit_decode(short *data, unsigned char *encodedData, unsigned numValues)
{
    float log_one_plus_mu = log10(1 + mu);
    float invserse_mu = 1.0 / mu;
    float quant_interval = 1.0 / 8.0;

    // Start decoding data
    for(unsigned i = 0; i < numValues; i++)
    {
        // Each char contains 2 value: real and complex
        // containg 4 bits, one for sign and the others for the value

        char real_sign  = ((encodedData[i] & 0x80) == 0) ? 1 : -1;
        char imag_sign  = ((encodedData[i] & 0x08) == 0) ? 1 : -1;
        char real_value = (encodedData[i] & 0x70) >> 4;
        char imag_value = encodedData[i] & 0x07;

        float real_datum = real_value * quant_interval;
        float imag_datum = imag_value * quant_interval;
        data[i*2]   = 32768 * (powf(10.0, (float) log_one_plus_mu * real_datum) - 1) * invserse_mu * real_sign;
        data[i*2+1] = 32768 * (powf(10.0, (float) log_one_plus_mu * imag_datum) - 1) * invserse_mu * imag_sign;
        printf("%d %d\n",data[i*2], data[i*2+1]);
    }
}

// ========================= Main functions =============================

// Main function 32bit
void test32BitConversion()
{
    struct timeval start, end;
    long mtime, seconds, useconds;

    // Define and allocate buffers
    unsigned char *encoded = (unsigned char *) malloc(nchans * nsamp * sizeof(float)) ;
    float *buffer = (float *) malloc(nchans * nsamp * sizeof(float));

    // Define files
    FILE *fp = fopen("Test_mulaw_original.dat", "wb");
    FILE *fe = fopen("Test_mulaw_decoded.dat", "wb");

    // Read 32-bit floating point data from file
    read_data(buffer, nsamp, nchans);

    // Write buffer to file
    fwrite(buffer, nchans * nsamp, sizeof(float), fp);
    fclose(fp);

    // Start performance counter
    gettimeofday(&start, NULL);

    // Encode data
    mulaw_encode_32bit_log_lookuptable(buffer, encoded, nchans * nsamp);

    // Stop performance counter
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Quantised data in: %ldms\n", mtime);

    // Reset original buffer to 0
    memset(buffer, 0, nchans * nsamp * sizeof(float));

    // Start performance counter
    gettimeofday(&start, NULL);

    // Decode data
    mulaw_32bit_decode(buffer, encoded, nchans * nsamp);

    // Stop performance counter
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Decoded data in: %ldms\n", mtime);

    // Write decoded buffer to file
    fwrite(buffer, nchans * nsamp, sizeof(float), fe);
    fclose(fe);
}

// Main function 16bit
void test16BitConversion()
{
    struct timeval start, end;
    long mtime, seconds, useconds;

    // Define and allocate buffers
    unsigned char *encoded = (unsigned char *) malloc(nchans * nsamp * sizeof(float)) ;
    float *buffer = (float *) malloc(nchans * nsamp * sizeof(float));

    // Define files
    FILE *fp = fopen("Test_mulaw_original_16.dat", "wb");
    FILE *fe = fopen("Test_mulaw_decoded_16.dat", "wb");

    // Generate 16-bit random values
    short *complexData = (short *) buffer;
    for(unsigned i = 0; i < nchans * nsamp; i++)
    {
        complexData[i * 2    ] = i * 0.1;
        complexData[i * 2 + 1] = i * 0.1;
    }

    // Write buffer to file
    fwrite(buffer, nchans * nsamp, sizeof(float), fp);
    fclose(fp);

    // Start performance counter
    gettimeofday(&start, NULL);

    // Encode data
    mulaw_encode_16bit_log_lookuptable(complexData, encoded, nchans * nsamp);

    // Stop performance counter
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Quantised data in: %ldms\n", mtime);

    // Reset original buffer to 0
    memset(buffer, 0, nchans * nsamp * sizeof(float));

    // Start performance counter
    gettimeofday(&start, NULL);

    // Decode data
    mulaw_16bit_decode(complexData, encoded, nchans * nsamp);

    // Stop performance counter
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Decoded data in: %ldms\n", mtime);

    // Write decoded buffer to file
    fwrite(buffer, nchans * nsamp, sizeof(float), fe);
    fclose(fe);
}

int main()
{
    if (0)
        // Test 32-bit to 8-bit conversion
        test32BitConversion();
    else
        // Test 16-bit to 4-bit conversion
        test16BitConversion();
}

#include "fftw3.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"

typedef enum { HAMMING, BLACKMAN, GAUSSIAN, KAISER } FirWindow;

unsigned nextPowerOf2(unsigned n)
{
    unsigned res = 1;
    while(1) 
    {
        if(res >= n) 
            return res;
        res *= 2;
    }
}

void interpolate(const double* x,
        const double* y, unsigned nX, unsigned n, double* result)
{
    unsigned nextX = 0;
    unsigned index = 0;

    for(double interpolatedX = 0.0; interpolatedX <= 1.0; interpolatedX += 1.0/(n-1), index++) {

        while(x[nextX] <= interpolatedX && nextX < nX - 1) {
            nextX++;
        }

        if(nextX == 0) 
        {
            fprintf(stderr, "Interpolation error when generating FIR filter. Exiting\n");
            exit(-1);
        }

        double prevXVal = x[nextX-1];
        double nextXVal = x[nextX];
        double prevYVal = y[nextX-1];
        double nextYVal = y[nextX];

        double rc = (nextYVal - prevYVal) / (nextXVal - prevXVal);
        double newVal = prevYVal + (interpolatedX - prevXVal) * rc;

        result[index] = newVal;
    }
}

// Compute the filter, similar to Octave's fir2(n, f, m, grid_n, ramp_n, window);
// Window and result must be of size n + 1.
// grid_n: length of ideal frequency response function
// ramp_n: transition width for jumps in filter response
// defaults to grid_n/20; a wider ramp gives wider transitions
// but has better stopband characteristics.
void generateFirFilter(unsigned n, double w, const double* window, double* result)
{
    // make sure grid is big enough for the window
    // the grid must be at least (n+1)/2
    // for all filters where the order is a power of two minus 1, grid_n = n+1;
    unsigned grid_n = nextPowerOf2(n + 1);
    unsigned ramp_n = 2; // grid_n / 20;

    // Apply ramps to discontinuities
    // this is a low pass filter
    // maybe we can omit the "w, 0" point?
    // I did observe a small difference
    double f[] = {0.0, w-ramp_n/grid_n/2.0, w, w+ramp_n/grid_n/2.0, 1.0};
    double m[] = {1.0, 1.0, 0.0, 0.0, 0.0};

    // grid is a 1-D array with grid_n+1 points. Values are 1 in filter passband, 0 otherwise
    double* grid = (double*) malloc((grid_n + 1) * sizeof(double));

    // interpolate between grid points
    interpolate(f, m, 5 /* length of f and m arrays */ , grid_n+1, grid);

    // the grid we do an ifft on is:
    // grid appended with grid_n*2 zeros
    // appended with original grid values from indices grid_n..2, i.e., the values in reverse order
    // (note, arrays start at 1 in octave!)
    // the input for the ifft is of size 4*grid_n
    // input = [grid ; zeros(grid_n*2,1) ;grid(grid_n:-1:2)];

    fftwf_complex* cinput  = (fftwf_complex*) fftwf_malloc(grid_n*4*sizeof(fftwf_complex));
    fftwf_complex* coutput = (fftwf_complex*) fftwf_malloc(grid_n*4*sizeof(fftwf_complex));

    if(cinput == NULL || coutput == NULL) {
        fprintf(stderr, "Cannot allocate buffer to generate FIR filter. Exiting\n");
        exit(-1);
    }

    // wipe imaginary part
    for(unsigned i=0; i<grid_n*4; i++) {
        cinput[i][1] = 0.0;
    }

    // copy first part of grid
    for(unsigned i=0; i<grid_n+1; i++) {
        cinput[i][0] = float(grid[i]);
    }

    // append zeros
    for(unsigned i=grid_n+1; i<=grid_n*3; i++) {
        cinput[i][0] = 0.0;
    }

    // now append the grid in reverse order
    for(unsigned i=grid_n-1, index=0; i >=1; i--, index++) {
        cinput[grid_n*3+1 + index][0] = float(grid[i]);
    }

    fftwf_plan plan = fftwf_plan_dft_1d(grid_n*4, cinput, coutput,
                      FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);

    unsigned index = 0;
    for(unsigned i=4*grid_n-n; i<4*grid_n; i+=2) {
        result[index] = coutput[i][0];
        index++;
    }

    for(unsigned i=1; i<=n; i+=2) {
        result[index] = coutput[i][0];
        index++;
    }

    fftwf_destroy_plan(plan);
    fftwf_free(cinput);
    fftwf_free(coutput);

    // multiply with window
    for(unsigned i=0; i<=n; i++) {
        result[i] *= window[i];
    }

    // normalize
    double factor = result[n/2];
    for(unsigned i=0; i<=n; i++) {
        result[i] /= factor;
    }

    free(grid);
}


/**
 * Compute the modified Bessel function I_0(x) for any real x.
 * Taken from the LOFAR CNProc package under GNU GPL (TODO: check this)
 *
 * @param x
 */
double besselI0(double x)
{
    // Parameters of the polynomial approximation.
    const double p1 = 1.0, p2 = 3.5156229, p3 = 3.0899424, p4 = 1.2067492,
            p5 = 0.2659732, p6 = 3.60768e-2,  p7 = 4.5813e-3;

    const double q1 = 0.39894228, q2 = 1.328592e-2, q3 = 2.25319e-3,
            q4 = -1.57565e-3, q5 = 9.16281e-3, q6 = -2.057706e-2,
            q7 = 2.635537e-2, q8 = -1.647633e-2, q9 = 3.92377e-3;

    const double k1 = 3.75;

    double ax = abs(x);
    double y = 0, result = 0;

    if (ax < k1) {
        double xx = x / k1;
        y = xx * xx;
        result = p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7)))));
    }
    else {
        y = k1 / ax;
        result = (exp(ax)/sqrt(ax))*
                (q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*(q7+y*(q8+y*q9))))))));
    }
    return result;
}


// Kaiser window function
void kaiser(int n, double beta, double* d)
{
    if (n == 1) {
        d[0] = 1.0;
        return;
    }

    int m = n - 1;

    for (int i = 0; i < n; i++) {
        double k = 2.0 * beta / m * sqrt(double(i * (m - i)));
        d[i] = besselI0(k) / besselI0(beta);
    }
}

// Guassian window function
void gaussian(int n, double a, double* d)
{
    int index = 0;
    for (int i=-(n-1); i<=n-1; i+=2) {
        d[index++] = exp( -0.5 * pow(( a/n * i), 2) );
    }
}

// Hamming window function
void hamming(unsigned n, double* d)
{
    if (n == 1) {
        d[0] = 1.0;
        return;
    }
    unsigned m = n - 1;
    for(unsigned i = 0; i < n; i++) {
        d[i] = 0.54 - 0.46 * cos((2.0 * M_PI * i) / m);
    }
}

// Blackman window function
void blackman(unsigned n, double* d)
{
    if(n == 1) {
        d[0] = 1.0;
        return;
    }
    unsigned m = n - 1;
    for(unsigned i = 0; i < n; i++) {
        double k = i / m;
        d[i] = 0.42 - 0.5 * cos(2.0 * M_PI * k) + 0.08 * cos(4.0 * M_PI * k);
    }
}

// Generate polyphase coefficients for channeliser
float *generatePolyphaseCoefficients(unsigned nchans, unsigned ntaps, FirWindow windowType)
{
    unsigned n = nchans * ntaps;

    double* window = new double[n];

    switch (windowType) {
        case HAMMING:
        {
            hamming(n, window);
            break;
        }
        case BLACKMAN:
        {
            blackman(n, window);
            break;
        }
        case GAUSSIAN:
        {
            double alpha = 3.5;
            gaussian(n, alpha, window);
            break;
        }
        case KAISER:
        {
            double beta = 9.0695;
            kaiser(n, beta, window);
            break;
        }
        default:
        {
            fprintf(stderr, "Invalid PFB window type. Exiting\n");
            exit(-1);
        }
    }

    double* result = new double[n];
    generateFirFilter(n - 1, 1.0 / nchans, window, result);

    delete[] window;
    
    float *coeff = (float *) malloc(nchans * ntaps * sizeof(float));

    for(unsigned t = 0; t < ntaps; ++t) {
        for(unsigned c = 0; c < nchans; ++c) {
            unsigned index = t * nchans + c;
            coeff[index] = result[index] / nchans;
            if (c % 2 == 0)
                coeff[index] = result[index] / nchans;
            else
                coeff[index] = -result[index] / nchans;
        }
    }

    delete[] result;

    return coeff;

}

int main()
{
    float *coeffs = generatePolyphaseCoefficients(64, 8, KAISER);
    for(unsigned i = 0; i < 8; i++)
    {
        printf("[");
        for(unsigned j = 0; j < 64; j++)
            printf("%f,", coeffs[i * 64 + j]);
        printf("]\n");
    }
}

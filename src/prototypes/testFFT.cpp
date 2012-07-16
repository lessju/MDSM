extern "C"
{
    #include "ransomfft.h"
}

#include "fftw3.h"
#include "stdlib.h"
#include "unistd.h"
#include "time.h"
#include "string.h"
#include "sys/time.h"

int main()
{
	int fftlen = 8192;

	// Create buffers
	float         fftw_input[fftlen];
	fftwf_complex fftw_output[fftlen];

	float         presto_input[fftlen];
	fcomplex      presto_output[fftlen];

	srand ( time(NULL) );
	for(unsigned i = 0; i < fftlen; i++)
		fftw_input[i] = presto_input[i] = rand() / (float) RAND_MAX;

	struct timeval start, end;
	long mtime, seconds, useconds;

	printf("Size of fftwf_complex: %ld\n", sizeof(fftwf_complex));

	// Execute FFTW routine (+ timing)
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(fftlen, fftw_input, fftw_output, FFTW_ESTIMATE);

	gettimeofday(&start, NULL);
//	for(unsigned i = 0; i < 10000; i++)
		fftwf_execute(plan);	
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	printf("FFTW in %ld ms\n", mtime);

	// Test inverse FFT
	float fftw_test[fftlen];
	fftwf_plan invPlan = fftwf_plan_dft_c2r_1d(fftlen, fftw_output, fftw_test, FFTW_ESTIMATE);

	fftwf_execute(invPlan);

	for(unsigned i = 0; i < fftlen; i++)
		if (abs((fftw_test[i] / fftlen) - fftw_input[i]) > 0.0001)
		{
			printf("IFFT mismatch at %d: input(%f) != test(%f)\n",
			i, fftw_input[i], fftw_test[i] / fftlen );
			exit(0);
		}

	fftwf_destroy_plan(plan);
	fftwf_destroy_plan(invPlan);

	// Execute Presto routine (+ timing)

	gettimeofday(&start, NULL);
	for(unsigned i = 0; i < 1; i++)
		realfft(presto_input, fftlen, -1);
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	printf("PRESTO in %ld ms\n", mtime);

	for(unsigned i = 0; i < fftlen / 2; i++)
	{
		presto_output[i].r = presto_input[i * 2];
		presto_output[i].i = presto_input[i * 2 + 1];
	}

	// Compare FFTs
	for(unsigned i = 0; i < fftlen / 2; i++)
		if (fabs(fftw_output[i][0] - presto_output[i].r) > 0.00001 && 
			fabs(fftw_output[i][1] - presto_output[i].i) > 0.00001)

		printf("Mismatch at %d: fftw(%f i%f) != presto(%f i%f)\n",
			i, fftw_output[i][0], fftw_output[i][1],
			   presto_output[i].r, presto_output[i].i);

	// Compare IFFTs
	for(unsigned i = 0; i < fftlen / 2; i++)
	{
		presto_input[i*2] = presto_output[i].r;
		presto_input[i*2+1] = presto_output[i].i;
	}

	realfft(presto_input, fftlen, 1);

	for(unsigned i = 0; i < fftlen; i++)
	{
	//	if (fabs((fftw_test[i] / fftlen) - presto_input[i]) > 0.00001)
	//	{
			printf("IFFT mismatch at %d: FFTW(%f) != presto(%f)\n",
			i, fftw_test[i] / fftlen, presto_input[i]);
	//		exit(0);
	//	}
	}
	
}

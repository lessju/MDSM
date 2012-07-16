#include "stdlib.h"
#include "unistd.h"
#include "time.h"
#include "string.h"
#include "sys/time.h"
#include "math.h"

#include <vector>
#include <utility>
#include <set>

#include <omp.h>

#include <fftw3.h>

//unsigned nsamp = 3789000, ndms = 1, threads = 1;
unsigned nsamp = 65536*2, ndms = 256, threads = 4;
float threshold = 5.0, tsamp = 4.096e-5;

unsigned downfactors[] = { 2, 3, 4, 6, 9, 14, 20, 30, 45, 70, 100, 150 };
unsigned fftlen = 8192;
unsigned chunklen = 8000;
unsigned maxDownfact = 30;

typedef struct candidate
{
	float dm;
	float value;
	double time;
	long bin;
	unsigned downfact;
} Candidate;

typedef struct pair
{
    unsigned bin;
    float    value;
} InitialCandidate;

// Create FFT kernels for template matching
void createFFTKernel(fftwf_complex *kernel, unsigned factor, unsigned fftlen)
{
	unsigned i;

	float values[fftlen];

	memset(values, 0, fftlen * sizeof(fftlen));

	for (i = 0; i < factor / 2 + 1	; i++)
		values[i] += 1;

	if (factor % 2) // Downfactor is odd
		for (i = fftlen - factor / 2; i < fftlen; i++)
			values[i] += 1;

	else if (factor > 2) // Downfactor is even
		for (i = fftlen - (factor / 2 - 1); i < fftlen; i++)
			values[i] += 1;

	for (i = 0; i < fftlen; i++)
		values[i] /= sqrt(factor);

	// Create FFT plan and execute
	fftwf_plan plan = fftwf_plan_dft_r2c_1d(fftlen, values, kernel, FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
}

void convolve(fftwf_complex *chunk, fftwf_complex* kernel, fftwf_complex *convChunk, float *result, unsigned fftlen, fftwf_plan plan)
{
    convChunk[0][0] = kernel[0][0] * chunk[0][0];
    convChunk[0][1] = kernel[0][1] * chunk[0][1];

    for(unsigned i = 1; i < fftlen / 2; i++)
	{
		convChunk[i][0] = (chunk[i][0] * kernel[i][0] - chunk[i][1] * kernel[i][1]);
		convChunk[i][1] = (chunk[i][1] * kernel[i][0] + chunk[i][0] * kernel[i][1]);
	}

	// Inverse FFT
    fftwf_execute(plan);

	// Normalise results
	float scale = 1.0f / fftlen;
    for(unsigned i = 0; i < fftlen; i++)
        result[i] *= scale;
}

// Prune related candidate with the same downfactor value
void pruneRelated(InitialCandidate *candidates, unsigned factor, unsigned num)
{
    char mask[num];

    memset(mask, 0, num * sizeof(char));

    for(unsigned i = 0; i < num - 1; i++)
    {
        if (mask[i]) continue;

        InitialCandidate x = candidates[i];
        for (unsigned j = i + 1; j < num; j++)
        {
            InitialCandidate y = candidates[j];
            if (abs(y.bin - x.bin > factor / 2)) break;

            if (mask[j] ) continue;

            if (x.value > y.value) mask[j] = 1;
            else mask[j] = i;
        }
    }

	// Clear from initial candidates OPTIMISE!!
	for(unsigned i = 0; i < num; i++)
		if (mask[i])
        {
            candidates[i].bin = 0;
            candidates[i].value = 0;
        }
}

void pruneRelatedDownfactors(std::vector<Candidate> dmCandidates, char *mask, unsigned numDownFacts)
{
    for(unsigned i = 0; i < dmCandidates.size(); i++)
    {
        if (mask[i]) continue;

        Candidate x = dmCandidates[i];
        for(unsigned j = i + 1; j < dmCandidates.size(); j++)
        {
            Candidate y = dmCandidates[j];
            if (abs(y.bin - x.bin) > downfactors[numDownFacts - 1] / 2) break;

            if (mask[j]) continue;

            unsigned prox = std::max(x.downfact/2, y.downfact/2);
            prox = (prox > 1) ? prox : 1;

            if (abs(y.bin - x.bin) <= prox)
            {
                if (x.value > y.value) mask[j] = 1;
                else mask[i] = 1;
            }
        }
    }
}

int main()
{
	unsigned overlap = (fftlen - chunklen) / 2;

	// Create data
	float *input = (float *) malloc(nsamp * ndms * sizeof(float));

    // LOAD FILE: FOR TESTING ONLY
//	FILE *fp = fopen("/home/lessju/Code/MDSM/src/prototypes/TestingCCode.dat", "rb");
//	printf("Read: %ld\n", fread(input, sizeof(float), nsamp, fp));
//	fclose(fp);

	// Initialise templating
	unsigned numDownFacts;
	for(numDownFacts = 0; numDownFacts < 12; numDownFacts++)
		if (downfactors[numDownFacts] > maxDownfact)
			break;

	// Allocate kernels
	fftwf_complex **kernels = (fftwf_complex **) malloc(numDownFacts * sizeof(fftwf_complex *));
	for(unsigned i = 0; i < numDownFacts; i++)
		kernels[i] = (fftwf_complex *) fftwf_malloc(fftlen / 2 * sizeof(fftwf_complex));

	// Create kernels
	for(unsigned i = 0; i < numDownFacts; i++)
		createFFTKernel(kernels[i], downfactors[i], fftlen);

	// Start timing
	struct timeval start, end;
	long mtime, seconds, useconds;
	gettimeofday(&start, NULL);

	// Set number of OpenMP threads
	omp_set_num_threads(threads);

	// Create candidate container
	std::vector<Candidate> **candidates = (std::vector<Candidate> **) malloc(threads * sizeof(std::vector<Candidate> *));

	unsigned nchunks = nsamp / chunklen;

	#pragma omp parallel \
		shared(kernels, input, ndms, nsamp, fftlen, chunklen, numDownFacts, tsamp, \
			   overlap, downfactors, threshold, nchunks, candidates)
	{
		// Get thread details
		unsigned numThreads = omp_get_num_threads();
		unsigned threadId = omp_get_thread_num();

		// Allocate memory to be used in processing
		candidates[threadId] = new std::vector<Candidate>();

        float *chunk = (float *) fftwf_malloc(fftlen * sizeof(float)); // Store input chunk
		fftwf_complex *fftChunk = (fftwf_complex *)
                fftwf_malloc(fftlen / 2 * sizeof(fftwf_complex));      // Store FFT'ed input chunk
        fftwf_complex *convolvedChunk = (fftwf_complex *)
                fftwf_malloc(fftlen / 2 * sizeof(fftwf_complex));      // Store FFT'ed, convolved input chunk
        InitialCandidate *initialCands = (InitialCandidate *)
                malloc(fftlen * sizeof(InitialCandidate));             // Store initial Candidate list

		// Create FFTW plans (these calls are note thread safe, place in critical section)
		fftwf_plan chunkPlan, convPlan;
		#pragma omp critical
		{
			chunkPlan = fftwf_plan_dft_r2c_1d(fftlen, chunk, fftChunk, FFTW_ESTIMATE);
		    convPlan  = fftwf_plan_dft_c2r_1d(fftlen, convolvedChunk, chunk, FFTW_ESTIMATE) ;
		}

		// Process all DM buffer associated with this thread
		for(unsigned j = 0; j < ndms / numThreads; j++)
		{
			unsigned d = ndms / numThreads * threadId + j;

			std::vector<Candidate> dmCandidates;

			// Process all data chunks
			for (unsigned c = 0; c < nchunks; c++)
			{
				int beg = d * nsamp + c * chunklen - overlap;
				if (c == 0)                // First chunk, we need to insert 0s at the beginning
				{
					memset(chunk, 0, overlap * sizeof(float));
                    memcpy(chunk + overlap, input, (fftlen - overlap) * sizeof(float));
				}
				else if (c == nchunks - 1) // Last chunk, insert 0s at the end
				{
					memset(chunk + fftlen - overlap, 0, overlap * sizeof(float));
					memcpy(chunk, input + beg, (fftlen - overlap) * sizeof(float));
				}
				else
					memcpy(chunk, input + beg, fftlen * sizeof(float));

				// Search non-downsampled data first
				for(unsigned i = overlap; i < chunklen; i++)
					if (chunk[i] >= threshold)
					{
						candidate newCand = { d, chunk[i], 25, c*chunklen+i, 1 };
						dmCandidates.push_back(newCand);
					}

				// FFT current chunk
				fftwf_execute(chunkPlan);

				// Loop over all downfactor levels
				for(unsigned s = 0; s < numDownFacts; s++)
				{
                    // Reset inital Candidate List
                    memset(initialCands, 0, fftlen * sizeof(InitialCandidate));

					// Perform convolution
					convolve(fftChunk, kernels[s], convolvedChunk, chunk, fftlen, convPlan);

					// Threshold results and build preliminary candidate list
                    unsigned numCands = 0;
					for(unsigned i = overlap; i < chunklen; i++)
					{
						if (chunk[i] >= threshold)
                        {
						//	printf("We have something %d %d \n", c, s);
						    initialCands[numCands].bin = i;
						    initialCands[numCands].value = chunk[i];
                            numCands++;
                        }
					}

                    if (numCands != 0)
                    {
                        // Prune candidate list
                        pruneRelated(initialCands, downfactors[s], numCands);

                        // Store candidate list
                        for(unsigned k = 0; k < numCands; k++)
							if (initialCands[k].value != 0)
							{
                                Candidate newCand = { d, initialCands[j].value, 5, c * chunklen + k, downfactors[s] };
                                dmCandidates.push_back(newCand);
                            }
                    }
				}
			}

			// Remove redundate candidates across downsampling levels
			if (dmCandidates.size() > 0)
			{
				char *mask = (char *) malloc(dmCandidates.size() * sizeof(char));
		        pruneRelatedDownfactors(dmCandidates, mask, numDownFacts);

	            // Append to final candidate list
	            for(j = 0; j < dmCandidates.size(); j++)
	                if (mask[j])
	                    candidates[threadId] -> push_back(dmCandidates[j]);

				free(mask);
			}
		}

		free(convolvedChunk);
		free(fftChunk);
		free(chunk);
	}

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	printf("Processed everything in %ld ms\n", mtime);

	// Now write everything to disk...
	FILE *fp2 = fopen("output.dat", "w");
	for(unsigned i = 0; i < threads; i++)
		for(unsigned j = 0; j < candidates[i] -> size(); j++)
		{
			Candidate cand = candidates[i] -> at(j);
			fprintf(fp2, "%f,%f,%f,%ld,%d\n", cand.dm, cand.value, cand.time, cand.bin, cand.downfact);
		}
	fflush(fp2);
	fclose(fp2);
}


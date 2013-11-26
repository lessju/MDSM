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

extern "C"
{
    #include "ransomfft.h"
}

//unsigned nsamp = 3789000, ndms = 1, threads = 1;
unsigned nsamp = 65536, ndms = 256, threads = 1;
float threshold = 5.0, tsamp = 4.096e-5;

unsigned downfactors[] = { 2, 3, 4, 6, 9, 14, 20, 30, 45, 70, 100, 150 } ;
unsigned fftlen = 8192;
unsigned chunklen = 8000;
unsigned maxDownfact = 30;

typedef struct candidate
{
	float dm;
	float value;
	double time;
	double bin;
	unsigned downfact;
} Candidate;

typedef struct pair
{
    unsigned bin;
    float    value;
} InitialCandidate;

// Perform a real FFT (using presto algorithms)
void realFFT(float *data, fcomplex *result, unsigned fftlen)
{
	unsigned i;

	// Perform kernel
	realfft(data, fftlen, -1);

	// Convert from real to complex
	for(i = 0; i < fftlen / 2; i++)
	{
		result[i].r = data[i * 2];
		result[i].i = data[i * 2 + 1];
	}
}

// Create FFT kernels for template matching
void createFFTKernel(fcomplex *kernel, unsigned factor, unsigned fftlen)
{
	unsigned i;

	float values[fftlen];

	for(i = 0; i < fftlen; i++)
        values[i] = 0;

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

	// FFT kernel
	realfft(values, fftlen, -1);

	// Convert from real to complex
	for(i = 0; i < fftlen / 2; i++)
	{
		kernel[i].r = values[i * 2];
		kernel[i].i = values[i * 2 + 1];
	}
}

// Convolve data buffer with template kernel
inline static void convolve(fcomplex *chunk, fcomplex* kernel, float *result, unsigned fftlen)
{
//	float *temp = result;

//	// Convolve (complex multiply) this way of writing the loop seems to improve performance a bit
//    // Loop unrolled (2)
//	for ( ; result != temp + fftlen; chunk += 2, kernel += 2, result++)
//	{
//		*result     = chunk -> r * kernel -> r - chunk -> i * kernel -> i;
//		*(++result) = chunk -> i * kernel -> r + chunk -> r * kernel -> i;
//		chunk++; kernel++;
//		*(++result) = chunk -> r * kernel -> r - chunk -> i * kernel -> i;
//		*(++result) = chunk -> i * kernel -> r + chunk -> r * kernel -> i;
//	}

//	// Reset result pointer
//	result -= fftlen;

    unsigned int i = 0;
	for(i = 1; i < fftlen / 2; i++)
	{
		result[i * 2]     = chunk[i].r * kernel[i].r - chunk[i].i * kernel[i].i;
		result[i * 2 + 1] = chunk[i].i * kernel[i].r + chunk[i].r * kernel[i].i;
	}

	result[0] = chunk -> r * kernel -> r;
	result[1] = chunk -> i * kernel -> i;

	// Inverse FFT
	realfft(result, fftlen, 1);
}

// Prune related candidate with the same downfactor value
void pruneRelatedFast(InitialCandidate *candidates, unsigned factor, unsigned num)
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
}

// Prune related candidate with the same downfactor value
std::set<unsigned> pruneRelated(std::vector<std::pair<unsigned, float> > candidates, unsigned factor)
{
    std::set<unsigned> toRemove;
    for(unsigned i = 0; i < candidates.size() - 1; i++)
    {
        if (toRemove.find(i) != toRemove.end()) continue;

        std::pair<unsigned, float> x = candidates[i];
        for (unsigned j = i + 1; j < candidates.size(); j++)
        {
            std::pair<unsigned, float> y = candidates[j];
            if (abs(y.first - x.first > factor / 2)) break;

            if (toRemove.find(j) != toRemove.end()) continue;

            if (x.second > y.second) toRemove.insert(j);
            else toRemove.insert(i);
        }
    }

    return toRemove;
}

// Prune related candidate across downfactor levels
std::set<unsigned> pruneRelatedDownfactors(std::vector<Candidate> dmCandidates, unsigned numDownFacts)
{
    std::set<unsigned> toRemove;
    for(unsigned i = 0; i < dmCandidates.size(); i++)
    {
        if (toRemove.find(i) != toRemove.end()) continue;

        Candidate x = dmCandidates[i];
        for(unsigned j = i + 1; j < dmCandidates.size(); j++)
        {
            Candidate y = dmCandidates[j];
            if (abs(y.bin - x.bin) > downfactors[numDownFacts - 1] / 2) break;

            if (toRemove.find(j) != toRemove.end()) continue;

            unsigned prox = 1;//max(x.downfact/2, y.downfact/2);
            prox = (prox > 1) ? prox : 1;

            if (abs(y.bin - x.bin) <= prox)
            {
                if (x.value > y.value) toRemove.insert(j);
                else toRemove.insert(i);
            }
        }
    }

    return toRemove;
}

int main()
{
	unsigned overlap = (fftlen - chunklen) / 2;
	unsigned i, j, c, s;

	// Create data
	float *input = (float *) malloc(nsamp * ndms * sizeof(float));

    // LOAD FILE: FOR TESTING ONLY
	FILE *fp = fopen("/home/lessju/Code/MDSM/src/prototypes/TestingCCode.dat", "rb");
	printf("Read: %ld\n", fread(input, sizeof(float), nsamp, fp));
	fclose(fp);

	// Initialise templating
	unsigned numDownFacts;
	for(numDownFacts = 0; numDownFacts < 12; numDownFacts++)
		if (downfactors[numDownFacts] > maxDownfact)
			break;

	// Allocate kernels
	fcomplex **kernels = (fcomplex **) malloc(numDownFacts * sizeof(fcomplex *));
	for(i = 0; i < numDownFacts; i++)
		kernels[i] = (fcomplex *) malloc(fftlen / 2 * sizeof(fcomplex));

	// Create kernels
	for(i = 0; i < numDownFacts; i++)
		createFFTKernel(kernels[i], downfactors[i], fftlen);

	// Start timing
	struct timeval start, end;
	long mtime, seconds, useconds;
	gettimeofday(&start, NULL);

	// Start processing the input buffer ...
	unsigned nchunks = nsamp / chunklen;

	// Set number of OpenMP threads
	omp_set_num_threads(threads);

	// Create candidate container
	std::vector<Candidate> **candidates = (std::vector<Candidate> **) malloc(threads * sizeof(std::vector<Candidate> *));

	#pragma omp parallel \
		shared(kernels, input, ndms, nsamp, fftlen, chunklen, numDownFacts, tsamp, \
			   overlap, downfactors, threshold, nchunks, candidates) \
		private(i, j, c, s)
	{
		// Get thread details
		unsigned numThreads = omp_get_num_threads();
		unsigned threadId = omp_get_thread_num();

		// Allocate memory to be used in processing
		candidates[threadId] = new std::vector<Candidate>();

        float *chunk = (float *) malloc(fftlen * sizeof(float));                 // Store input chunk
		fcomplex *fftChunk = (fcomplex *) malloc(fftlen / 2 * sizeof(fcomplex)); // Store FFT'ed input chunk
        InitialCandidate *initialCands = (InitialCandidate *)                    // Store initial Candidate list
                malloc(fftlen * sizeof(InitialCandidate));

		// Process all DM buffer associated with this thread
		for(j = 0; j < ndms / numThreads; j++)
		{
			unsigned d = ndms / numThreads * threadId + j;

			std::vector<Candidate> dmCandidates;

			// Process all data chunks
			for (c = 0; c < nchunks; c++)
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
				for(i = overlap; i < chunklen; i++)
					if (chunk[i] >= threshold)
					{
						candidate newCand = { d, chunk[i], 25, c*chunklen+i, 1 };
						dmCandidates.push_back(newCand);
					}

				// FFT current chunk
				realFFT(chunk, fftChunk, fftlen);

				// Loop over all downfactor levels
				for(s = 0; s < numDownFacts; s++)
				{
                    // Reset inital Candidate List
                    memset(initialCands, 0, fftlen * sizeof(InitialCandidate));

					// Perform convolution
					convolve(fftChunk, kernels[s], chunk, fftlen);

					// Threshold results and build preliminary candidate list
                    unsigned numCands = 0;
					for(i = overlap; i < chunklen; i++)
					{
						if (chunk[i] >= threshold)
                        {
						//	printf("We have something %d %d \n", c, s);
							InitialCandidate x = {i, chunk[i] };	
						    initialCands[i] = x;
                            numCands++;
                        }
					}

                    if (numCands != 0)
                    {
                        // Prune candidate list
//                        pruneRelatedFast(initialCands, downfactors[s], numCands);

//                        // Store candidate list
//                        for(j = 0; j < cands.size(); j++)
//                            if (redundant.find(j) == redundant.end())
//                            {
//                                pair<unsigned, float> val = cands[j];
//                                candidate newCand = { d, val.second, 5, c*chunklen+i, downfactors[s] };
//                                dmCandidates.push_back(newCand);
//                            }
                    }
				}

				// Free up FFT chunk

			}

			// Remove redundate candidates across downsampling levels
//            set<unsigned> redundant = pruneRelatedDownfactors(dmCandidates, numDownFacts);

//            // Append to final candidate list
//            for(j = 0; j < dmCandidates.size(); j++)
//                if (redundant.find(j) == redundant.end())
//                    candidates[threadId] -> push_back(dmCandidates[j]);

//            printf("Found %ld %ld\n", candidates[threadId] -> size(), redundant.size());
		}

		free(fftChunk);
		free(chunk);
	}

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	printf("Processed everything in %ld ms\n", mtime);

	// Now write everything to disk...
}


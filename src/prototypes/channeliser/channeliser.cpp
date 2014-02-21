#include "PPFChanneliser.h"

#include "pelican/utility/ConfigNode.h"

#include "TimeSeriesDataSet.h"
#include "SpectrumDataSet.h"

#include <QtCore/QString>
#include <QtCore/QTime>
#include <QtCore/QFile>

#include <omp.h>

#include <cfloat>
#include <cstring>
#include <iostream>
#include <iostream>

/**
 * @details
 * Constructor.
 *
 * @param[in] config XML configuration node.
 */
PPFChanneliser::PPFChanneliser(const ConfigNode& config)
: AbstractModule(config), _buffersInitialised(false)
{
    // Get options from the XML configuration node.
    _nChannels = config.getOption("outputChannelsPerSubband", "value", "512").toUInt();
    _nThreads  = config.getOption("processingThreads", "value", "2").toUInt();
    unsigned nTaps = config.getOption("filter", "nTaps", "8").toUInt();
    QString window = config.getOption("filter", "filterWindow", "kaiser").toLower();

    // Set the number of processing threads.
    omp_set_num_threads(_nThreads);
    _iOldestSamples.resize(_nThreads, 0);

    // Enforce even number of channels.
    if (_nChannels != 1 && _nChannels%2 == 1)
       throw _err("Number of channels needs to be even.");

    // Generate the FIR coefficients;
    _generateFIRCoefficients(window, nTaps);

    // Allocate buffers used for holding the output of the FIR stage.
    _filteredData.resize(_nThreads);
    for (unsigned i = 0; i < _nThreads; ++i)
        _filteredData[i].resize(_nChannels);

    // Create the FFTW plan.
    _createFFTWPlan(_nChannels, _fftPlan);
}

/**
* @details
* Destroys the channeliser module.
*/
PPFChanneliser::~PPFChanneliser()
{
    fftwf_destroy_plan(_fftPlan);
}


/**
* @details
* Method to run the channeliser.
*
* The channeliser performs channelisation of a number of sub-bands containing
* a complex time series.
*
* Parallelisation, by means of openMP threads, is carried out by splitting
* the sub-bands as evenly as possible between threads.
*
* @param[in]  timeSeries 	Buffer of time samples to be channelised.
* @param[out] spectrum	 	Set of spectra produced.
*/
void PPFChanneliser::run(const TimeSeriesDataSetC32* timeSeries,
        SpectrumDataSetC32* spectra)
{
    // Perform a number of sanity checks on the input data.
    _checkData(timeSeries);

    // Make local copies of the data dimensions.
    unsigned nSubbands      = timeSeries->nSubbands();
    unsigned nPolarisations = timeSeries->nPolarisations();
    unsigned nTimeBlocks    = timeSeries->nTimeBlocks();
    unsigned nTimesPerBlock = timeSeries->nTimesPerBlock();

    // Resize the output spectra blob (if required).
    spectra->resize(nTimeBlocks, nSubbands, nPolarisations, _nChannels);

    // Set the timing parameters - Only need the timestamp of the first packet
    // for this version of the Channeliser.
    spectra->setLofarTimestamp(timeSeries->getLofarTimestamp());
    spectra->setBlockRate(timeSeries->getBlockRate() * _nChannels);

    const float* coeffs = &_coeffs[0];
    unsigned threadId = 0, nThreads = 0, start = 0, end = 0;
    Complex *workBuffer = 0, *filteredSamples = 0;
    Complex const * timeData = 0;
    const Complex* timeStart = timeSeries->constData();
    Complex* spectraStart = spectra->data();

    // Set up work buffers (if required).
    unsigned nFilterTaps = _ppfCoeffs.nTaps();
    if (!_buffersInitialised)
        _setupWorkBuffers(nSubbands, nPolarisations, _nChannels, nFilterTaps);

    // Channeliser processing.
    #pragma omp parallel \
        shared(nTimeBlocks, nPolarisations, nSubbands, nFilterTaps, coeffs,\
                timeStart, spectraStart) \
        private(threadId, nThreads, start, end, workBuffer, filteredSamples, \
                timeData)
    {
        threadId = omp_get_thread_num();
        nThreads = omp_get_num_threads();

        // Assign processing threads in a round robin fashion to subbands.
        _assign_threads(start, end, nSubbands, nThreads, threadId);

        // Pointer to work buffer for the thread.
        filteredSamples = &_filteredData[threadId][0];

        // Loop over data to be channelised.
        for (unsigned subband = start; subband < end; ++subband)
        {
            for (unsigned pol = 0; pol < nPolarisations; ++pol)
            {
                for (unsigned block = 0; block < nTimeBlocks; ++block)
                {
                    // Get pointer to time series array.
                    unsigned index = timeSeries->index(subband, nTimesPerBlock,
                                 pol, nPolarisations, block, nTimeBlocks);
                    timeData = &timeStart[index];

                    // Get a pointer to the work buffer.
                    workBuffer = &(_workBuffer[subband * nPolarisations + pol])[0];

                    // Update buffered (lagged) data for the sub-band.
                    _updateBuffer(timeData, _nChannels, nFilterTaps, workBuffer);

                    // Apply the PPF.
                    _filter(workBuffer, nFilterTaps, _nChannels, coeffs, filteredSamples);

                    // FFT the filtered sub-band data to form a new spectrum.
                    unsigned indexSpectra = spectra->index(subband, nSubbands,
                            pol, nPolarisations, block, _nChannels);
                    _fft(filteredSamples, &spectraStart[indexSpectra]);
                }
            }
        }

    } // end of parallel region.

}


/**
 * @details
 * Generate FIR coefficients for the specified window.
 */
void PPFChanneliser::_generateFIRCoefficients(const QString& window, unsigned nTaps)
{
    _ppfCoeffs.resize(nTaps, _nChannels);

    PolyphaseCoefficients::FirWindow windowType;
    if (window == "kaiser")
        windowType = PolyphaseCoefficients::KAISER;
    else if (window == "gaussian")
        windowType = PolyphaseCoefficients::GAUSSIAN;
    else if (window == "blackman")
        windowType = PolyphaseCoefficients::BLACKMAN;
    else if (window == "hamming")
        windowType = PolyphaseCoefficients::HAMMING;
    else
        throw _err("Unknown coefficient window type '%1'.").arg(window);

    _ppfCoeffs.genereateFilter(nTaps, _nChannels, windowType);

    // Convert Coefficients to single precision.
    _coeffs.resize(_ppfCoeffs.size());
    double const* coeffs = _ppfCoeffs.ptr();
    for (unsigned i = 0u; i < _ppfCoeffs.size(); ++i)
        _coeffs[i] = (float)coeffs[i];
}


/**
* @details
*/
void PPFChanneliser::_checkData(const TimeSeriesDataSetC32* timeData)
{
    if (!timeData) throw _err("Time stream data blob missing.");

    if (!timeData->size()) throw _err("Empty time data blob");
    if (!timeData->nSubbands()) throw _err("Empty time data blob");
    if (!timeData->nPolarisations()) throw _err("Empty time data blob");
    if (!timeData->nTimeBlocks()) throw _err("Empty time data blob");

    if (_ppfCoeffs.nChannels() != _nChannels)
        throw _err("Dimension mismatch: Number of FIR coefficient channels %1 "
                "!= number of output channels %2.")
                .arg(_ppfCoeffs.nChannels()).arg(_nChannels);
}


/**
* @details
* Prepend nSamples complex data into the start of the buffer moving along
* other data.
*
* @param samples
* @param nSamples
*/
void PPFChanneliser::_updateBuffer(const Complex* samples, unsigned nSamples,
        unsigned nTaps, Complex* buffer)
{
    unsigned tId = omp_get_thread_num();
    size_t blockSize = nSamples * sizeof(Complex);
    memcpy(&buffer[_iOldestSamples[tId] * nSamples], samples, blockSize);
    _iOldestSamples[tId] = (_iOldestSamples[tId] + 1) % nTaps;
}


/**
 * @details
 * Filter a buffer of time samples.
 *
 * @param samples
 * @param nTaps
 * @param nChannels
 * @param filteredSamples
 */
void PPFChanneliser::_filter(const Complex* sampleBuffer, unsigned nTaps,
        unsigned nChannels, const float* coeffs, Complex* filteredSamples)
{
    for (unsigned i = 0; i < nChannels; ++i)
        filteredSamples[i] = Complex(0.0, 0.0);

    unsigned iBuffer = 0, idx = 0;
    float re, im, coeff;
    unsigned tId = omp_get_thread_num();

    for (unsigned i = 0, t = 0; t < nTaps; ++t)
    {
        iBuffer = ((_iOldestSamples[tId] + t) % nTaps) * nChannels;
        for (unsigned c = 0; c < nChannels; ++c)
        {
            idx = iBuffer + c;
            coeff = coeffs[i];
            re = sampleBuffer[idx].real() * coeff;
            im = sampleBuffer[idx].imag() * coeff;
            filteredSamples[c].real() += re;
            filteredSamples[c].imag() += im;
            i++;
        }
    }
}


/**
 * @details
 * Calculates start and end indices for a given thread for splitting nValues
 * of a processing dimension between nThreads.
 */
void PPFChanneliser::_assign_threads(unsigned& start,
        unsigned& end, unsigned nValues, unsigned nThreads,
        unsigned threadId)
{
    if (threadId >= nThreads)
        throw _err("threadProcessingIndices(): threadId '%1' out of range for"
                " nThreads = %2.").arg(threadId).arg(nThreads);

    if (threadId >= nValues)
    {
        start = end = 0;
        return;
    }

    unsigned number = nValues / nThreads;
    unsigned remainder = nValues % nThreads;
    if (threadId < remainder) number++;

    start = threadId * number;
    if (threadId >= remainder) start += remainder;
    end = start + number;
}


/**
* @details
* Set up buffers used to store the last nTaps * nChannels time series values
* for each sub-band and polarisation.
*/
unsigned PPFChanneliser::_setupWorkBuffers(unsigned nSubbands,
        unsigned nPolarisations, unsigned nChannels, unsigned nTaps)
{
    unsigned bufferSize = nChannels * nTaps;
    _workBuffer.resize(nSubbands * nPolarisations);
    for (unsigned i = 0, s = 0; s < nSubbands; ++s)
    {
        for (unsigned p = 0; p < nPolarisations; ++p)
        {
            _workBuffer[i].resize(bufferSize, Complex(0.0, 0.0));
            i++;
        }
    }
    _buffersInitialised = true;
    return bufferSize;
}



void PPFChanneliser::_createFFTWPlan(unsigned nChannels, fftwf_plan& plan)
{
    size_t fftSize = nChannels * sizeof(fftwf_complex);
    fftwf_complex* in  = (fftwf_complex*) fftwf_malloc(fftSize);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(fftSize);
    plan = fftwf_plan_dft_1d(_nChannels, in, out, FFTW_FORWARD, FFTW_MEASURE);
    fftwf_free(in);
    fftwf_free(out);
}


/**
 * @details
 * Returns a message use for errors and throws from the channeliser.
 */
inline QString PPFChanneliser::_err(const QString& message)
{
    return QString("PPFChanneliser: ") + message;
}


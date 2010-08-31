// MDSM stuff
#include "MdsmModule.h"
#include "DedispersedSeries.h"
#include "dedispersion_manager.h"

// Pelican stuff
#include "pelican/data/DataBlob.h"

// C++ stuff
#include <iostream>
#include <complex>
#include "math.h"
#include <cstdlib>

#include <QString>

using namespace pelican;

// Constructor
MdsmModule::MdsmModule(const ConfigNode& config)
    : AbstractModule(config), _samples(0), _counter(0), _iteration(0)
{
    // Configure MDSM Module
    QString _filepath = config.getOption("observationfile", "filepath");
    _createOutputBlob = (bool) config.getOption("createOutputBlob", "value", "0").toUInt();

    // Process Survey parameters (through observation file)
    _survey = processSurveyParameters(_filepath);

    // Start up MDSM
    _input_buffer = initialiseMDSM(_survey);
}

// Destructor
MdsmModule::~MdsmModule()
{
    tearDownMDSM();
}

// Run MDSM
void MdsmModule::run(SpectrumDataSetStokes* streamData, DedispersedTimeSeriesF32* dedispersedData)
{
    unsigned nSamples, nSubbands, nChannels, reqSamp, copySamp;
    nSamples = streamData -> nTimeBlocks();
    nSubbands = streamData -> nSubbands();
    nChannels = (nSamples == 0) ? 0 : streamData->nChannels();
    float *data;

    // We need the timestamp of the first packet in the first blob (assuming that we don't
    // lose any packets), and the sampling time. This will give each sample a unique timestamp
    if (_samples == 0) {
        _timestamp = streamData -> getLofarTimestamp();
        _blockRate = streamData -> getBlockRate();
    }

    // Calculate number of required samples
    reqSamp = _counter == 0 ? _survey -> nsamp + _survey -> maxshift : _survey -> nsamp;

    // Check to see whether all samples will fit in memory
    copySamp = nSamples <= reqSamp - _samples ? nSamples : reqSamp - _samples;

    // Check if we reached the end of the stream, in which case we clear the MDSM buffers
    if (nSamples == 0) {
        reqSamp = 0;
        copySamp = 0;
    }

    // NOTE: we always care about the first Stokes parameter (XX)
    for(unsigned t = 0; t < copySamp; t++) {
        for (unsigned s = 0; s < nSubbands; s++) {
            data = streamData -> spectrumData(t, s, 0);
            for (unsigned c = 0; c < nChannels; c++) {
                _input_buffer[(_samples + t)* nSubbands * nChannels
                              + s * nChannels + c] = data[c];
            }
        }
    }
    _samples += copySamp;

    // We have enough samples to pass to MDSM
    if (_samples == reqSamp || nSamples == 0) {
        // Copy this chunk and get previous output
        unsigned int numSamp;
        unsigned samples;

        if (_counter == 0)
            numSamp = _samples - _survey -> maxshift;
        else
            numSamp = _samples;

        float *outputBuffer = next_chunk(numSamp, samples, _timestamp, _blockRate);

        if (outputBuffer != NULL && _createOutputBlob) {

            // Output available, create data blob
            dedispersedData -> resize(_survey -> tdms);
            if (_survey -> useBruteForce) {
                // All DMs have same number of samples
                DedispersedSeries<float>* data;
                for (unsigned d = 0; d < _survey -> tdms; d++) {
                    data  = dedispersedData -> samples(d);
                    data -> resize(samples);
                    data -> setDmValue(_survey -> lowdm + _survey -> dmstep * d);
                    for (unsigned xx = 0; xx < samples; xx++)
                    	(data -> ptr())[xx] = outputBuffer[d * samples + xx];
//                    memcpy(data -> ptr(), &outputBuffer[d * samples], samples * sizeof(float));
                }

            }
            else {
             ;	// Number of samples differs among passes
            }
        }
        else
            dedispersedData -> resize(0);

        // Tell MDSM to start processing this chunk
        if (!start_processing(_samples))  return;
        _counter++;
        _samples = 0;

        for(unsigned t = copySamp; t < nSamples; t++) {
            for (unsigned s = 0; s < nSubbands; s++) {
                data = streamData -> spectrumData(t, s, 0);
                for (unsigned c = 0; c < nChannels; c++)
                    _input_buffer[(t - copySamp) * nSubbands * nChannels
                                  + s * nChannels + c] = data[c];
            }
        }
        _samples += nSamples - copySamp;
    }
    else
        dedispersedData -> resize(0);

    _iteration++;
}

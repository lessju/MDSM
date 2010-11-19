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
    : AbstractModule(config), _samples(0), _gettime(0), _counter(0), _iteration(0)
{
    // Configure MDSM Module
    QString _filepath = config.getOption("observationfile", "filepath");
    _createOutputBlob = (bool) config.getOption("createOutputBlob", "value", "0").toUInt();
    _invertChannels = (bool) config.getOption("invertChannels", "value", "1").toUInt();

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
    // _blockRate currently contains the number of time samples per chunk... not very useful
    if (_gettime == 0) {
        _timestamp = streamData -> getLofarTimestamp();
        _blockRate = streamData -> getBlockRate();

        if (_counter > 0)
            _timestamp = streamData -> getLofarTimestamp() - _blockRate * _survey -> maxshift;
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
            data = streamData -> spectrumData(t, (_invertChannels) ? nSubbands - 1 - s : s, 0);
            for (unsigned c = 0; c < nChannels; c++)
//                _input_buffer[(_samples + t) * nSubbands * nChannels
//                              + s * nChannels + c] = data[(_invertChannels) ? nChannels - 1 - c : c];

                  // Corner turn first...
                if (_counter == 0)
                    _input_buffer[s * nChannels * (_survey -> nsamp + _survey -> maxshift)
                                  + c * (_survey -> nsamp + _survey -> maxshift)
                                  + (_samples + t)] = data[(_invertChannels) ? nChannels - 1 - c : c];
                else
                    _input_buffer[s * nChannels * (_survey -> nsamp + _survey -> maxshift)
                                  + c * (_survey -> nsamp + _survey -> maxshift)
                                  + _survey -> maxshift + _samples + t] = data[(_invertChannels) ? nChannels - 1 - c : c];

        }
    }
    _samples += copySamp;
    _gettime = _samples;

    // We have enough samples to pass to MDSM
    if (_samples == reqSamp || nSamples == 0) {
        // Copy this chunk and get previous output
        unsigned int numSamp;
        unsigned samples;

        numSamp = (_counter == 0) ? _samples - _survey -> maxshift : _samples;

        float *outputBuffer = next_chunk(numSamp, samples, _timestamp, _blockRate);

        // Copy remaining samples (if any) to MDSM input buffer
        _gettime = _samples;
        if (!start_processing(_samples))  return;

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
                    memcpy(data -> ptr(), &outputBuffer[d * samples], samples * sizeof(float));
                }
            }
            else {
                // Number of samples differs among passes
                DedispersedSeries<float>* data;
                unsigned totdms = 0, shift = 0; 
                for (unsigned thread = 0; thread < _survey -> num_threads; thread++) {
                    for(unsigned pass = 0; pass < _survey -> num_passes; pass++) {
                        unsigned ndms = (_survey -> pass_parameters[pass].ncalls / _survey -> num_threads) 
                                       * _survey -> pass_parameters[pass].calldms;

                        float startdm = _survey -> pass_parameters[pass].lowdm + _survey -> pass_parameters[pass].sub_dmstep 
                                        * (_survey -> pass_parameters[pass].ncalls / _survey -> num_threads) * thread;
                        float dmstep  = _survey -> pass_parameters[pass].dmstep;

                        unsigned nsamp = samples / _survey -> pass_parameters[pass].binsize;
                        for(unsigned dm = 0; dm < ndms; dm++) {
                            data = dedispersedData -> samples(totdms + dm);
                            data -> resize(nsamp);
                            data -> setDmValue(startdm + dm * dmstep);
                            memcpy(data -> ptr(), &outputBuffer[shift], nsamp * sizeof(float));
                            shift += nsamp;
                        }
                        totdms += ndms;
                    }   
                }
            }
        }
        else
            dedispersedData -> resize(0);

        _counter++;
        _samples = 0;
        _gettime = _samples;

        for(unsigned t = copySamp; t < nSamples; t++) {
            for (unsigned s = 0; s < nSubbands; s++) {
                data = streamData -> spectrumData(t, (_invertChannels) ? nSubbands - 1 - s : s, 0);
                for(unsigned c = 0 ; c < nChannels ; ++c)
//                      _input_buffer[(t - copySamp) * nSubbands * nChannels
//                                    + s * nChannels + c] = data[(_invertChannels) ? nChannels - 1 - c : c];

                  // Corner turn first...
                    _input_buffer[s * nChannels * (_survey -> nsamp + _survey -> maxshift)
                                + c * (_survey -> nsamp + _survey -> maxshift)
                                + _survey -> maxshift + _samples + t] = data[(_invertChannels) ? nChannels - 1 - c : c];

            }
        }
        _samples += nSamples - copySamp;
    }
    else
        dedispersedData -> resize(0);

    _iteration++;
}

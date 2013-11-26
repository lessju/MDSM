// MDSM stuff
#include "CoherentMdsmModule.h"
#include "coherent_dedispersion_manager.h"

// Pelican stuff
#include "pelican/data/DataBlob.h"
#include "WeightedSpectrumDataSet.h"

// C++ stuff
#include <iostream>
#include <complex>
#include "math.h"
#include <cstdlib>

#include <QString>

using namespace pelican;

// Constructor
CoherentMdsmModule::CoherentMdsmModule(const ConfigNode& config)
  : AbstractModule(config), _samples(0), _gettime(0), _counter(0), _iteration(0)
{
    // Configure MDSM Module
    QString _filepath = config.getOption("observationfile", "filepath");
    _invertChannels = (bool) config.getOption("invertChannels", "value", "1").toUInt();

    // Process Survey parameters (through observation file)
    _obs = processObservationParameters(_filepath);

    // Start up MDSM
    _input_buffer = initialiseMDSM(_obs);
}

// Destructor
CoherentMdsmModule::~CoherentMdsmModule()
{
    tearDownCoherentMDSM();
}

// Run MDSM
void CoherentMdsmModule::run(DataBlob* incoming)
{
    TimeSeriesDataSetC32* timeSeriesData;

    // If incoming data blob is of type WeightedSpectrumDataSet, then we will forward the mean and rms
    // value per chunk to MDSM. Get the mean and rms per chunk from the datablob
    if ( !(timeSeriesData = (TimeSeriesDataSetC32*) dynamic_cast<TimeSeriesDataSetC32*>(incoming))) 
        throw QString("MDSM:  No useful datablob");
    
    unsigned nSamples, nSubbands, nChannels, reqSamp, copySamp;
    nSamples = timeSeriesData -> nTimeBlocks();
    nSubbands = timeSeriesData -> nSubbands();
    nChannels = 1;
    Complex *data;

    // We need the timestamp of the first packet in the first blob (assuming that we don't
    // lose any packets), and the sampling time. This will give each sample a unique timestamp
    // _blockRate currently contains the number of time samples per chunk... [Aris: not very useful]
    if (_gettime == 0) {
        _timestamp = timeSeriesData -> getLofarTimestamp();
        _blockRate = timeSeriesData -> getBlockRate();

        if (_counter > 0)
            _timestamp = timeSeriesData -> getLofarTimestamp() - _blockRate * _obs -> wingLen;
    }

    // Calculate number of required samples
    reqSamp = _counter == 0 ? _obs -> gpuSamples : _obs -> gpuSamples - _obs -> wingLen;

    // Check to see whether all samples will fit in memory
    copySamp = nSamples <= reqSamp - _samples ? nSamples : reqSamp - _samples;
   
    // Check if we reached the end of the stream, in which case we clear the MDSM buffers
    if (nSamples == 0) { reqSamp = 0; copySamp = 0; }

    // NOTE: CURRENTL ONLY ASSUMES ONE POLARISATION (OR BEAM)
    // NOTE: THIS VERSION ASSUMES NO FURTHER CHANNELISATION
    data = timeSeriesData -> data();
    for(unsigned t = 0; t < copySamp; t++) 
        for (unsigned s = 0; s < nSubbands; s++) 
        {
            Complex datum = data[s * nSamples + t];

            // Corner turn whilst copying data to MDSM
            if (_counter == 0)
            {
                unsigned index = (_invertChannels) 
                                 ? ((nSubbands-1-s) * _obs -> gpuSamples + _samples + t) * 2
                                 : (s * _obs -> gpuSamples + _samples + t) * 2;
                _input_buffer[index] = datum.real();
                _input_buffer[index + 1] = datum.imag();
            }
            else
            {
                unsigned index = (_invertChannels) 
                                 ? ((nSubbands-1-s)*_obs -> gpuSamples+_obs -> wingLen + _samples + t) * 2
                                 : (s * _obs -> gpuSamples + _obs -> wingLen + _samples + t) * 2;
                _input_buffer[index]     = datum.real();
                _input_buffer[index + 1] = datum.imag();
            }
        }

    _samples += copySamp;
    _gettime = _samples;

    // We have enough samples to pass to MDSM
    if (_samples == reqSamp || nSamples == 0) {

        // Copy this chunk and get previous output
        unsigned int numSamp;
        unsigned samples;
        
        numSamp = (_counter == 0) ? _samples - _obs -> wingLen : _samples;

        // Notify MDSM of next processable chunk
        next_coherent_chunk(numSamp, samples, _timestamp, _blockRate);
        
        // Copy remaining samples (if any) to MDSM input buffer
        _gettime = _samples;
        if (!start_coherent_processing(_samples))  return;
      
        _counter++;
        _samples = 0;
        _gettime = _samples;

        data = timeSeriesData -> data();
        for(unsigned t = copySamp; t < nSamples; t++) 
            for (unsigned s = 0; s < nSubbands; s++) 
            {
                Complex datum = data[s * nSamples + t];

                // Corner turn whilst copying data to MDSM
                unsigned index = (_invertChannels)
                                 ? ((nSubbands-1-s)*_obs -> gpuSamples+_obs -> wingLen + _samples + t) * 2
                                 : (s * _obs -> gpuSamples + _obs -> wingLen + _samples + t) * 2;
                _input_buffer[index]     = datum.real();
                _input_buffer[index + 1] = datum.imag();
            }

        _samples += nSamples - copySamp;
    }

    _iteration++;
}

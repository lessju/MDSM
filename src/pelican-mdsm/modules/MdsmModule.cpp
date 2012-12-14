// MDSM stuff
#include "MdsmModule.h"
//#include "DedispersedSeries.h"
#include "dedispersion_manager.h"

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
void MdsmModule::run(DataBlob* incoming, DedispersedTimeSeriesF32* dedispersedData)
{
    SpectrumDataSetStokes*   spectrumData;
    WeightedSpectrumDataSet* weightedData;
    SpectrumDataSet<float>*  streamData;

    // Check incoming blob's datatype
    if ( (weightedData = (WeightedSpectrumDataSet*) dynamic_cast<WeightedSpectrumDataSet*>(incoming))) 
        streamData = weightedData -> dataSet();

    else if ((spectrumData = (SpectrumDataSetStokes*) dynamic_cast<SpectrumDataSetStokes*> (incoming)))
    {
        streamData = spectrumData;
        _survey -> samplesPerChunk = streamData -> nTimeBlocks();
    }
    else
        throw QString("MDSM: No useful datablob");
    
    unsigned nSamples, nSubbands, nChannels, copySamp;
    nSamples = streamData -> nTimeBlocks();
    nSubbands = streamData -> nSubbands();
    nChannels = (nSamples == 0) ? 0 : streamData -> nChannels();
    float *data;

    // We need the timestamp of the first packet in the first blob (assuming that we don't
    // lose any packets), and the sampling time. This will give each sample a unique timestamp
    // _blockRate currently contains the number of time samples per chunk... [Aris: not very useful]
    if (_gettime == 0) {
        _timestamp = streamData -> getLofarTimestamp();
        _blockRate = streamData -> getBlockRate();
    }

    // Check to see whether all samples will fit in memory
    copySamp = nSamples <= _survey -> nsamp - _samples ? nSamples : _survey -> nsamp - _samples;
   
    // Check if we reached the end of the stream, in which case we clear the MDSM buffers
    if (nSamples == 0) copySamp = 0;

    // NOTE: we always care about the first Stokes parameter (XX)
    for(unsigned t = 0; t < copySamp; t++) {
        for (unsigned s = 0; s < nSubbands; s++) {
            data = streamData -> spectrumData(t, (_invertChannels) ? nSubbands - 1 - s : s, 0);

            for (unsigned c = 0; c < nChannels; c++)
         
                // Corner turn whilst copying data to MDSM
                if (_counter == 0)
                    _input_buffer[s * nChannels * _survey -> nsamp + c * _survey -> nsamp
                                  + (_samples + t)] = data[(_invertChannels) ? nChannels - 1 - c : c];
                else
                    _input_buffer[s * nChannels * _survey -> nsamp + c * _survey -> nsamp
                                  + _samples + t] = data[(_invertChannels) ? nChannels - 1 - c : c];
        }
    }

    _samples += copySamp;
    _gettime = _samples;

    // We have enough samples to pass to MDSM
    if (_samples == _survey -> nsamp || nSamples == 0) {

        // Copy this chunk and get previous output
        unsigned samples;

        // Notify MDSM of next processable chunk
        float *outputBuffer = next_chunk(_survey -> nsamp, samples, _timestamp, _blockRate);

        // Copy remaining samples (if any) to MDSM input buffer
        _gettime = _samples;
        if (!start_processing(_samples))  return;

        if (outputBuffer != NULL && _createOutputBlob) {
        
            // Output available, create data blob
            dedispersedData -> resize(_survey -> tdms);

            // All DMs have same number of samples
            DedispersedSeries<float>* data;
            for (unsigned d = 0; d < _survey -> tdms; d++) {
                data  = dedispersedData -> samples(d);
                data -> resize(samples);
                data -> setDmValue(_survey -> lowdm + _survey -> dmstep * d);
                memcpy(data -> ptr(), &outputBuffer[d * samples], samples * sizeof(float));
            }
        }
        else
            dedispersedData -> resize(0);

        _counter++;
        _samples = 0;
        _gettime = _samples;

        // Copy remaining samples of last chunk into next block for processing
        for(unsigned t = copySamp; t < nSamples; t++) {
            for (unsigned s = 0; s < nSubbands; s++) {
                data = streamData -> spectrumData(t, (_invertChannels) ? nSubbands - 1 - s : s, 0);
                for(unsigned c = 0 ; c < nChannels ; ++c)
                    // Corner turn whilst copying data to 
                    _input_buffer[s * nChannels * _survey -> nsamp + c * _survey -> nsamp
                                  + _samples + t] = data[(_invertChannels) ? nChannels - 1 - c : c];
            }
        }
        _samples += nSamples - copySamp;
    }
    else
        dedispersedData -> resize(0);

    _iteration++;
}

// NOTE: If we do not want to corner turn when copying data to MDSM, use:
//    _input_buffer[(t - copySamp) * nSubbands * nChannels
//                   + s * nChannels + c] = data[(_invertChannels) ? nChannels - 1 - c : c];

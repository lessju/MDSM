// MDSM stuff
#include "MdsmModule.h"
#include "dedispersion_manager.h"

// Pelican stuff
#include "pelican/data/DataBlob.h"

// C++ stuff
#include <iostream>
#include <complex>
#include "math.h"

#include <QString>

// Constructor
MdsmModule::MdsmModule(const ConfigNode& config)
    : AbstractModule(config), _samples(0), _counter(0), _iteration(0)
{ 
    // Configure MDSM Module
	QString _filepath = config.getOption("observationfile", "filepath");

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
void MdsmModule::run(ChannelisedStreamData* streamData)
{
    unsigned i;

    // We need the timestamp of the first packet in the first blob (assuming that we don't
    // lose any packets), and the sampling time. This will give each sample a unique timestamp
    if (_samples == 0) {
        _timestamp = streamData -> getLofarTimestamp();
        _blockRate = streamData -> getBlockRate();
        printf("timestamp: %lld, blockRate: %ld\n", _timestamp, _blockRate);
    }   

    std::complex<double> *data = streamData -> data();
    for(i = 0; i < streamData -> size(); i++) {
        _input_buffer[_samples * streamData -> size() + i] = sqrt(pow(data -> real(), 2) + pow(data -> imag(), 2));
        data++;
    }
    _samples++;

    if ((_counter == 0 && (_samples == _survey -> nsamp + _survey -> maxshift))
        || _samples == _survey -> nsamp) {
        
        if (!process_chunk(_samples))  throw QString("MDSM error while processing chunk");
        _counter++;
        _samples = 0;
    }

    if (_samples % 1000 == 0) std::cout << "Received: " << _samples << "\tNeed: " << _survey -> nsamp + _survey -> maxshift << std::endl;
}

// Run MDSM - method overload
void MdsmModule::run(TimeStreamData* streamData)
{
    unsigned i;

    // We need the timestamp of the first packet in the first blob (assuming that we don't
    // lose any packets), and the sampling time. This will give each sample a unique timestamp
    if (_samples == 0) {
        _timestamp = streamData -> getLofarTimestamp();
        _blockRate = streamData -> getBlockRate();
    }   

    std::complex<double> *data = streamData -> data();

    // Calculate number of required samples
    unsigned int reqSamp = _counter == 0 ? _survey -> nsamp + _survey -> maxshift : _survey -> nsamp;
    
    // Check to see whether all samples will fit in memory
    unsigned int copySamp = streamData -> nSamples() <= reqSamp - _samples ? streamData -> nSamples() : reqSamp - _samples; 

    for(i = 0; i < copySamp * streamData -> nSubbands(); i++) {
        _input_buffer[_samples * streamData -> nSubbands() + i] = sqrt(pow(data -> real(), 2) + pow(data -> imag(), 2));
        data++;
    }
    _samples += copySamp;

    if (_samples == reqSamp) {
        if (!process_chunk(_samples))  throw QString("MDSM error while processing chunk");
        _counter++;
        _samples = 0;
    }

    // Check if there are leftover samples to copy
    for(i = 0; i < (streamData -> nSamples() - copySamp) * streamData -> nSubbands(); i++) {
        _input_buffer[i] = sqrt(pow(data -> real(), 2) + pow(data -> imag(), 2));
        data++;
    }

    if (_samples % 10000 == 0) std::cout << "Received: " << _samples << ", Need: " << _survey -> nsamp + _survey -> maxshift << std::endl;
}

// Run MDSM - method overload
void MdsmModule::run(SubbandSpectraStokes* streamData) 
{
    unsigned nSamples = streamData -> nTimeBlocks();
    unsigned nSubbands = streamData -> nSubbands();
    unsigned nChannels = streamData -> ptr(0,0,0) -> nChannels();

    // We need the timestamp of the first packet in the first blob (assuming that we don't
    // lose any packets), and the sampling time. This will give each sample a unique timestamp
    // NOTE: Need to be implemented in SubbandSpectraStokes
    if (_samples == 0) {
        _timestamp = streamData -> getLofarTimestamp();
        _blockRate = streamData -> getBlockRate();
    }  

    // Calculate number of required samples
    unsigned reqSamp = _counter == 0 ? _survey -> nsamp + _survey -> maxshift : _survey -> nsamp;
    
    // Check to see whether all samples will fit in memory
    unsigned int copySamp = nSamples <= reqSamp - _samples ? nSamples : reqSamp - _samples; 

    // NOTE: we always care about the first stokes parameter (XX)
    float *data;
    for(unsigned t = 0; t < copySamp; t++) {
        for (unsigned s = 0; s < nSubbands; s++) {
            data = streamData -> ptr(t, s, 0) -> ptr();
            for (unsigned c = 0; c < nChannels; c++)
                _input_buffer[_samples * nSubbands * nChannels 
                              + s * nChannels + c] = data[c];
        }
        _samples++;
    }

    if (_samples == reqSamp) {
        if (!process_chunk(_samples))  throw QString("MDSM error while processing chunk");
        _counter++;
        _samples = 0;
    }

    for(unsigned t = 0; t < nSamples - copySamp; t++) {
        for (unsigned s = 0; s < nSubbands; s++) {
            data = streamData -> ptr(t, s, 0) -> ptr();
            for (unsigned c = 0; c < nChannels; c++)
                _input_buffer[_samples * nSubbands * nChannels 
                              + s * nChannels + c] = data[c];
        }
        _samples++;
    }

    _iteration++;
    if (_iteration % 50 == 0) std::cout << "Received: " << _samples << ", Need: " << _survey -> nsamp + _survey -> maxshift << std::endl;
}

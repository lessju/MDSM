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
void MdsmModule::run(SubbandSpectraStokes* streamData) 
{
    unsigned nSamples = streamData -> nTimeBlocks();
    unsigned nSubbands = streamData -> nSubbands();
    unsigned nChannels = streamData -> ptr(0,0,0) -> nChannels();

    // We need the timestamp of the first packet in the first blob (assuming that we don't
    // lose any packets), and the sampling time. This will give each sample a unique timestamp
    if (_samples == 0) {
        _timestamp = streamData -> getLofarTimestamp();
        _blockRate = streamData -> getBlockRate();
    }  

    // Calculate number of required samples
    unsigned reqSamp = _counter == 0 ? _survey -> nsamp + _survey -> maxshift : _survey -> nsamp;
    
    // Check to see whether all samples will fit in memory
    unsigned int copySamp = nSamples <= reqSamp - _samples ? nSamples : reqSamp - _samples;

    // NOTE: we always care about the first Stokes parameter (XX)
    float *data;
    for(unsigned t = 0; t < copySamp; t++) {
        for (unsigned s = 0; s < nSubbands; s++) {
            data = streamData -> ptr(t, s, 0) -> ptr();
            for (unsigned c = 0; c < nChannels; c++) {
                _input_buffer[(_samples + t)* nSubbands * nChannels
                              + s * nChannels + c] = data[nChannels - c - 1];
//                std::cout << data[nChannels - c- 1] << " ";
            }
//            	std::cout << std::endl;
        }
        _samples++;
    }

    if (_samples == reqSamp) {
        if (!process_chunk(_samples, _timestamp, _blockRate))  throw QString("MDSM error while processing chunk");
        _counter++;
        _samples = 0;

		for(unsigned t = 0; t < nSamples - copySamp; t++) {
			for (unsigned s = 0; s < nSubbands; s++) {
				data = streamData -> ptr(t, s, 0) -> ptr();
				for (unsigned c = 0; c < nChannels; c++)
					_input_buffer[(_samples + t) * nSubbands * nChannels
								  + s * nChannels + c] = data[c];
			}
			_samples++;
		}
    }

    _iteration++;
//    if ((reqSamp / _samples) % 1 == 0.1) std::cout << "Received: " << _samples << ", Need: " << _survey -> nsamp + _survey -> maxshift << std::endl;
}

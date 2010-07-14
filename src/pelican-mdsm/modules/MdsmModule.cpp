// MDSM stuff
#include "MdsmModule.h"
#include "dedispersion_manager.h"

// Pelican stuff
#include "pelican/data/DataBlob.h"

// PelicanLofar stuff
#include "ChannelisedStreamData.h"

// C++ stuff
#include <iostream>
#include <complex>
#include "math.h"

#include <QString>

// Constructor
MdsmModule::MdsmModule(const ConfigNode& config)
    : AbstractModule(config), _samples(0), _counter(0)
{ 
    // Configure MDSM Module
	QString _filepath = config.getOption("observationfile", "filepath");
    _filepath = QString("/home/lessju/Code/MDSM/build/pelican-mdsm/pipelines/data/obs.xml");

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


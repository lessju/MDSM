#include "MdsmPipeline.h"
#include <iostream>

MdsmPipeline::MdsmPipeline()
    : AbstractPipeline()
{
    _iteration = 0;
}

MdsmPipeline::~MdsmPipeline()
{ }

// Initialise the pipeline
void MdsmPipeline::init()
{
    // Create modules
    mdsm = (MdsmModule *) createModule("MdsmModule");
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");

    // Create local datablobs
    spectra = (SubbandSpectraC32*) createBlob("SubbandSpectraC32");
    stokes = (SubbandSpectraStokes*) createBlob("SubbandSpectraStokes");

    // Request remote data
    requestRemoteData("SubbandTimeSeriesC32");
}

// Run the pipeline
void MdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    timeSeries = (SubbandTimeSeriesC32*) remoteData["SubbandTimeSeriesC32"];

    if (timeSeries -> size() == 0) {
    	std::cout << "Reached end of file" << std::endl;
    	// TODO: make mdsm process this batch...
    }

    // Run modules
    ppfChanneliser->run(timeSeries, spectra);
    stokesGenerator->run(spectra, stokes);
    mdsm->run(stokes);

    // Output channelised data
//    dataOutput(stokes, "SubbandSpectraStokes");

    if (_iteration % 10000 == 0) std::cout << "Iteration: " << _iteration << std::endl;
    _iteration++;
}

#include "AtaMdsmPipeline.h"
#include <iostream>

AtaMdsmPipeline::AtaMdsmPipeline()
    : AbstractPipeline()
{
    _iteration = 0;
}

AtaMdsmPipeline::~AtaMdsmPipeline()
{ }

// Initialise the pipeline
void AtaMdsmPipeline::init()
{
    // Create modules
    mdsm = (MdsmModule *) createModule("MdsmModule");
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");
    rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    dedispersedData = (DedispersedTimeSeriesF32*) createBlob("DedispersedTimeSeriesF32");

    // Request remote data
    requestRemoteData("TimeSeriesDataSetC32");
}

// Run the pipeline
void AtaMdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    timeSeries = (TimeSeriesDataSetC32*) remoteData["TimeSeriesDataSetC32"];

    if (timeSeries -> nTimeBlocks() == 0) {
        std::cout << "Reached end of file" << std::endl;
        for (unsigned i = 0; i < 2; i++) { // NOTE: Too dependent on MDSM's internal state
            std::cout << "Processing extra step " << i << std::endl;
            mdsm->run(stokes, dedispersedData);
//            dataOutput(dedispersedData, "DedispersedTimeSeriesF32");
            stop();
        }
    }

    // Run modules
    ppfChanneliser->run(timeSeries, spectra);

    stokesGenerator->run(spectra, stokes);
    rfiClipper->run(stokes);

    // Output stokes data
//    dataOutput(stokes, "SpectrumDataSetStokes");

    mdsm->run(stokes, dedispersedData);

    dataOutput(dedispersedData, "DedispersedTimeSeriesF32");

    if (_iteration % 50000 == 0)
    std::cout << "Iteration: " << _iteration << std::endl;
    _iteration++;
}

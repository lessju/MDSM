#include "MdsmPipeline.h"
#include "DedispersedDataWriter.h"
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
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    dedispersedData = (DedispersedTimeSeriesF32*) createBlob("DedispersedTimeSeriesF32");

    // Request remote data
    requestRemoteData("TimeSeriesDataSetC32");
}

// Run the pipeline
void MdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    timeSeries = (TimeSeriesDataSetC32*) remoteData["TimeSeriesDataSetC32"];

    if (timeSeries -> size() == 0) {
        std::cout << "Reached end of stream" << std::endl;
        for (unsigned i = 0; i < 2; i++) { // NOTE: Too dependent on MDSM's internal state
            std::cout << "Processing extra step " << i << std::endl;
            mdsm->run(stokes, dedispersedData);
            dataOutput(dedispersedData, "DedispersedTimeSeriesF32");
            stop();
        }
    }

    // Run modules
    ppfChanneliser->run(timeSeries, spectra);
    stokesGenerator->run(spectra, stokes);
    mdsm->run(stokes, dedispersedData);

    // Output channelised data
    //    dataOutput(stokes, "DedispersedDataWriter");

    dataOutput(dedispersedData, "DedispersedTimeSeriesF32");

    _iteration++;
}

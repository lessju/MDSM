#include "GuppiMdsmPipeline.h"
#include <iostream>

GuppiMdsmPipeline::GuppiMdsmPipeline()
    : AbstractPipeline()
{
    _iteration = 0;
}

GuppiMdsmPipeline::~GuppiMdsmPipeline()
{ }

// Initialise the pipeline
void GuppiMdsmPipeline::init()
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
void GuppiMdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    timeSeriesData = (TimeSeriesDataSetC32*) remoteData["TimeSeriesDataSetC32"];

    if (timeSeriesData -> nTimeBlocks() == 0)
    {
        std::cout << "Reached the end of file: " << _iteration << std::endl;

        for (unsigned i = 0; i < 2; i++) {
            std::cout << "Processing extra step " << i << std::endl;
            mdsm -> run(stokes, dedispersedData);
            dataOutput(dedispersedData, "DedispersedTimeSeriesF32");
        }
        stop();
        std::cout << "Should have ended here!" << std::endl;
    }

    // Run modules
    ppfChanneliser -> run(timeSeriesData, spectra);
    stokesGenerator -> run(spectra, stokes);
    mdsm -> run(stokes, dedispersedData);

    // Output dedispersed data
//    dataOutput(dedispersedData, "DedispersedTimeSeriesF32");

    _iteration++;
}

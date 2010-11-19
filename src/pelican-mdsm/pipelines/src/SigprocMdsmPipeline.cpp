#include "SigprocMdsmPipeline.h"
#include <iostream>

SigprocMdsmPipeline::SigprocMdsmPipeline()
    : AbstractPipeline()
{
    _iteration = 0;
}

SigprocMdsmPipeline::~SigprocMdsmPipeline()
{ }

// Initialise the pipeline
void SigprocMdsmPipeline::init()
{
    // Create modules
    mdsm = (MdsmModule *) createModule("MdsmModule");

    // Create local datablobs
    dedispersedData = (DedispersedTimeSeriesF32*) createBlob("DedispersedTimeSeriesF32");

    // Request remote data
    requestRemoteData("SpectrumDataSetStokes");
}

// Run the pipeline
void SigprocMdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    stokes = (SpectrumDataSetStokes*) remoteData["SpectrumDataSetStokes"];

    if (stokes -> nSpectra() == 0) {
        std::cout << "Reached end of file" << std::endl;
        for (unsigned i = 0; i < 2; i++) { // NOTE: Too dependent on MDSM's internal state
            std::cout << "Processing extra step " << i << std::endl;
            mdsm->run(stokes, dedispersedData);
            dataOutput(dedispersedData, "DedispersedTimeSeriesF32");
            stop();
        }
    }

    // Run modules
    mdsm->run(stokes, dedispersedData);

    // Output channelised data
 //   dataOutput(dedispersedData, "DedispersedTimeSeriesF32");

    _iteration++;
}

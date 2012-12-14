#include "MedicinaCoherentPipeline.h"
#include <iostream>

MedicinaCoherentPipeline::MedicinaCoherentPipeline()
    : AbstractPipeline(), _iteration(0) { }

MedicinaCoherentPipeline::~MedicinaCoherentPipeline()
{ }

// Initialise the pipeline
void MedicinaCoherentPipeline::init()
{
    // Create modules
//    mdsm = (CoherentMdsmModule *) createModule("CoherentMdsmModule");

    // Request remote data
    requestRemoteData("TimeSeriesDataSetC32");
}

// Run the pipeline
void MedicinaCoherentPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    printf("Even more yeah\n");
    // Get pointer to the remote TimeStreamData data blob
//    std::cout << "Asking for data blob" << std::endl;
    timeSeriesData = (TimeSeriesDataSetC32*) remoteData["TimeSeriesDataSetC32"];
//    std::cout << "got a data blob" << std::endl;

    // Run modules
    // mdsm -> run(timeSeriesData);

    if (_iteration % 100 == 99)
        std::cout << "Received 100 data blobs" << std::endl;

    _iteration++;
}

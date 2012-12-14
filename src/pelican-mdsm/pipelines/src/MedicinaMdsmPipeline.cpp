#include "MedicinaMdsmPipeline.h"
#include <iostream>

MedicinaMdsmPipeline::MedicinaMdsmPipeline()
    : AbstractPipeline()
{
    _iteration = 0;
}

MedicinaMdsmPipeline::~MedicinaMdsmPipeline()
{ }

// Initialise the pipeline
void MedicinaMdsmPipeline::init()
{
    // Create modules
//    mdsm = (MdsmModule *) createModule("MdsmModule");

    // Create local datablobs

    // Request remote data
    requestRemoteData("MultiBeamTimeSeriesDataSetC32");
}

// Run the pipeline
void MedicinaMdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    printf("Pipeline iteration\n");
    timeSeriesData = (MedicinaStream*) remoteData["MultiBeamTimeSeriesDataSetC32"];

    if (_iteration++ % 1000 == 999)
        std::cout << "Processed 1000 iterations" << std::endl;
}

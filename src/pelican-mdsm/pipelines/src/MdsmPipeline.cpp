#include "MdsmPipeline.h"
#include "AdapterTimeStream.h"
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

    // Request remote data
    requestRemoteData("TimeStreamData");
}

// Run the pipeline
void MdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    TimeStreamData* timeData = (TimeStreamData *) remoteData["TimeStreamData"];

    // Run MDSM
    mdsm -> run(timeData);

    // Output channelised data
    dataOutput( timeData, "TimeStreamData" );

//    if (_iteration % 1000 == 0) std::cout << "Iteration: " << _iteration << std::endl;
    _iteration++;
}

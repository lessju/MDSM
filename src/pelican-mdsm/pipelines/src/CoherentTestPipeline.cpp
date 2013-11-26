#include "CoherentTestPipeline.h"
#include <iostream>
#include <complex>

CoherentTestPipeline::CoherentTestPipeline()
    : AbstractPipeline()
{
    _iteration = 0;
}

CoherentTestPipeline::~CoherentTestPipeline()
{ }

// Initialise the pipeline
void CoherentTestPipeline::init()
{
    // Create modules
//    mdsm = (CoherentMdsmModule *) createModule("CoherentMdsmModule");

    // Request remote data
    requestRemoteData("TimeSeriesDataSetC32");
}

// Run the pipeline
void CoherentTestPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    timeData = (TimeSeriesDataSetC32*) remoteData["TimeSeriesDataSetC32"];

//    mdsm->run(timeData);
}

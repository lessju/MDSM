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

void MdsmPipeline::init()
{
    // Create modules
    channeliser = (ChanneliserPolyphase *) createModule("ChanneliserPolyphase");
    mdsm = (MdsmModule *) createModule("MdsmModule");

    // Create local datablobs
    polyphaseCoeff = (PolyphaseCoefficients*) createBlob("PolyphaseCoefficients");
    channelisedData = (ChannelisedStreamData*) createBlob("ChannelisedStreamData");

    // Hard-code filename, taps and channels.
    // FIXME These are quick hard-coded hacks at the moment.
    QString coeffFileName = "/home/lessju/Code/MDSM/src/pelican-mdsm/pipelines/data/coeffs_512_1.dat";
    int nTaps = 8;
    int nChannels = 512;
    polyphaseCoeff->load(coeffFileName, nTaps, nChannels);

    // Request remote data
    requestRemoteData("TimeStreamData");
}

/**
 * @details
 * Runs the pipeline.
 */
void MdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    TimeStreamData* timeData = (TimeStreamData *) remoteData["TimeStreamData"];

    // Run the polyphase channeliser.
    channeliser -> run(timeData, polyphaseCoeff, channelisedData);

    // Run the polyphase channeliser.
    mdsm -> run(channelisedData);

    _iteration++;
}

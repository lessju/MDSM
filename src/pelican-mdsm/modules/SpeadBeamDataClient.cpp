#include "SpeadBeamDataClient.h"
#include "SpeadBeamChunker.h"
#include "MultiBeamTimeSeriesDataSet.h"

namespace pelican {

namespace lofar {

/**
 *@details SpeadBeamDataClient
 */
SpeadBeamDataClient::SpeadBeamDataClient(const ConfigNode& configNode,
        const DataTypes& types, const Config* config)
: DirectStreamDataClient(configNode, types, config)
{
    addChunker("MultiBeamTimeSeriesDataSetC32", "SpeadBeamChunker");
}


/**
 *@details
 */
SpeadBeamDataClient::~SpeadBeamDataClient() { }


} // namespace lofar
} // namespace pelican

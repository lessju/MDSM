#include "SpeadBeamDataClient.h"
#include "SpeadBeamChunker.h"

namespace pelican {

namespace lofar {

/**
 *@details SpeadBeamDataClient
 */
SpeadBeamDataClient::SpeadBeamDataClient(const ConfigNode& configNode,
        const DataTypes& types, const Config* config)
: DirectStreamDataClient(configNode, types, config)
{
    addChunker("TimeSeriesDataSetC32", "SpeadBeamChunker");
}


/**
 *@details
 */
SpeadBeamDataClient::~SpeadBeamDataClient() { }


} // namespace lofar
} // namespace pelican

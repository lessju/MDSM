#include "MedicinaStreamDataClient.h"
#include "LofarChunker.h"

#include "pelican/utility/memCheck.h"


/**
 *@details LofarStreamDataClient
 */
MedicinaStreamDataClient::MedicinaStreamDataClient(const ConfigNode& configNode,
        const DataTypes& types, const Config* config)
: DirectStreamDataClient(configNode, types, config)
{
    addChunker( "SubbandTimeSeriesC32", "MedicinaChunker" );
}

/**
 *@details
 */
MedicinaStreamDataClient::~MedicinaStreamDataClient()
{
}


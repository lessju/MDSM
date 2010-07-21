#ifndef LOFARSTREAMDATACLIENT_H
#define LOFARSTREAMDATACLIENT_H

#include "pelican/core/DirectStreamDataClient.h"
#include "LofarTypes.h"

/**
 * @file LofarStreamDataClient.h
 */

using namespace pelican;
using namespace pelican::lofar;

/**
 * @class LofarStreamDataClient
 *
 * @ingroup pelican_lofar
 *
 * @brief
 *    Lofar Data Client to Connect directly to the LOFAR station
 *    output stream.
 *
 * @details
 *
 */

class MedicinaStreamDataClient : public DirectStreamDataClient
{
    public:
        MedicinaStreamDataClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config);
        ~MedicinaStreamDataClient();

    private:
};

PELICAN_DECLARE_CLIENT(MedicinaStreamDataClient)

#endif // LOFARSTREAMDATACLIENT_H

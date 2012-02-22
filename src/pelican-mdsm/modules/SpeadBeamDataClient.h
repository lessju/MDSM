#ifndef SPEADBEAMDATACLIENT_H
#define SPEADBEAMDATACLIENT_H

#include "pelican/core/DirectStreamDataClient.h"

/**
 * @file SpeadBeamDataClient.h
 */

namespace pelican {
namespace lofar {

/**
 * @class SpeadBeamDataClient
 *
 * @ingroup pelican_lofar
 *
 * @brief
 *    Lofar data client to connect directly to the LOFAR station
 *    output stream.
 *
 * @details
 *
 */

class SpeadBeamDataClient : public DirectStreamDataClient
{
    public:
        SpeadBeamDataClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config);
        ~SpeadBeamDataClient();

    private:
};

PELICAN_DECLARE_CLIENT(SpeadBeamDataClient)

} // namespace lofar
} // namespace pelican

#endif // SPEADBEAMDATACLIENT_H

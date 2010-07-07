#ifndef MDSM_MODULE_H
#define MDSM_MODULE_H

/**
 * @file 
 */

#include "pelican/modules/AbstractModule.h"
#include "ChannelisedStreamData.h"

/**
 * @class MdsmModule
 *
 * @brief
 * Module to warp MDSM functionality
 *
 */

class MdsmModule : public AbstractModule
{
	public:
        /// Constructs the channeliser module.
        MdsmModule(const ConfigNode& config) {}

        /// Destroys the channeliser module.
        ~MdsmModule() {}

        /// Method converting the time stream to a spectrum.
        void run(const ChannelisedStreamData* timeData) {}
};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(MdsmModule)

#endif // MDSM_MODULE_H_

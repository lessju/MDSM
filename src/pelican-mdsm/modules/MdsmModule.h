#ifndef MDSM_MODULE_H
#define MDSM_MODULE_H

#include "pelican/modules/AbstractModule.h"
#include "ChannelisedStreamData.h"
#include "survey.h"

using namespace pelican;
using namespace pelican::lofar;

class MdsmModule : public AbstractModule
{
	public:
        /// Constructs the channeliser module.
        MdsmModule(const ConfigNode& config);

        /// Destroys the channeliser module.
        ~MdsmModule();

        /// Method converting the time stream to a spectrum.
        void run(ChannelisedStreamData* timeData);

    private:
        SURVEY       *_survey;
        float        *_input_buffer;
        unsigned int _samples;
        unsigned int _counter;
        long long    _timestamp;
        long         _blockRate;
};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(MdsmModule)

#endif // MDSM_MODULE_H_

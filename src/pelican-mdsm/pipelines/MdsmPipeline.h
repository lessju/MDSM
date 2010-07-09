#ifndef MDSMPIPELINE_H
#define MDSMPIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "ChanneliserPolyphase.h"
#include "PolyphaseCoefficients.h"
#include "ChannelisedStreamData.h"
#include "TimeStreamData.h"
#include "MdsmModule.h"

using namespace pelican;
using namespace pelican::lofar;

class MdsmPipeline : public AbstractPipeline
{
    public:
        MdsmPipeline();
        ~MdsmPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers
        ChanneliserPolyphase* channeliser;
        MdsmModule*           mdsm;

        /// Local data blob
        PolyphaseCoefficients* polyphaseCoeff;
        ChannelisedStreamData* channelisedData;

        unsigned _iteration;
};

#endif // MDSMPIPELINE_H 

#ifndef MDSM_MODULE_H
#define MDSM_MODULE_H
// The following is related to the number of data blocks that MDSM is
// processing at any moment: 3 comes from 1 for input, 1 for processing,
// 1 for output
#define MDSM_STAGES 3 

#include "pelican/modules/AbstractModule.h"
#include "DedispersedTimeSeries.h"
#include "SpectrumDataSet.h"
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

        /// Perform Dedispersion;
        void run(DataBlob* timeData, DedispersedTimeSeriesF32* dedispersedData);

    private:
        SURVEY       *_survey;
        bool		 _createOutputBlob;
        float        *_input_buffer;
        unsigned int _samples;
        unsigned int _gettime;
        unsigned int _counter;
        double       _timestamp, _blockRate;
        long         _iteration;
        bool         _invertChannels;
};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(MdsmModule)

#endif // MDSM_MODULE_H_

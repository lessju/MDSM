#ifndef MDSM_MODULE_H
#define MDSM_MODULE_H

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
        void run(SpectrumDataSetStokes* timeData, DedispersedTimeSeriesF32* dedispersedData);

    private:
        SURVEY       *_survey;
        bool		 _createOutputBlob;
        float        *_input_buffer;
        unsigned int _samples;
        unsigned int _gettime;
        unsigned int _counter;
        long long    _timestamp;
        long         _blockRate;
        long         _iteration;
};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(MdsmModule)

#endif // MDSM_MODULE_H_

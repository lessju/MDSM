#ifndef COHERENT_MDSM_MODULE_H
#define COHERENT_MDSM_MODULE_H

#include "pelican/modules/AbstractModule.h"
#include "TimeSeriesDataSet.h"
#include "observation.h"

using namespace pelican;
using namespace pelican::lofar;

class CoherentMdsmModule : public AbstractModule
{
    private:
        typedef float Real;
        typedef std::complex<Real> Complex;

    public:
        /// Constructs the channeliser module.
        CoherentMdsmModule(const ConfigNode& config);

        /// Destroys the channeliser module.
        ~CoherentMdsmModule();

        /// Perform Dedispersion;
        void run(DataBlob* channelisedData);

    private:
        OBSERVATION  *_obs;
        float        *_input_buffer;
        unsigned int _samples;
        unsigned int _gettime;
        unsigned int _counter;
        double       _timestamp, _blockRate;
        long         _iteration;
        bool         _invertChannels;
};

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(CoherentMdsmModule)

#endif // COHERENT_MDSM_MODULE_H

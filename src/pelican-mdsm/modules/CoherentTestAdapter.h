#ifndef CoherentTestAdapter_H
#define CoherentTestAdapter_H

#include "pelican/core/AbstractStreamAdapter.h"
#include "TimeSeriesDataSet.h"
#include <complex>

using namespace pelican;
using namespace pelican::lofar;

class CoherentTestAdapter: public AbstractStreamAdapter
{
    public:
        /// Constructs a new SigprocAdapter.
        CoherentTestAdapter(const ConfigNode& config);

        /// Destroys the SigprocAdapter.
        ~CoherentTestAdapter() {}

    protected:
        /// Method to deserialise a sigproc file
        void deserialise(QIODevice* in);

    private:
        /// Updates and checks the size of the time stream data.
        void _checkData();

    private:
        TimeSeriesDataSetC32* _timeData;
        FILE *_fp;

        unsigned _nSamples;
        unsigned _nSubbands;
        double   _tsamp;
        unsigned long int _iteration;
};

PELICAN_DECLARE_ADAPTER(CoherentTestAdapter)

#endif // CoherentTestAdapter_H

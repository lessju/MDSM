#ifndef RawVoltageAdapter_H
#define RawVoltageAdapter_H

#include "pelican/core/AbstractStreamAdapter.h"
#include "TimeSeriesDataSet.h"
#include <complex>

using namespace pelican;
using namespace pelican::lofar;


class RawVoltageAdapter: public AbstractStreamAdapter
{
    public:
        /// Constructs a new RawVoltageAdapter.
        RawVoltageAdapter(const ConfigNode& config);

        /// Destroys the RawVoltageAdapter.
        ~RawVoltageAdapter() {}

    protected:
        /// Method to deserialise a sigproc file
        void deserialise(QIODevice* in);

    private:
        /// Updates and checks the size of the time stream data.
        void _checkData();

    private:
        TimeSeriesDataSetC32* _timeData;
        unsigned _nSamples;
        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nBits;
        unsigned long int _iteration;
};

PELICAN_DECLARE_ADAPTER(RawVoltageAdapter)

#endif // RawVoltageAdapter_H
